import gc
import numpy as np
import polars as pl
import sklearn.model_selection as ms
import torch
import torch.utils.data as data
import torch_geometric.data as tg_data
import torch_geometric.utils as tg_utils
import torch_geometric.loader as tg_loader
from src.magicdl import magic


class MagicStereoDataset(data.Dataset):
    def __init__(self, m1, m2, y, train_min, train_max, epsilon=1e-4, num_pixels=1039):
        assert m1.shape[0] == m2.shape[0] == y.shape[0], "Mismatch in number of samples between M1, M2 and labels"
        
        m1 = m1.astype(np.float32)[:, :num_pixels]
        m2 = m2.astype(np.float32)[:, :num_pixels]

        images_m1 = torch.from_numpy(m1).float()
        images_m2 = torch.from_numpy(m2).float() 
        labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        intensity_min = train_min
        intensity_span = train_max - train_min + epsilon
        sqrt_min = 2.0 * np.sqrt(0.375)
        sqrt_span = 2.0 * np.sqrt(max(0.0, train_max) + 0.375) - sqrt_min + epsilon
        background_intensity = 0.0

        num_pixels = 1039  # Anzahl der gültigen Pixel pro Bild (ohne Padding)
        global_edge_index, global_pos = compute_camera_topology(num_valid_pixels=num_pixels)

        # 1. Globale Feature-Berechnung (Vektorisierung)
        # Anstatt Clipping und Wurzelziehen pro Graph durchzuführen,
        # geschieht dies hochoptimiert für den gesamten Datensatz auf einmal.
        m1_clipped = torch.clamp(images_m1, min=0.0)
        m2_clipped = torch.clamp(images_m2, min=0.0)

        sqrt_m1 = 2 * torch.sqrt(m1_clipped + 0.375)
        sqrt_m2 = 2 * torch.sqrt(m2_clipped + 0.375)

        # Erstellung des globalen Feature-Tensors der Form [N, Num_Pixels, 4]
        # all_features = torch.stack([images_m1, images_m2, sqrt_m1, sqrt_m2], dim=-1)
        all_features = torch.stack([sqrt_m1, sqrt_m2], dim=-1)  # Nur die nicht-linearen Features verwenden, da sie besser skaliert sind

        # Globale Normalisierung
        # all_features[:, :, :2] = (all_features[:, :, :2] - intensity_min) / intensity_span
        # all_features[:, :, 2:4] = (all_features[:, :, 2:4] - sqrt_min) / (sqrt_span + epsilon)
        all_features = (all_features - sqrt_min) / (sqrt_span + epsilon)

        # 2. Graphen-Vorausberechnung (Caching)
        self.graphs = []
        num_samples = images_m1.shape[0]

        for idx in range(num_samples):
            m1_img = images_m1[idx]
            m2_img = images_m2[idx]

            mask = (m1_img.abs() > background_intensity) | (m2_img.abs() > background_intensity)
            if not mask.any():
                max_idx = (m1_img.abs() + m2_img.abs()).argmax()
                mask[max_idx] = True

            active_nodes = mask.nonzero(as_tuple=False).view(-1)

            # Subgraph-Extraktion ist die teuerste Operation. Einmalig in __init__ ausführen.
            edge_index, _ = tg_utils.subgraph(active_nodes, global_edge_index, relabel_nodes=True)

            x_sparse = all_features[idx][mask]
            pos_sparse = global_pos[mask]

            self.graphs.append(tg_data.Data(
                x=x_sparse,
                edge_index=edge_index,
                pos=pos_sparse,
                y=labels[idx]
            ))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_label_name(self, label):
        val = label.item() if isinstance(label, torch.Tensor) else label
        return {0: "Proton", 1: "Gamma"}.get(val, "Unknown")

def compute_camera_topology(num_valid_pixels = 1039):
    camera = magic.Camera()
    sources = []
    targets = []
    positions = []

    for i in range(num_valid_pixels):
        # Topologie (Kanten)
        neighbors = camera.get_pixel_neighbors(i)
        for j in neighbors:
            if j < num_valid_pixels:  # Verhindert Kanten zu weggepaddeten Pixeln
                sources.append(i)
                targets.append(j)

        # Geometrie (Node Positions)
        positions.append(camera.get_pixel_coordinates_xy(i))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)

    return edge_index, pos

def load_stereo_clean_images(num_samples=None, gamma_file="./data/magic-gammas-chunked.parquet", proton_file="./data/magic-protons-chunked.parquet", num_pixels=1039):

    def extract_images(file_path):
        # Lazy API: Schneide die Listen ab und caste sie VOR dem collect in Float32
        ds = pl.scan_parquet(file_path).select([
            pl.col("clean_image_m1").list.slice(0, num_pixels).cast(pl.List(pl.Float32)),
            pl.col("clean_image_m2").list.slice(0, num_pixels).cast(pl.List(pl.Float32))
        ])

        if num_samples:
            ds = ds.head(num_samples)

        df = ds.collect()

        # Kritischer Fehler im vorherigen Code behoben:
        # np.vstack auf Listen-Spalten erzeugt massiven Python-Object-Overhead.
        # Explode erzeugt stattdessen einen einzigen, kontinuierlichen Speicherblock.
        m1 = df["clean_image_m1"].list.explode().to_numpy().reshape(-1, num_pixels)
        m2 = df["clean_image_m2"].list.explode().to_numpy().reshape(-1, num_pixels)

        del df
        gc.collect()
        return m1, m2

    try:
        p_m1, p_m2 = extract_images(proton_file)
        g_m1, g_m2 = extract_images(gamma_file)
        return p_m1, p_m2, g_m1, g_m2

    except Exception as e:
        print(f"Fehler: {e}")
        return None

def preprocess_images(p_m1, p_m2, g_m1, g_m2, train_split=0.7, create_HPO_set=False, hpo_fraction=0.4):
    x_m1 = np.vstack((p_m1, g_m1))
    x_m2 = np.vstack((p_m2, g_m2))
    y = np.concatenate((np.zeros(p_m1.shape[0]), np.ones(g_m1.shape[0])), axis=0)

    x_m1_train, x_m1_rem, x_m2_train, x_m2_rem, y_train, y_rem = ms.train_test_split(
        x_m1, x_m2, y, test_size=1-train_split, random_state=0, stratify=y, shuffle=True
    )

    x_m1_val, x_m1_test, x_m2_val, x_m2_test, y_val, y_test = ms.train_test_split(
        x_m1_rem, x_m2_rem, y_rem, test_size=0.5, random_state=0, stratify=y_rem, shuffle=True
    )
    
    if create_HPO_set:
        _, x_m1_hpo_pool, _, x_m2_hpo_pool, _, y_hpo_pool = ms.train_test_split(
            x_m1_train, x_m2_train, y_train, test_size=hpo_fraction, random_state=0, stratify=y_train, shuffle=True
        )

        # 4. HPO-Pool 50/50 in Train und Val aufteilen
        x_m1_hpo_train, x_m1_hpo_val, x_m2_hpo_train, x_m2_hpo_val, y_hpo_train, y_hpo_val = ms.train_test_split(
            x_m1_hpo_pool, x_m2_hpo_pool, y_hpo_pool, test_size=0.5, random_state=0, stratify=y_hpo_pool, shuffle=True
        )
    
    if create_HPO_set: 
        return x_m1_train, x_m1_val, x_m1_test, x_m2_train, x_m2_val, x_m2_test, y_train, y_val, y_test, x_m1_hpo_train, x_m1_hpo_val, x_m2_hpo_train, x_m2_hpo_val, y_hpo_train, y_hpo_val
    else:
        return x_m1_train, x_m1_val, x_m1_test, x_m2_train, x_m2_val, x_m2_test, y_train, y_val, y_test


def get_stereo_clean_dataset(num_samples=10000, batch_size=128, train_split=0.7, return_HPO_subset=False, fraction_for_hpo=0.4):
    print("Loading and preprocessing data...")
    protons_m1, protons_m2, gammas_m1, gammas_m2 = load_stereo_clean_images(num_samples)
    gc.collect()
    print("Data loaded. Preprocessing...")
    # m1_train, m1_test, m1_val, m2_train, m2_test, m2_val, y_train, y_test, y_val = preprocess_images(protons_m1, protons_m2, gammas_m1, gammas_m2, )
    m1_train, m1_val, m1_test, m2_train, m2_val, m2_test, y_train, y_val, y_test, hpo_m1_train, hpo_m1_val, hpo_m2_train, hpo_m2_val, y_hpo_train, y_hpo_val = preprocess_images(protons_m1, protons_m2, gammas_m1, gammas_m2, train_split=train_split, create_HPO_set=True, hpo_fraction=fraction_for_hpo)
    gc.collect()
    print("Preprocessing complete.")

    num_proton = (y_train == 0).sum()
    num_gamma = (y_train == 1).sum()
    pos_weight = float(num_proton / max(num_gamma, 1))

    train_min = min(m1_train.min(), m2_train.min())
    train_max = max(m1_train.max(), m2_train.max())

    train_dataset = MagicStereoDataset(m1_train, m2_train, y_train, train_min, train_max)
    test_dataset = MagicStereoDataset(m1_test, m2_test, y_test, train_min, train_max)
    val_dataset = MagicStereoDataset(m1_val, m2_val, y_val, train_min, train_max)
    
    if return_HPO_subset:
        hpo_train_dataset = MagicStereoDataset(hpo_m1_train, hpo_m2_train, y_hpo_train, train_min, train_max)
        hpo_val_dataset = MagicStereoDataset(hpo_m1_val, hpo_m2_val, y_hpo_val, train_min, train_max)

    train_loader = tg_loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = tg_loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = tg_loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    if return_HPO_subset:
        hpo_train_loader = tg_loader.DataLoader(hpo_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        hpo_val_loader = tg_loader.DataLoader(hpo_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    gc.collect()
    
    print("Data loaded successfully:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    if return_HPO_subset:
        print(f"  HPO Training samples: {len(hpo_train_loader.dataset)}")
        print(f"  HPO Validation samples: {len(hpo_val_loader.dataset)}")
    print(f"  Positive class weight: {pos_weight:.2f}")

    if return_HPO_subset:
        return train_loader, val_loader, test_loader, pos_weight, hpo_train_loader, hpo_val_loader
    else:
        return train_loader, val_loader, test_loader, pos_weight

if __name__ == "__main__":
    load_stereo_clean_images(num_samples=None)