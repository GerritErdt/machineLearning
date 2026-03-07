from typing import List, Union, Optional
import sys
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
    def __init__(self, m1, m2, y, train_min, train_max, epsilon=1e-4):
        assert m1.shape[0] == m2.shape[0] == y.shape[0], "Mismatch in number of samples between M1, M2 and labels"

        self.images_m1 = torch.from_numpy(m1).float()
        self.images_m2 = torch.from_numpy(m2).float()
        self.labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.intensity_min = train_min
        self.intensity_max = train_max
        self.intensity_span = self.intensity_max - self.intensity_min + epsilon
        self.sqrt_min = 2.0 * np.sqrt(0.375)
        self.sqrt_span = 2.0 * np.sqrt(max(0.0, self.intensity_max) + 0.375) - self.sqrt_min + epsilon
        
        # use the most common value across all images as parameter for the mask generation
        self.background_intensity = 0.0
        
        # Topologie wird einmalig berechnet und als Klassenattribut gespeichert.
        num_pixels = self.images_m1.shape[1]
        self.edge_index, self.pos = compute_camera_topology(num_valid_pixels=num_pixels - 1)

    def __len__(self):
        return self.images_m1.shape[0]

    def __getitem__(self, idx, epsilon=1e-4):
        # 1. Early Fusion
        x_fused = torch.stack([self.images_m1[idx], self.images_m2[idx]], dim=1)
        y = self.labels[idx]

        # 2. Maske für aktive Knoten erstellen (True, wenn M1 ODER M2 ungleich 0)
        # epsilon-Vergleich ist bei floats sicherer als exakt == 0
        mask = (x_fused[:, 0].abs() > self.background_intensity) | (x_fused[:, 1].abs() > self.background_intensity)
        if not mask.any(): # special case: the mask would be empty. In this case, keep the brightest pixel
            max_idx = (x_fused[:, 0].abs() + x_fused[:, 1].abs()).argmax()
            mask[max_idx] = True

        # 3. Indizes der überlebenden Knoten ermitteln
        active_nodes = mask.nonzero(as_tuple=False).view(-1)

        # 4. Subgraph extrahieren und Kanten neu nummerieren lassen
        edge_index, _ = tg_utils.subgraph(active_nodes, self.edge_index, relabel_nodes=True)

        # 5. Features und Positionen auf aktive Knoten reduzieren
        x_sparse = x_fused[mask]
        pos_sparse = self.pos[mask]
        
        # create additional non-linear features
        m1_sparse = x_sparse[:, 0]
        m2_sparse = x_sparse[:, 1]
        
        m1_clipped = torch.clamp(m1_sparse, min=0.0)
        m2_clipped = torch.clamp(m2_sparse, min=0.0)
        
        sqrt_m1 = 2 * torch.sqrt(m1_clipped + 0.375)
        sqrt_m2 = 2 * torch.sqrt(m2_clipped + 0.375)
        
        x_features = torch.stack([m1_sparse, m2_sparse, sqrt_m1, sqrt_m2], dim=1)
        x_features[:, :2] = (x_features[:, :2] - self.intensity_min) / self.intensity_span
        x_features[:, 2:4] = (x_features[:, 2:4] - self.sqrt_min) / (self.sqrt_span + epsilon)

        # Graph MUSS nun Topologie enthalten, da sie dynamisch ist
        return tg_data.Data(x=x_features, edge_index=edge_index, pos=pos_sparse, y=y)

    def get_label_name(self, label):
        val = label.item() if isinstance(label, torch.Tensor) else label
        if val == 0:
            return "Proton"
        elif val == 1:
            return "Gamma"
        else:
            return "Unknown"

def compute_camera_topology(num_valid_pixels = 1038):
    camera = magic.Camera()
    sources = []
    targets = []
    positions = []

    for i in range(num_valid_pixels + 1):
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

def load_stereo_clean_images(num_samples=10000, num_valid_pixels=1038, random_sampling=False, gamma_file="./data/magic-gammas-chunked.parquet", proton_file="./data/magic-protons-chunked.parquet"):
    try:         
        # Lazy scan, will not load in the whole dataset
        gammas = pl.read_parquet(gamma_file).sample(num_samples, seed=42) if random_sampling else pl.scan_parquet(gamma_file).head(num_samples).collect()
        protons = pl.read_parquet(proton_file).sample(num_samples, seed=42) if random_sampling else pl.scan_parquet(proton_file).head(num_samples).collect()

        # Reshape the clean image data into 2D arrays
        protons_clean_image_m1 = np.vstack(protons["clean_image_m1"].to_numpy())[:, :num_valid_pixels + 1]
        protons_clean_image_m2 = np.vstack(protons["clean_image_m2"].to_numpy())[:, :num_valid_pixels + 1]
        gammas_clean_image_m1 = np.vstack(gammas["clean_image_m1"].to_numpy())[:, :num_valid_pixels + 1]
        gammas_clean_image_m2 = np.vstack(gammas["clean_image_m2"].to_numpy())[:, :num_valid_pixels + 1]
        
        assert(protons_clean_image_m1.shape[0] == protons_clean_image_m2.shape[0] == gammas_clean_image_m1.shape[0] == gammas_clean_image_m2.shape[0]), "Mismatch in number of samples between datasets"
        assert(protons_clean_image_m1.shape[1] == num_valid_pixels + 1), f"Expected {num_valid_pixels + 1} pixels in clean_image_m1"
        
        return protons_clean_image_m1, protons_clean_image_m2, gammas_clean_image_m1, gammas_clean_image_m2
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_images(p_m1, p_m2, g_m1, g_m2, train_split = 0.7): 
    x_m1 = np.vstack((p_m1, g_m1))
    x_m2 = np.vstack((p_m2, g_m2))
    y = np.concatenate((np.zeros(p_m1.shape[0]), np.ones(g_m1.shape[0])), axis=0)
    
    x_m1_train, x_m1_rem, x_m2_train, x_m2_rem, y_train, y_rem = ms.train_test_split(
        x_m1, x_m2, y, test_size=1-train_split, random_state=0, stratify=y, shuffle=True
    )
    
    x_m1_val, x_m1_test, x_m2_val, x_m2_test, y_val, y_test = ms.train_test_split(
        x_m1_rem, x_m2_rem, y_rem, test_size=0.5, random_state=0, stratify=y_rem, shuffle=True
    )
    
    return x_m1_train, x_m1_test, x_m1_val, x_m2_train, x_m2_test, x_m2_val, y_train, y_test, y_val
    
def get_stereo_clean_dataset(num_samples = 10000, batch_size = 32):
    protons_m1, protons_m2, gammas_m1, gammas_m2 = load_stereo_clean_images(num_samples)
    m1_train, m1_test, m1_val, m2_train, m2_test, m2_val, y_train, y_test, y_val = preprocess_images(protons_m1, protons_m2, gammas_m1, gammas_m2)
    
    num_proton = (y_train == 0).sum()
    num_gamma = (y_train == 1).sum()
    pos_weight = float(num_proton / max(num_gamma, 1))
    
    train_min = min(m1_train.min(), m2_train.min())
    train_max = max(m1_train.max(), m2_train.max())

    train_dataset = MagicStereoDataset(m1_train, m2_train, y_train, train_min, train_max)
    test_dataset = MagicStereoDataset(m1_test, m2_test, y_test, train_min, train_max)
    val_dataset = MagicStereoDataset(m1_val, m2_val, y_val, train_min, train_max)
    
    train_loader = tg_loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = tg_loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = tg_loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, val_loader, pos_weight


def get_subset_from_loader(full_loader, fraction, shuffle=True):
    dataset = full_loader.dataset
    dataset_size = len(dataset)
    subset_size = max(int(dataset_size * fraction), 1)  # Mindestens 1 Element

    # 1. Zufällige Indizes generieren
    indices = torch.randperm(dataset_size)[:subset_size].tolist()

    # 2. Subset erstellen
    subset = data.Subset(dataset, indices)

    # 3. Neuen PyG DataLoader mit den Parametern des alten Loaders erstellen
    subset_loader = tg_loader.DataLoader(
        subset,
        batch_size=full_loader.batch_size,
        shuffle=shuffle,  # True für Train, False für Val
        num_workers=full_loader.num_workers,
        pin_memory=full_loader.pin_memory
    )

    return subset_loader
