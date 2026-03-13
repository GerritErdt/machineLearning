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
    def __init__(self, m1, m2, y, train_min, train_max, epsilon=1e-6, num_pixels=1039):
        assert m1.shape[0] == m2.shape[0] == y.shape[0], "Mismatch in number of samples between M1, M2 and labels"
        
        # type casting to float32 and ensuring only the first num_pixels are used to get rid of empty pixels
        m1 = m1.astype(np.float32)[:, :num_pixels]
        m2 = m2.astype(np.float32)[:, :num_pixels]

        # perform all transformations in-place to save memory
        transformed_m1 = torch.from_numpy(m1).float().clamp_(min=0.0).sqrt_()
        transformed_m2 = torch.from_numpy(m2).float().clamp_(min=0.0).sqrt_()
        labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # calculate normalization and clipping constants based on training data
        sqrt_min = 0.0
        sqrt_span = np.sqrt(max(0.0, train_max)) - sqrt_min + epsilon
        background_intensity = 0.0

        # get neighborhood topology and node positions for the camera pixels (same for all samples)
        global_edge_index, global_pos = compute_camera_topology(num_valid_pixels=num_pixels)

        # stack M1 and M2 features together and normalize
        all_features = torch.stack([transformed_m1, transformed_m2], dim=-1)  
        all_features = (all_features - sqrt_min) / (sqrt_span + epsilon)

        # build all graphs
        self.graphs = []
        num_samples = transformed_m1.shape[0]

        for idx in range(num_samples):
            m1_img = transformed_m1[idx]
            m2_img = transformed_m2[idx]

            # only use regions where the pixels are actually 'active' (above background intensity) in at least one of the two telescopes
            mask = (m1_img.abs() > background_intensity) | (m2_img.abs() > background_intensity)
            
            # edge case for empty images (there is at least one): we take the pixel with the highest combined intensity to ensure we have at least one node in the graph
            if not mask.any():
                max_idx = (m1_img.abs() + m2_img.abs()).argmax()
                mask[max_idx] = True

            active_nodes = mask.nonzero(as_tuple=False).view(-1)

            # get the subgraph and add it
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

    def __getitem__(self, idx): # return one of the pre-calculated graphs
        return self.graphs[idx]

    def get_label_name(self, label):
        val = label.item() if isinstance(label, torch.Tensor) else label
        return {0: "Proton", 1: "Gamma"}.get(val, "Unknown")

# basically a copy from Jarred
def compute_camera_topology(num_valid_pixels = 1039):
    camera = magic.Camera()
    sources = []
    targets = []
    positions = []

    for i in range(num_valid_pixels):
        neighbors = camera.get_pixel_neighbors(i)
        for j in neighbors:
            if j < num_valid_pixels:  
                sources.append(i)
                targets.append(j)

        positions.append(camera.get_pixel_coordinates_xy(i))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)

    return edge_index, pos

def load_stereo_clean_images(num_samples=None, gamma_file="./data/magic-gammas-chunked.parquet", proton_file="./data/magic-protons-chunked.parquet", num_pixels=1039):
    def extract_images(file_path):
        # load only what is actually needed
        ds = pl.scan_parquet(file_path).select([
            pl.col("clean_image_m1").list.slice(0, num_pixels).cast(pl.List(pl.Float32)),
            pl.col("clean_image_m2").list.slice(0, num_pixels).cast(pl.List(pl.Float32))
        ])

        if num_samples:
            ds = ds.head(num_samples)

        df = ds.collect()
        
        # reshape to 1d arrays of shape (num_samples, num_pixels)
        m1 = df["clean_image_m1"].list.explode().to_numpy().reshape(-1, num_pixels)
        m2 = df["clean_image_m2"].list.explode().to_numpy().reshape(-1, num_pixels)

        del df
        gc.collect() # free memory immediately after extracting the images if possible
        return m1, m2

    try:
        # extract images, separately for type and telescopes
        p_m1, p_m2 = extract_images(proton_file)
        g_m1, g_m2 = extract_images(gamma_file)
        return p_m1, p_m2, g_m1, g_m2

    except Exception as e:
        print(f"Error in load_stereo_clean_images: {e}")
        return None

def split_data(p_m1, p_m2, g_m1, g_m2, train_split=0.7, create_HPO_set=False, hpo_fraction=0.4):
    x_m1 = np.vstack((p_m1, g_m1))
    x_m2 = np.vstack((p_m2, g_m2))
    y = np.concatenate((np.zeros(p_m1.shape[0]), np.ones(g_m1.shape[0])), axis=0)

    # first, get a split for training data
    x_m1_train, x_m1_rem, x_m2_train, x_m2_rem, y_train, y_rem = ms.train_test_split(
        x_m1, x_m2, y, test_size=1-train_split, random_state=0, stratify=y, shuffle=True
    )

    # split the remaining part in half for test and val data
    x_m1_val, x_m1_test, x_m2_val, x_m2_test, y_val, y_test = ms.train_test_split(
        x_m1_rem, x_m2_rem, y_rem, test_size=0.5, random_state=0, stratify=y_rem, shuffle=True
    )
    
    # only when HPO is needed, perform a new split on the training data
    if create_HPO_set:
        _, x_m1_hpo_pool, _, x_m2_hpo_pool, _, y_hpo_pool = ms.train_test_split(
            x_m1_train, x_m2_train, y_train, test_size=hpo_fraction, random_state=0, stratify=y_train, shuffle=True
        )

        x_m1_hpo_train, x_m1_hpo_val, x_m2_hpo_train, x_m2_hpo_val, y_hpo_train, y_hpo_val = ms.train_test_split(
            x_m1_hpo_pool, x_m2_hpo_pool, y_hpo_pool, test_size=0.4, random_state=0, stratify=y_hpo_pool, shuffle=True
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
    m1_train, m1_val, m1_test, m2_train, m2_val, m2_test, y_train, y_val, y_test, hpo_m1_train, hpo_m1_val, hpo_m2_train, hpo_m2_val, y_hpo_train, y_hpo_val = split_data(protons_m1, protons_m2, gammas_m1, gammas_m2, train_split=train_split, create_HPO_set=True, hpo_fraction=fraction_for_hpo)
    gc.collect()
    
    print("Computing normalization constants...")

    num_proton = (y_train == 0).sum()
    num_gamma = (y_train == 1).sum()
    pos_weight = float(num_proton / max(num_gamma, 1))

    # min/max scaling on data
    train_min = min(m1_train.min(), m2_train.min())
    train_max = max(m1_train.max(), m2_train.max())
    
    print("Creating  datasets and dataloaders...")

    # get datasets from it
    train_dataset = MagicStereoDataset(m1_train, m2_train, y_train, train_min, train_max)
    test_dataset = MagicStereoDataset(m1_test, m2_test, y_test, train_min, train_max)
    val_dataset = MagicStereoDataset(m1_val, m2_val, y_val, train_min, train_max)
    
    if return_HPO_subset:
        hpo_train_dataset = MagicStereoDataset(hpo_m1_train, hpo_m2_train, y_hpo_train, train_min, train_max)
        hpo_val_dataset = MagicStereoDataset(hpo_m1_val, hpo_m2_val, y_hpo_val, train_min, train_max)

    # create dataloaders from it, needed for torch_geometric
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