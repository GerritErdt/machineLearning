import sys
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from magicdl import magic


class MagicStereoDataset(Dataset):
    def __init__(self, m1, m2, y):
        assert(m1.shape[0] == m2.shape[0] == y.shape[0]), "Mismatch in number of samples between M1, M2 and labels"

        # 1. Daten zusammenführen und in Tensoren konvertieren
        # np.vstack verbindet die Arrays entlang der Batch-Dimension (Zeilen)
        self.images_m1 = torch.tensor(m1, dtype=torch.float32)
        self.images_m2 = torch.tensor(m2, dtype=torch.float32)

        self.labels = torch.tensor(y, dtype=torch.long)

        # 3. Geteilte Kamerageometrie
        self.edge_index, self.pos = compute_camera_topology(num_valid_pixels=self.images_m1.shape[1] - 1)

    def __len__(self):
        return self.images_m1.shape[0]

    def __getitem__(self, idx):
        # Features als Spaltenvektor [1039, 1] extrahieren
        x_m1 = self.images_m1[idx].unsqueeze(1)
        x_m2 = self.images_m2[idx].unsqueeze(1)
        y = self.labels[idx]

        # Graphen On-the-Fly erzeugen
        graph_m1 = Data(x=x_m1, edge_index=self.edge_index, pos=self.pos)
        graph_m2 = Data(x=x_m2, edge_index=self.edge_index, pos=self.pos)

        return graph_m1, graph_m2, y
    
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

def load_stereo_clean_images(num_samples = 10000, num_valid_pixels = 1038):
    try: 
        gamma_file = r"C:/Users/gerri/Meine Ablage/Studium/Maschinelles Lernen/projectWorkspace/data/magic-gammas-chunked.parquet"
        proton_file = r"C:/Users/gerri/Meine Ablage/Studium/Maschinelles Lernen/projectWorkspace/data/magic-protons-chunked.parquet"
        
        # Lazy scan, will not load in the whole dataset
        gammas = pl.scan_parquet(gamma_file)
        protons = pl.scan_parquet(proton_file)

        protons = protons.head(num_samples).collect()
        gammas = gammas.head(num_samples).collect()

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

def preprocess_images(p_m1, p_m2, g_m1, g_m2): 
    x_m1 = np.vstack((p_m1, g_m1))
    x_m2 = np.vstack((p_m2, g_m2))
    y = np.concatenate((np.zeros(p_m1.shape[0]), np.ones(g_m1.shape[0])), axis=0)
    
    x_m1_train, x_m1_test, x_m2_train, x_m2_test, y_train, y_test = train_test_split(
        x_m1, x_m2, y, test_size=0.3, random_state=0, stratify=y, shuffle=True
    )    
    
    m1_min = x_m1_train.min()
    m1_max = x_m1_train.max()
    m2_min = x_m2_train.min()
    m2_max = x_m2_train.max()
    
    x_m1_train = (x_m1_train - m1_min) / (m1_max - m1_min + 1e-8)
    x_m1_test = (x_m1_test - m1_min) / (m1_max - m1_min + 1e-8)
    x_m2_train = (x_m2_train - m2_min) / (m2_max - m2_min + 1e-8)
    x_m2_test = (x_m2_test - m2_min) / (m2_max - m2_min + 1e-8)
    
    return x_m1_train, x_m1_test, x_m2_train, x_m2_test, y_train, y_test
    
    
def get_stereo_clean_dataset(num_samples = 10000):
    protons_m1, protons_m2, gammas_m1, gammas_m2 = load_stereo_clean_images(num_samples)
    m1_train, m1_test, m2_train, m2_test, y_train, y_test = preprocess_images(protons_m1, protons_m2, gammas_m1, gammas_m2)

    trainingDataset = MagicStereoDataset(m1_train, m2_train, y_train)
    testingDataset = MagicStereoDataset(m1_test, m2_test, y_test)
    
    return trainingDataset, testingDataset
