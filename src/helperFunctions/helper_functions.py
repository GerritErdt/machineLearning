import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import src.dataLoading.data_loading as dl
import torch
import random


def plot_fused_magic_graph(data, label_name="Event"):
    # Topologie und Geometrie direkt aus dem dynamischen data-Objekt beziehen
    pos_np = data.pos.numpy()
    edge_index_np = data.edge_index.numpy()

    # Signale aus dem 2D-Feature-Tensor [num_nodes, 2] extrahieren
    signal_m1 = data.x[:, 0].numpy()
    signal_m2 = data.x[:, 1].numpy()

    # Differenz berechnen (Parallaxen-Effekt)
    signal_diff = signal_m1 - signal_m2

    # Globale Min/Max für konsistente Farbskala bei M1 und M2
    vmin = min(signal_m1.min(), signal_m2.min())
    vmax = max(signal_m1.max(), signal_m2.max())

    # Figure mit 3 Subplots erstellen
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    def draw_single_graph(ax, positions, edges, signal, title, v_min, v_max, cmap='viridis'):
        # Kanten zeichnen (nur die, die das Filtern überlebt haben)
        if edges.size > 0:
            lines = [[positions[u], positions[v]] for u, v in edges.T]
            lc = mcoll.LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
            ax.add_collection(lc)

        # Knoten zeichnen
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=signal, cmap=cmap,
                             s=40, zorder=2, vmin=v_min, vmax=v_max)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        return scatter

    # M1 und M2 plotten
    scatter_m1 = draw_single_graph(axes[0], pos_np, edge_index_np, signal_m1,
                                   f"Kanal 0 (M1) - {label_name}", vmin, vmax)
    draw_single_graph(axes[1], pos_np, edge_index_np, signal_m2,
                      f"Kanal 1 (M2) - {label_name}", vmin, vmax)

    # Differenz-Plot mit divergierender Colormap (Null = Weiß/Grau)
    vmax_diff = np.max(np.abs(signal_diff))
    # Fallback, falls vmax_diff 0 ist (verhindert Warning bei identischen Signalen)
    vmax_diff = vmax_diff if vmax_diff > 0 else 1.0

    scatter_diff = draw_single_graph(axes[2], pos_np, edge_index_np, signal_diff,
                                     "Differenz (M1 - M2)", -vmax_diff, vmax_diff, cmap='coolwarm')

    # Colorbars zuweisen
    fig.colorbar(scatter_m1, ax=axes[:2], orientation='vertical', fraction=0.02, pad=0.04,
                 label='Signalintensität')
    fig.colorbar(scatter_diff, ax=axes[2], orientation='vertical', fraction=0.04, pad=0.04,
                 label='Signal-Differenz')

    plt.tight_layout()
    plt.show()

def show_many_graphs():
    train_loader, _ = dl.get_stereo_clean_dataset(num_samples=100)
    train_dataset = train_loader.dataset
    edge_index, pos = dl.compute_camera_topology()

    for i in range(100):
        fused_graph = train_dataset[i]
        label_name = train_dataset.get_label_name(fused_graph.y)
        print(f"Sample {i}: Label: {label_name}")
        print(f"Anzahl Knoten: {fused_graph.num_nodes} (sollte < 1138 sein)")
        print(f"Feature-Shape: {fused_graph.x.shape} (sollte [num_nodes, 2] sein)")
        print(f"Pos-Shape: {fused_graph.pos.shape} (sollte [num_nodes, 2] sein)")
        print(f"Maximaler Kanten-Index: {fused_graph.edge_index.max().item()}")
        plot_fused_magic_graph(fused_graph, label_name)

def plot_histograms_for_telescopes():
    proton_m1, proton_m2, gamma_m1, gamma_m2 = dl.load_stereo_clean_images(num_samples=40000, random_sampling=False)
    
    # 2D Arrays in 1D Arrays umwandeln für pixelweise Verteilung
    proton_m1_flat = proton_m1.flatten()
    proton_m2_flat = proton_m2.flatten()
    gamma_m1_flat = gamma_m1.flatten()
    gamma_m2_flat = gamma_m2.flatten()

    # Erstelle zwei nebeneinander liegende Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gemeinsame Bin-Einstellungen für bessere visuelle Vergleichbarkeit
    bins = 100

    # Plot für Protonen
    axes[0].hist(proton_m1_flat, bins=bins, alpha=0.6, label='M1', color='blue', log=True)
    axes[0].hist(proton_m2_flat, bins=bins, alpha=0.6, label='M2', color='orange', log=True)
    axes[0].set_title('Pixel-Intensitätsverteilung: Protonen')
    axes[0].set_xlabel('Intensität (Photoelektronen)')
    axes[0].set_ylabel('Häufigkeit')
    axes[0].legend()

    # Plot für Gammas
    axes[1].hist(gamma_m1_flat, bins=bins, alpha=0.6, label='M1', color='blue', log=True)
    axes[1].hist(gamma_m2_flat, bins=bins, alpha=0.6, label='M2', color='orange', log=True)
    axes[1].set_title('Pixel-Intensitätsverteilung: Gammas')
    axes[1].set_xlabel('Intensität (Photoelektronen)')
    axes[1].set_ylabel('Häufigkeit')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    import matplotlib.pyplot as plt


def plot_history(history):
    # Definition der Metriken und ihrer Anzeigenamen
    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("f1", "F1-Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("roc", "ROC-AUC")
    ]

    epochs = range(1, len(history["train_loss"]) + 1)

    # Erstelle ein 2x3 Raster
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (key, title) in enumerate(metrics):
        train_vals = history[f"train_{key}"]
        test_vals = history[f"test_{key}"]

        axes[i].plot(epochs, train_vals, label='Train', marker='o', markersize=4, linewidth=1.5)
        axes[i].plot(epochs, test_vals, label='Test', marker='x', markersize=4, linewidth=1.5, linestyle='--')

        axes[i].set_title(f"Model {title}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Speziell für Loss-Plot: y-Achse ggf. logarythmisch, falls Werte stark schwanken
        # if key == "loss": axes[i].set_yscale('log')

    plt.tight_layout()
    plt.show()
  
def set_all_seeds(seed=42):
    # 1. Standard Python Random Seed
    random.seed(seed)

    # 2. NumPy Random Seed (wichtig für scikit-learn train_test_split und Daten-Preprocessing)
    np.random.seed(seed)

    # 3. PyTorch Random Seed (CPU)
    torch.manual_seed(seed)

    # 4. PyTorch Random Seed (GPU/CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Falls Multi-GPU genutzt wird

        # 5. Deterministische CUDA-Operationen erzwingen
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    show_many_graphs()
    # plot_histograms_for_telescopes()