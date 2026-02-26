import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def plot_stereo_magic_graphs(data_m1, data_m2, label_name="Event"):
    # PyTorch-Tensoren extrahieren
    pos_m1 = data_m1.pos.numpy()
    edge_index_m1 = data_m1.edge_index.numpy()
    signal_m1 = data_m1.x.numpy().flatten()

    pos_m2 = data_m2.pos.numpy()
    edge_index_m2 = data_m2.edge_index.numpy()
    signal_m2 = data_m2.x.numpy().flatten()

    # Zwingend erforderlich: Globale Min/Max-Werte für konsistente Farbskala
    vmin = min(signal_m1.min(), signal_m2.min())
    vmax = max(signal_m1.max(), signal_m2.max())

    # Figure mit 2 Subplots nebeneinander erstellen
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def draw_single_graph(ax, pos, edge_index, signal, title):
        # Kanten
        lines = [[pos[u], pos[v]] for u, v in edge_index.T]
        lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.6, zorder=1)
        ax.add_collection(lc)

        # Knoten (vmin und vmax erzwingen die gleiche Farbskala)
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=signal, cmap='viridis', s=40, zorder=2, vmin=vmin, vmax=vmax)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        return scatter

    # M1 (links) und M2 (rechts) plotten
    scatter_m1 = draw_single_graph(axes[0], pos_m1, edge_index_m1, signal_m1, f"Teleskop M1 - {label_name}")
    draw_single_graph(axes[1], pos_m2, edge_index_m2, signal_m2, f"Teleskop M2 - {label_name}")

    # Gemeinsame Colorbar für die gesamte Figure
    cbar = fig.colorbar(scatter_m1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Signalintensität (Standardisiert)')

    plt.show()


def show_many_graphs():
    trainingSet, testSet = dl.get_stereo_clean_dataset()

    for i in range(100):
        graph_m1, graph_m2, label = trainingSet[i]
        label_name = trainingSet.get_label_name(label)
        print(f"Sample {i}: Label: {label_name}")
        plot_stereo_magic_graphs(graph_m1, graph_m2, label_name=label_name)

if __name__ == "__main__":
    show_many_graphs()