import torch
import random
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import src.data_loading.data_loading as dl

def show_fused_magic_graph(data, label_name="Event"):
    pos_np = data.pos.numpy()
    edge_index_np = data.edge_index.numpy()

    signal_m1 = data.x[:, 0].numpy()
    signal_m2 = data.x[:, 1].numpy()

    signal_diff = signal_m1 - signal_m2

    vmin = min(signal_m1.min(), signal_m2.min())
    vmax = max(signal_m1.max(), signal_m2.max())

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    def draw_single_graph(ax, positions, edges, signal, title, v_min, v_max, cmap='viridis'):
        if edges.size > 0:
            lines = [[positions[u], positions[v]] for u, v in edges.T]
            lc = mcoll.LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
            ax.add_collection(lc)

        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=signal, cmap=cmap,
                             s=40, zorder=2, vmin=v_min, vmax=v_max)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        return scatter

    scatter_m1 = draw_single_graph(axes[0], pos_np, edge_index_np, signal_m1,
                                   f"Channel 0 (M1) - {label_name}", vmin, vmax)
    draw_single_graph(axes[1], pos_np, edge_index_np, signal_m2,
                      f"Channel 1 (M2) - {label_name}", vmin, vmax)

    vmax_diff = np.max(np.abs(signal_diff))
    vmax_diff = vmax_diff if vmax_diff > 0 else 1.0

    scatter_diff = draw_single_graph(axes[2], pos_np, edge_index_np, signal_diff,
                                     "Difference (M1 - M2)", -vmax_diff, vmax_diff, cmap='coolwarm')

    fig.colorbar(scatter_m1, ax=axes[:2], orientation='vertical', fraction=0.02, pad=0.04,
                 label='Signal intensity')
    fig.colorbar(scatter_diff, ax=axes[2], orientation='vertical', fraction=0.04, pad=0.04,
                 label='Signal-difference (M1 - M2)')


    plt.show()

def show_many_graphs(num_plots = 10):
    train_loader, _, _, _= dl.get_stereo_clean_dataset(num_samples=100)
    train_dataset = train_loader.dataset
    edge_index, pos = dl.compute_camera_topology()

    for i in range(num_plots):
        fused_graph = train_dataset[i]
        label_name = train_dataset.get_label_name(fused_graph.y)
        show_fused_magic_graph(fused_graph, label_name)

def show_histograms_for_telescopes(num_samples=50000):
    proton_m1, proton_m2, gamma_m1, gamma_m2 = dl.load_stereo_clean_images(num_samples=num_samples)
    
    proton_m1_flat = proton_m1.flatten()
    proton_m2_flat = proton_m2.flatten()
    gamma_m1_flat = gamma_m1.flatten()
    gamma_m2_flat = gamma_m2.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = 100

    axes[0].hist(proton_m1_flat, bins=bins, alpha=0.6, label='M1', color='blue', log=True)
    axes[0].hist(proton_m2_flat, bins=bins, alpha=0.6, label='M2', color='orange', log=True)
    axes[0].set_title('Pixel-Intensity distribution: Protonen')
    axes[0].set_xlabel('Intensity (Photoelectrons)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    axes[1].hist(gamma_m1_flat, bins=bins, alpha=0.6, label='M1', color='blue', log=True)
    axes[1].hist(gamma_m2_flat, bins=bins, alpha=0.6, label='M2', color='orange', log=True)
    axes[1].set_title('Pixel-Intensity distribution: Gammas')
    axes[1].set_xlabel('Intensity (Photoelectrons)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def show_history(history, feature_names=None):
    metrics = [
        ("loss", "Loss"),
        ("AUROC", "AUROC")
    ]

    epochs = range(1, len(history["train_loss"]) + 1)
    last_epoch = epochs[-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (key, title) in enumerate(metrics):
        train_vals = history[f"train_{key}"]
        val_vals = history[f"val_{key}"]

        test_val = history[f"test_{key}"][-1]

        axes[i].plot(epochs, train_vals, label='Train', marker='o', markersize=4, linewidth=1.5)
        axes[i].plot(epochs, val_vals, label='Validation', marker='s', markersize=4, linewidth=1.5)

        axes[i].plot(last_epoch, test_val, label='Test', marker='X', markersize=10, color='red', linestyle='None', zorder=3)

        axes[i].set_title(f"Model {title}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    if "feature_importances" in history:
        importances = history["feature_importances"]
        num_features = len(importances)

        if feature_names is None or len(feature_names) != num_features:
            feature_names = [f"F{i}" for i in range(num_features)]

        axes[2].bar(feature_names, importances, color='steelblue', zorder=2)
        axes[2].set_title("Permutation Feature Importance")
        axes[2].set_xlabel("Features")
        axes[2].set_ylabel("Delta AUROC")  
        axes[2].grid(axis='y', alpha=0.3, zorder=1)

        for i, v in enumerate(importances):
            axes[2].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    plt.show()


def show_predictions(model, test_loader, num_samples=12):
    device = next(model.parameters()).device
    model.eval()
    dataset = test_loader.dataset

    target_per_category = num_samples // 4

    categories = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Vorzeitiger Abbruch, wenn alle Quoten erfüllt sind
            if all(len(items) >= target_per_category for items in categories.values()):
                break

            data = dataset[idx]
            batch = Batch.from_data_list([data]).to(device)
            prob = model.predict(batch.x, batch.edge_index, batch.batch, batch.num_graphs).item()

            pred_class = int(prob > 0.5)
            true_class = int(data.y.item())

            cat_key = (true_class, pred_class)

            if len(categories[cat_key]) < target_per_category:
                categories[cat_key].append((idx, prob, pred_class, true_class))

    selected_samples = []
    for items in categories.values():
        selected_samples.extend(items)

    def draw_single_graph(ax, positions, edges, signal, title, v_min, v_max, cmap='viridis'):
        if edges.size > 0:
            lines = [[positions[u], positions[v]] for u, v in edges.T]
            lc = mcoll.LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
            ax.add_collection(lc)

        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=signal, cmap=cmap,
                             s=40, zorder=2, vmin=v_min, vmax=v_max)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        return scatter

    named_predictions = []

    with torch.no_grad():
        for idx, prob, pred_class, true_class in selected_samples:
            data = dataset[idx]

            pred_name = dataset.get_label_name(pred_class)
            label_name = dataset.get_label_name(true_class)

            named_predictions.append((prob, pred_name, label_name))

            pos_np = data.pos.cpu().numpy()
            edge_index_np = data.edge_index.cpu().numpy()

            signal_m1 = data.x[:, 0].cpu().numpy()
            signal_m2 = data.x[:, 1].cpu().numpy()
            signal_diff = signal_m1 - signal_m2

            vmin = min(signal_m1.min(), signal_m2.min())
            vmax = max(signal_m1.max(), signal_m2.max())

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            title_color = "green" if pred_class == true_class else "red"

            fig.suptitle(
                f"Sample {idx} | Output: {prob:.4f} | "
                f"Prediction: {pred_name} | True Label: {label_name}",
                fontsize=18, fontweight='bold', color=title_color
            )

            scatter_m1 = draw_single_graph(axes[0], pos_np, edge_index_np, signal_m1,
                                           "Channel 0 (M1)", vmin, vmax)
            draw_single_graph(axes[1], pos_np, edge_index_np, signal_m2,
                              "Channel 1 (M2)", vmin, vmax)

            vmax_diff = np.max(np.abs(signal_diff))
            vmax_diff = vmax_diff if vmax_diff > 0 else 1.0
            scatter_diff = draw_single_graph(axes[2], pos_np, edge_index_np, signal_diff,
                                             "Difference (M1 - M2)", -vmax_diff, vmax_diff, cmap='coolwarm')

            fig.colorbar(scatter_m1, ax=axes[:2], orientation='vertical', fraction=0.02, pad=0.04,
                         label='Signal intensity')
            fig.colorbar(scatter_diff, ax=axes[2], orientation='vertical', fraction=0.04, pad=0.04,
                         label='Signal Difference')

            plt.show()

    return named_predictions

def set_all_seeds(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_all_seeds()
    show_many_graphs()
    # plot_histograms_for_telescopes()