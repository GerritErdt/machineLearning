import torchmetrics as tm
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torchmetrics.classification as tm


class BaselineGNN(nn.Module):
    def __init__(self, in_channels=2, internal_dimensions=64):
        super().__init__()

        self.out_channels = 1

        # Standard GCN Schichten statt komplexer EdgeConvs
        self.conv1 = gnn.GCNConv(in_channels, internal_dimensions)
        self.conv2 = gnn.GCNConv(internal_dimensions, internal_dimensions)

        # Einfacher Klassifikator ohne Dropout und LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(internal_dimensions, internal_dimensions // 2),
            nn.Sigmoid(),
            nn.Linear(internal_dimensions // 2, self.out_channels)
        )
        
        # print number of parameters for debugging
        total_params = sum(p.numel() for p in self.parameters())
        print(f"BaselineGNN initialized with {total_params} parameters.")

    def forward(self, x, edge_index, batch, num_graphs):
        # GNN-Pass ohne Residual Connections
        x = self.conv1(x, edge_index)
        x = torch.sigmoid(x)

        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)

        # Standard Pooling (nur Mean) statt Jumping Knowledge
        x_pooled = gnn.global_mean_pool(x, batch, size=num_graphs)

        # Klassifikation
        out = self.classifier(x_pooled)

        return out

    def predict(self, x, edge_index, batch, num_graphs):
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch, num_graphs)
            probs = torch.sigmoid(logits)

        return probs

def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=100, lr=0.01, weight_decay=1e-4, device='cuda'):
    model = model.to(device)

    # AdamW inkludiert entkoppelte Weight Decay für bessere Regularisierung
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    metrics_train = {"AUROC": tm.BinaryAUROC().to(device)}
    metrics_val = {"AUROC": tm.BinaryAUROC().to(device)}
    metrics_test = {"AUROC": tm.BinaryAUROC().to(device)}

    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_AUROC": [], "val_AUROC": [], "test_AUROC": [],
    }

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch, data.num_graphs)
            target = data.y.view(-1, 1).float()

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs

            # Wichtig: BinaryAUROC erwartet Wahrscheinlichkeiten im Bereich [0, 1]
            probs = torch.sigmoid(out)
            metrics_train["AUROC"].update(probs, target.long())

        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["train_AUROC"].append(metrics_train["AUROC"].compute().item())
        metrics_train["AUROC"].reset()

        # --- EVALUATION (Val & Test) ---
        model.eval()

        with torch.no_grad():
            # Validierung
            val_loss = 0.0
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch, data.num_graphs)
                target = data.y.view(-1, 1).float()

                val_loss += criterion(out, target).item() * data.num_graphs
                metrics_val["AUROC"].update(torch.sigmoid(out), target.long())

            history["val_loss"].append(val_loss / len(val_loader.dataset))
            history["val_AUROC"].append(metrics_val["AUROC"].compute().item())
            metrics_val["AUROC"].reset()

            # Test
            test_loss = 0.0
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch, data.num_graphs)
                target = data.y.view(-1, 1).float()

                test_loss += criterion(out, target).item() * data.num_graphs
                metrics_test["AUROC"].update(torch.sigmoid(out), target.long())

            history["test_loss"].append(test_loss / len(test_loader.dataset))
            history["test_AUROC"].append(metrics_test["AUROC"].compute().item())
            metrics_test["AUROC"].reset()

        # Logging
        print(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {history['train_loss'][-1]:.4f}, AUROC: {history['train_AUROC'][-1]:.4f} | "
                f"Val Loss: {history['val_loss'][-1]:.4f}, AUROC: {history['val_AUROC'][-1]:.4f}")

    return history
