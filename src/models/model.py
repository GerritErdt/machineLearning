from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.nn import Linear, BatchNorm1d
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


class SimpleGNN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        # Einzelner Verarbeitungsstrang für ein Teleskop
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing (Graphen-Faltung)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 2. Readout Layer (Globales Pooling)
        x = global_mean_pool(x, batch)

        # 3. Klassifikation
        x = self.lin(x)
        return x


class RobustSingleTelescopeGNN(torch.nn.Module):
    def __init__(self, num_node_features=1, hidden_channels=64, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        # 1. Convolutional Block
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)

        # 2. Convolutional Block
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)

        # 3. Convolutional Block
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        # Klassifikator: Input ist 2 * hidden_channels (wegen Mean + Max Pooling)
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.bn_lin = BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # Block 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Block 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Block 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        # Kein Dropout direkt vor dem Pooling, um keine wichtigen Graphen-Features zu nullen

        # Readout: Kombination aus Durchschnitt und Maximum
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP Classifier
        x = self.lin1(x)
        x = self.bn_lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        return x


def train_robust_model_m1(train_dataset, test_dataset, num_epochs=30, batch_size=64):
    # Dataloader Setup bleibt identisch
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=m1_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=m1_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustSingleTelescopeGNN(hidden_channels=64, dropout_rate=0.5).to(device)

    # Reduzierte Lernrate (0.001) und L2-Regularisierung (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # ... (Trainings- und Testschleife bleiben exakt wie im vorherigen Beispiel)

def m1_collate(data_list):
    # Entpackt (graph_m1, graph_m2, y) und verwirft graph_m2
    m1_list = [item[0] for item in data_list]
    y_list = [item[2] for item in data_list]

    return Batch.from_data_list(m1_list), torch.tensor(y_list, dtype=torch.long)


def train_model_m1(train_dataset, test_dataset, num_epochs=10, batch_size=32):
    # Dataloader für Training und Test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=m1_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Testdaten müssen nicht gemischt werden
        collate_fn=m1_collate
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustSingleTelescopeGNN(hidden_channels=64, dropout_rate=0.5).to(device)

    # Reduzierte Lernrate (0.001) und L2-Regularisierung (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # --- TRAININGS-PHASE ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for batch_m1, labels in train_loader:
            batch_m1 = batch_m1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(batch_m1.x, batch_m1.edge_index, batch_m1.batch)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = out.argmax(dim=1)
            train_correct += int((preds == labels).sum())
            train_samples += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_samples

        # --- TEST-PHASE ---
        model.eval()  # Schaltet Dropout/BatchNorm auf Evaluierungsmodus (falls vorhanden)
        test_loss = 0
        test_correct = 0
        test_samples = 0

        with torch.no_grad():  # Deaktiviert die Gradientenberechnung für Speichereffizienz
            for batch_m1, labels in test_loader:
                batch_m1 = batch_m1.to(device)
                labels = labels.to(device)

                out = model(batch_m1.x, batch_m1.edge_index, batch_m1.batch)
                loss = criterion(out, labels)

                test_loss += loss.item()
                preds = out.argmax(dim=1)
                test_correct += int((preds == labels).sum())
                test_samples += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_samples

        # Ausgabe pro Epoche
        print(f'Epoch: {epoch+1:02d} | '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return model

# Aufruf in deiner main.py:
# train_dataset, test_dataset = dl.get_stereo_clean_dataset()
# model = train_model_m1(train_dataset, test_dataset)
