import torch
import torch.nn as nn
import torchmetrics.classification as tm
import torch_geometric.nn as gnn
import torch.amp as amp
import torchviz as tv

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*") # otherwise, the warning spams the console after every epoch

class GNNModel(nn.Module):
    def __init__(self, in_channels = 2, hidden_channels = 128, num_edge_convs = 3, out_channels = 1, dropout_rate=0.5):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_edge_convs = num_edge_convs
        
        # input transformation, project from two dimensions (M1 and M2) to hidden_channels dimensions for the GNN layers
        self.input_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), 
            gnn.GraphNorm(hidden_channels),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_channels, hidden_channels), # no activation here, to allow for residual connections right after the input layer as well
        )
        
        # GNN-layers
        self.convs = nn.ModuleList() # GraphConv-layers
        self.projs = nn.ModuleList() # re-projection layers
        self.norms = nn.ModuleList() # normalization layers
        
        multi_aggr = gnn.aggr.MultiAggregation(["max", "mean"]) 
        
        for _ in range(self.num_edge_convs):
            conv_mlp = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate / 2),
                
                nn.Linear(hidden_channels, hidden_channels), # linear output without activation, to be used in residual connection
            )
            self.convs.append(gnn.EdgeConv(conv_mlp, aggr=multi_aggr))
            
            self.projs.append(nn.Sequential( # only perform linear projection to reduce dimensionality
                nn.Linear(2 * hidden_channels, hidden_channels),
            ))
            
            self.norms.append(gnn.GraphNorm(hidden_channels))
        
        self.final_norm = gnn.GraphNorm(hidden_channels) # final normalization before the classifier
            
        # classifier-Schicht
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # weight initialization
        self.apply(self._init_weights)
        
        # print network statistics
        print(f"Model initialized with {self.num_edge_convs} EdgeConv layers, hidden dimension {self.hidden_channels}, and dropout rate {dropout_rate}.")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
    
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01) # adapted to leaky-relu
            if module.bias is not None: 
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight) # no normalization, to not introduce a bias in weight-change-direction
            nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch, num_graphs): 
        # Eingangs-Transformation
        x = self.input_net(x)
        
        # GNN-Schichten
        for conv, proj, norm in zip(self.convs, self.projs, self.norms):
            identity = x
            
            x = norm(x, batch)
            x = nn.functional.leaky_relu(x)
            
            x = conv(x, edge_index)
            x = proj(x)
            
            x = x + identity  # residual-connection to prevent vanishing gradient problem
        
        x = self.final_norm(x, batch)
        x = nn.functional.leaky_relu(x)
        
        # feature pooling
        x_max = gnn.global_max_pool(x, batch, size=num_graphs)  
        x_mean = gnn.global_mean_pool(x, batch, size=num_graphs)
        
        x_pooled = torch.cat([x_max, x_mean], dim=1)  # concatenate all pooled features
        
        # final classification
        out = self.classifier(x_pooled)
        
        return out

    def predict(self, x, edge_index, batch, num_graphs):
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch, num_graphs)
            probs = torch.sigmoid(logits)
            
        return probs

    def save(self, x, edge_index, batch, num_graphs, path="./model.onnx"):
        # 1. Device ermitteln, auf dem das Modell liegt
        device = next(self.parameters()).device

        # 2. Modell in den Evaluations-Modus versetzen (wichtig für Dropout/Norm-Layer)
        self.eval()

        # 3. Alle Eingangsdaten auf dieses Device schieben
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        # num_graphs ist meist ein Integer, falls es ein Tensor ist: .to(device)

        # 4. Export starten
        torch.onnx.export(
            self,
            (x, edge_index, batch, num_graphs),
            path,
            export_params=True,
            opset_version=14,
            input_names=['x', 'edge_index', 'batch', 'num_graphs'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'batch': {0: 'num_nodes'}
            }
        )
        print(f"Modell erfolgreich unter {path} gespeichert.")

def get_parameter_groups(model, weight_decay): 
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else: 
            decay.append(param)
    
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

# TODO: include validation data
def learn(model, train_loader, test_loader, epochs=10, lr_start=1e-4, lr_patience=3, l2_reg=5e-4, pos_weight=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # auto-detect GPU-availability
    model = model.to(device) 
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))  # automatic mixed precision for faster training on GPU
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    param_groups = get_parameter_groups(model, l2_reg)
    optimizer = torch.optim.AdamW(param_groups, lr=lr_start) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience) # TODO: LR scheduler needs to run on validation data not test data
    
    metrics_train = {
        "accuracy": tm.BinaryAccuracy().to(device),
        "precision": tm.BinaryPrecision().to(device),
        "recall": tm.BinaryRecall().to(device),
        "f1": tm.BinaryF1Score().to(device),
        "roc": tm.BinaryAUROC().to(device)
    }
    
    metrics_test = {
        "accuracy": tm.BinaryAccuracy().to(device),
        "precision": tm.BinaryPrecision().to(device),
        "recall": tm.BinaryRecall().to(device),
        "f1": tm.BinaryF1Score().to(device),
        "roc": tm.BinaryAUROC().to(device)
    }
    
    history = {
        "train_loss": [], "test_loss": [],
        "train_accuracy": [], "test_accuracy": [],
        "train_precision": [], "test_precision": [],
        "train_recall": [], "test_recall": [],
        "train_f1": [], "test_f1": [],
        "train_roc": [], "test_roc": []
    }
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        
        # collect metrics outside of the batch loop for performance reasons
        epoch_train_logits = []
        epoch_train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16):  # automatic mixed precision for faster training on GPU                           
                out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
                logits = out.view(-1)
                targets = batch.y.view(-1).float()
                
                loss = criterion(logits, targets)
            
            total_train_loss += loss.item() * batch.num_graphs
            total_train_samples += batch.num_graphs
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)  
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.75)  # gradient Clipping to prevent exploding/excessive gradients at the beginning of training
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_logits.append(logits.detach())
            epoch_train_targets.append(targets.detach())
        
        all_train_logits = torch.cat(epoch_train_logits)
        all_train_targets = torch.cat(epoch_train_targets).long()
        all_train_probs = torch.sigmoid(all_train_logits)
        
        for metric in metrics_train.values():
            metric.update(all_train_probs, all_train_targets)
    
        # evaluation on testset after each epoch
        model.eval()
        total_test_loss = 0
        total_test_samples = 0
        
        # collect metrics outside of the batch loop for performance reasons
        epoch_val_logits = []
        epoch_val_targets = []
        
        with torch.no_grad():
            for batch in test_loader: 
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
                
                logits = out.view(-1)
                targets = batch.y.view(-1).float()
                
                loss = criterion(logits, targets)
                total_test_loss += loss.item() * batch.num_graphs
                total_test_samples += batch.num_graphs
                
                epoch_val_logits.append(logits)
                epoch_val_targets.append(targets)
        
        all_val_logits = torch.cat(epoch_val_logits)
        all_val_targets = torch.cat(epoch_val_targets).long()
        all_val_probs = torch.sigmoid(all_val_logits)
        
        for metric in metrics_test.values():
            metric.update(all_val_probs, all_val_targets)
        
        epoch_train_loss = total_train_loss / total_train_samples
        epoch_test_loss = total_test_loss / total_test_samples
        
        history["train_loss"].append(epoch_train_loss)
        history["test_loss"].append(epoch_test_loss)
        
        for key in metrics_train.keys():
            history[f"train_{key}"].append(metrics_train[key].compute().item())
            metrics_train[key].reset()
            
            history[f"test_{key}"].append(metrics_test[key].compute().item())
            metrics_test[key].reset()
        
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | "
              f"Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | "
              f"Test Acc: {history['test_accuracy'][-1]:.4f} | Test F1: {history['test_f1'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(epoch_test_loss)
    
    return model, history

if __name__ == "__main__":
    model = GNNModel()