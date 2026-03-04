import torch
import torch.nn as nn
import torchmetrics.classification as tm
import torch_geometric.nn as gnn
import torch.amp as amp

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*") # otherwise, the warning spams the console after every epoch

class GNNModel(nn.Module):
    def __init__(self, input_net_layer_count, internal_dimensions, num_edge_convs, gnn_step_layer_count, gnn_step_dropout_reduction, classifier_layer_count, dropout_rate=0.5, in_channels=4):
        assert input_net_layer_count >= 2, "input_net_layer_count must be at least 2 to have a final linear layer for the residual connection"
        assert gnn_step_layer_count >= 2, "gnn_step_layer_count must be at least 2 to have a final linear layer in the EdgeConv-MLP for the residual connection"
        assert classifier_layer_count >= 2, "classifier_layer_count must be at least 2 to have a final linear layer for the output"
        
        super().__init__()
        
        self.hidden_channels = internal_dimensions
        self.num_edge_convs = num_edge_convs
        self.out_channels = 1
        
        # input transformation, project from two dimensions (M1 and M2) to hidden_channels dimensions for the GNN layers
        input_net_layers = []
        input_net_layers.extend([
            nn.Linear(in_channels, internal_dimensions),
            gnn.GraphNorm(internal_dimensions),
            nn.GELU(),
        ])
        
        for _ in range(input_net_layer_count - 2):
            input_net_layers.extend([
                nn.Linear(internal_dimensions, internal_dimensions),
                gnn.GraphNorm(internal_dimensions),
                nn.GELU(),
            ])
        
        input_net_layers.append(nn.Linear(internal_dimensions, internal_dimensions))  # final linear layer without activation, to be used in residual connection
        self.input_net = nn.Sequential(*input_net_layers)
        
        # GNN-layers
        self.convs = nn.ModuleList() # GraphConv-layers
        self.projs = nn.ModuleList() # re-projection layers
        self.norms = nn.ModuleList() # normalization layers
        
        multi_aggr = gnn.aggr.MultiAggregation(["max", "mean"]) 
        
        for _ in range(self.num_edge_convs):
            conv_mlp_layers = []
            conv_mlp_layers.extend([
                nn.Linear(2 * internal_dimensions, internal_dimensions),  # input dimension is doubled due to EdgeConv's concatenation of central node and neighbor features
                nn.GELU(),
                nn.Dropout(dropout_rate / gnn_step_dropout_reduction), 
            ])
            
            for _ in range(gnn_step_layer_count - 2):
                conv_mlp_layers.extend([
                    nn.Linear(internal_dimensions, internal_dimensions),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / gnn_step_dropout_reduction),
                ])
            
            conv_mlp_layers.append(nn.Linear(internal_dimensions, internal_dimensions))  # final linear layer without activation, to be used in residual connection
            conv_mlp = nn.Sequential(*conv_mlp_layers)
            
            self.convs.append(gnn.EdgeConv(conv_mlp, aggr=multi_aggr))
            
            self.projs.append(nn.Sequential( # only perform linear projection to reduce dimensionality
                nn.Linear(2 * internal_dimensions, internal_dimensions),
            ))
            
            self.norms.append(gnn.GraphNorm(internal_dimensions))
        
        self.final_norm = gnn.GraphNorm(internal_dimensions) # final normalization before the classifier
            
        # classifier-Schicht
        classifier_layers = []
        classifier_layers.extend([
            nn.Linear(2 * internal_dimensions, internal_dimensions),
            nn.LayerNorm(internal_dimensions),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        ])
        
        for _ in range(classifier_layer_count - 2):
            classifier_layers.extend([
                nn.Linear(internal_dimensions, internal_dimensions),
                nn.LayerNorm(internal_dimensions),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
        
        classifier_layers.extend([
            nn.Linear(internal_dimensions, self.out_channels)
        ])
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # weight initialization
        self.apply(self._init_weights)
        
        # print network statistics
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

def get_input_importance(model, test_loader, history, device):
    print("\nCalculating feature-importance")
    model.eval()
    num_features = test_loader.dataset[0].x.shape[1]
    
    baseline_f1 = history["test_f1"][-1]
    feature_importances = []
    
    with torch.no_grad():
        for feature_idx in range(num_features):
            perm_metric = tm.BinaryF1Score().to(device)

            for batch in test_loader:
                batch = batch.to(device)

                # 1. Batch klonen, um Originaldaten im Loader nicht zu zerstören
                batch_cloned = batch.clone()

                # 2. Nur die spezifische Feature-Spalte über alle Knoten hinweg permutieren
                perm_indices = torch.randperm(batch_cloned.x.size(0))
                batch_cloned.x[:, feature_idx] = batch_cloned.x[perm_indices, feature_idx]

                # 3. Infernz mit dem zerstörten Feature
                out = model(batch_cloned.x, batch_cloned.edge_index, batch_cloned.batch, batch_cloned.num_graphs)
                probs = torch.sigmoid(out.view(-1))
                targets = batch_cloned.y.view(-1).long()

                perm_metric.update(probs, targets)

            # 4. Leistungsabfall berechnen
            perm_f1 = perm_metric.compute().item()
            importance = baseline_f1 - perm_f1
            feature_importances.append(importance)

            print(f"Feature {feature_idx}: Importance (Delta F1) = {importance:.4f}")
    
    history["feature_importances"] = feature_importances
    print("Feature-Importance calculation finished.\n")

def grade_model(model, data_loader, device, history, metrics, prefix, criterion):
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
            
            batch_logits = out.view(-1)
            batch_targets = batch.y.view(-1).float()
            
            loss = criterion(batch_logits, batch_targets)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            
            batch_probs = torch.sigmoid(batch_logits)
            for metric in metrics.values():
                metric.update(batch_probs, batch_targets.long())
    
    avg_loss = total_loss / total_samples
    history[f"{prefix}_loss"].append(avg_loss)
    
    for key, metric in metrics.items():
        history[f"{prefix}_{key}"].append(metric.compute().item())
        metric.reset()

def train_one_epoch(model, data_loader, device, history, metrics, optimizer, scaler, criterion):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16):  # automatic mixed precision for faster training on GPU                           
            out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
            logits = out.view(-1)
            targets = batch.y.view(-1).float()
            
            loss = criterion(logits, targets)
        
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)  
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.75)  # gradient Clipping to prevent exploding/excessive gradients at the beginning of training
        
        scaler.step(optimizer)
        scaler.update()
        
        probs = torch.sigmoid(logits).detach()
        for metric in metrics.values():
            metric.update(probs, targets.long())
    
    avg_loss = total_loss / total_samples
    history['train_loss'].append(avg_loss)
    
    for key, metric in metrics.items():
        history[f"train_{key}"].append(metric.compute().item())
        metric.reset()

def learn(model, train_loader, test_loader, val_loader, epochs=10, lr_start=1e-4, lr_patience=3, l2_reg=5e-1, pos_weight=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # auto-detect GPU-availability
    model = model.to(device) 
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))  # automatic mixed precision for faster training on GPU
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    param_groups = get_parameter_groups(model, l2_reg)
    optimizer = torch.optim.AdamW(param_groups, lr=lr_start) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience) 
    
    metrics_train = {
        "precision": tm.BinaryPrecision().to(device),
        "recall": tm.BinaryRecall().to(device),
        "f1": tm.BinaryF1Score().to(device),
    }
    
    metrics_val = {
        "precision": tm.BinaryPrecision().to(device),
        "recall": tm.BinaryRecall().to(device),
        "f1": tm.BinaryF1Score().to(device),
    }
    
    metrics_test = {
        "precision": tm.BinaryPrecision().to(device),
        "recall": tm.BinaryRecall().to(device),
        "f1": tm.BinaryF1Score().to(device),
    }
    
    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_precision": [], "val_precision": [], "test_precision": [],
        "train_recall": [], "val_recall": [], "test_recall": [],
        "train_f1": [], "val_f1": [], "test_f1": [],
    }
    
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, device, history, metrics_train, optimizer, scaler, criterion)
    
        # evaluation on val data after each epoch
        grade_model(model, val_loader, device, history, metrics_val, "val", criterion)
        
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val F1: {history['val_f1'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(history['val_loss'][-1])  
    
    # final evaluation on test data after training
    grade_model(model, test_loader, device, history, metrics_test, "test", criterion)
    print(f"\nFinal Test Loss: {history['test_loss'][-1]:.4f} | Test F1: {history['test_f1'][-1]:.4f}")
    
    get_input_importance(model, test_loader, history, device)
    
    return model, history

if __name__ == "__main__":
    model = GNNModel()
    