import torch
import torch.nn as nn
import torchmetrics.classification as tm
import torch_geometric.nn as gnn
import torch.amp as amp
import optuna

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*") # otherwise, the warning spams the console after every epoch

class GNNModel(nn.Module):
    def __init__(self, input_net_dropout, internal_dimensions, num_edge_convs, gnn_step_dropout, classifier_dropout, in_channels=4):
        
        super().__init__()
        
        self.hidden_channels = internal_dimensions
        self.num_edge_convs = num_edge_convs
        self.out_channels = 1
        
        # input transformation, project from four dimensions (M1, M2, sqrt(M1) and sqrt(M2)) to hidden_channels dimensions for the GNN layers
        self.input_net = nn.Sequential(
                nn.Linear(in_channels, internal_dimensions),
                nn.LayerNorm(internal_dimensions),
                nn.GELU(),
                nn.Dropout(input_net_dropout),
                
                nn.Linear(internal_dimensions, internal_dimensions),
                nn.LayerNorm(internal_dimensions),
                # no activation, to not 'squeeze' the feature space too much
                # no dropout, as there is no additional layer to restore information here
        )
        
        # GNN-layers
        self.convs = nn.ModuleList() # GraphConv-layers
        self.projs = nn.ModuleList() # re-projection layers
        self.norms = nn.ModuleList() # normalization layers
        
        multi_aggr = gnn.aggr.MultiAggregation(["max", "mean"]) 
        
        for _ in range(self.num_edge_convs):            
            conv_mlp = nn.Sequential( # Combines information about neighboring pixels
                nn.Linear(2 * internal_dimensions, internal_dimensions),
                nn.LayerNorm(internal_dimensions),
                nn.GELU(),
                nn.Dropout(gnn_step_dropout),
                
                nn.Linear(internal_dimensions, internal_dimensions),
                nn.LayerNorm(internal_dimensions),
                nn.GELU(),
            )
            
            self.convs.append(gnn.EdgeConv(conv_mlp, aggr=multi_aggr))
            
            self.projs.append(nn.Sequential( # only perform linear projection to reduce dimensionality (combine max/mean features)
                nn.Linear(2 * internal_dimensions, internal_dimensions),
            ))
            
            self.norms.append(nn.LayerNorm(internal_dimensions)) # normalize after each GNN step
        
        self.norm_after_gnn = nn.LayerNorm(internal_dimensions) # additional normalization after all GNN steps, before pooling. Necessary due to residual connections
            
        # classifier-Schicht
        self.classifier = nn.Sequential(
            nn.Linear(4 * internal_dimensions, internal_dimensions),
            nn.LayerNorm(internal_dimensions),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(internal_dimensions, internal_dimensions),
            nn.LayerNorm(internal_dimensions),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(internal_dimensions, internal_dimensions // 4),
            nn.LayerNorm(internal_dimensions // 4),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(internal_dimensions // 4, self.out_channels)
        )
        
        # weight initialization
        self.apply(self._init_weights)
        
        # print network statistics
        print(f"Initialized GNNModel with: internal_dimensions={internal_dimensions:<3}, num_edge_convs={num_edge_convs:<2}, "\
              f"gnn_step_dropout={gnn_step_dropout:<4.2f}, classifier_dropout={classifier_dropout:<4.2f}. Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
    
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # todo: remove not used normalization layers
        # todo: is there a better way to initialize norms? 
        elif isinstance(module, (nn.LayerNorm)): # Normalization should act as identity in the beginning
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch, num_graphs): 
        # initial input transformation
        x = self.input_net(x)
        
        # jumping knowledge of initial features against over-smoothing
        x_input_max = gnn.global_max_pool(x, batch, size=num_graphs)
        x_input_mean = gnn.global_mean_pool(x, batch, size=num_graphs)
        
        # GNN-layers with residual connections
        for conv, proj, norm in zip(self.convs, self.projs, self.norms):
            identity = x

            h = norm(x) # pre-norm ensures that the data are always normalized, even when from the residual connection
            h = conv(h, edge_index)
            h = proj(h)
            
            x = identity + h  # residual connection. This creates unnormalized data
        
        x = self.norm_after_gnn(x) # final normalization after GNN steps, before pooling. This is necessary due to the residual connections
        
        # feature pooling of gnn features
        x_gnn_max = gnn.global_max_pool(x, batch, size=num_graphs)  
        x_gnn_mean = gnn.global_mean_pool(x, batch, size=num_graphs)
        
        x_pooled = torch.cat([x_input_max, x_input_mean, x_gnn_max, x_gnn_mean], dim=1)  # concatenate all pooled features, using jumping knowledge
        
        # final classification
        out = self.classifier(x_pooled)
        
        return out

    def predict(self, x, edge_index, batch, num_graphs):
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch, num_graphs)
            probs = torch.sigmoid(logits)
            
        return probs

def get_input_importance(model, test_loader, history, device):
    print("\nCalculating feature-importance")
    model.eval()
    num_features = test_loader.dataset[0].x.shape[1]

    baseline_auroc = history["test_AUROC"][-1]
    feature_importances = []

    with torch.no_grad():
        for feature_idx in range(num_features):
            perm_metric = tm.BinaryAUROC().to(device)

            for batch in test_loader:
                batch = batch.to(device)

                # 1. Batch klonen und x explizit neu allozieren (verhindert In-Place-Speicherfehler)
                batch_cloned = batch.clone()
                batch_cloned.x = batch_cloned.x.clone()

                # 2. Vektorisierte Permutation isoliert innerhalb jedes Graphen
                # Wir generieren Rauschen in [0, 1) und addieren einen Offset basierend auf der Graph-ID.
                # Dadurch bleiben die Werte eines Graphen bei der anschließenden Sortierung strikt unter sich.
                noise = torch.rand(batch_cloned.x.size(0), device=device)
                offset = batch_cloned.batch.float() * 2.0
                sort_keys = noise + offset
                _, perm_indices = torch.sort(sort_keys)

                # 3. Das spezifische Feature mit den gruppierten Indizes vertauschen
                batch_cloned.x[:, feature_idx] = batch_cloned.x[perm_indices, feature_idx]

                # 4. Inferenz mit dem gestörten Feature
                out = model(batch_cloned.x, batch_cloned.edge_index, batch_cloned.batch, batch_cloned.num_graphs)
                probs = torch.sigmoid(out.view(-1))
                targets = batch_cloned.y.view(-1).long()

                perm_metric.update(probs, targets)

            # 5. Leistungsabfall (Delta AUROC) berechnen
            perm_auroc = perm_metric.compute().item()
            importance = baseline_auroc - perm_auroc
            feature_importances.append(importance)

            print(f"Feature {feature_idx}: Importance (Delta AUROC) = {importance:.4f}")

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

def learn(model, train_loader, val_loader, test_loader, epochs, lr_start, l2_reg, pos_weight, lr_patience=5, early_stopping_patience=10, trial=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # auto-detect GPU-availability
    model = model.to(device) 
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))  # automatic mixed precision for faster training on GPU
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=lr_patience, min_lr=1e-6) 
    
    metrics_train = {
        "AUROC": tm.BinaryAUROC().to(device),
    }
    
    metrics_val = {
        "AUROC": tm.BinaryAUROC().to(device),
    }
    
    metrics_test = {
        "AUROC": tm.BinaryAUROC().to(device),
    }
    
    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_AUROC": [], "val_AUROC": [], "test_AUROC": [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, device, history, metrics_train, optimizer, scaler, criterion)
    
        # evaluation on val data after each epoch
        grade_model(model, val_loader, device, history, metrics_val, "val", criterion)
        current_val_loss = history['val_loss'][-1]
        
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {current_val_loss:.4f} | "
              f"Val AUROC: {history['val_AUROC'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if trial is not None:
            trial.report(current_val_loss, epoch)
            if(trial.should_prune()): 
                print("Trial pruned due to no improvement in validation loss.")
                raise optuna.exceptions.TrialPruned()
        
        # early stopping and model checkpointing
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # save the best model's state_dict
        else:
            patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                model.load_state_dict(best_model)  # load the best model's weights before stopping
                break

        scheduler.step(current_val_loss)
    
    model.load_state_dict(best_model)  # ensure the best model is loaded after training loop ends, in case of early stopping
    
    # final evaluation on test data after training, but only if not in HP optimization
    if trial is None:
        grade_model(model, test_loader, device, history, metrics_test, "test", criterion)
        print(f"\nFinal Test Loss: {history['test_loss'][-1]:.4f} | Test AUROC: {history['test_AUROC'][-1]:.4f}")
        
        get_input_importance(model, test_loader, history, device)
    
    return model, history

def objective(trial, train_loader, val_loader, test_loader, epochs, pos_weight):
    config = {
        # model parameters
        "input_net_dropout": trial.suggest_float("input_net_dropout", 0.00, 0.2, step=0.05), 
        "internal_dimensions": trial.suggest_categorical("internal_dimensions", [32, 64, 128]), # 2er potences, which is better for memory-alignment
        "num_edge_convs": trial.suggest_int("num_edge_convs", 3, 6, step=1),
        "gnn_step_dropout": trial.suggest_float("gnn_step_dropout", 0.1, 0.5, step=0.1),
        "classifier_dropout": trial.suggest_float("classifier_dropout", 0.1, 0.5, step=0.1),
        
        # training parameters
        "lr_start": trial.suggest_float("lr_start", 5e-4, 5e-3, log=True),
        "l2_reg": trial.suggest_float("l2_reg", 1e-4, 1e-2, log=True)
    }
    
    model = GNNModel(
        input_net_dropout=config["input_net_dropout"],
        internal_dimensions=config["internal_dimensions"],
        num_edge_convs=config["num_edge_convs"],
        gnn_step_dropout=config["gnn_step_dropout"],
        classifier_dropout=config["classifier_dropout"]
    )
    
    trained_model, history = learn(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr_start=config["lr_start"],
        l2_reg=config["l2_reg"],
        pos_weight=pos_weight,
        trial=trial
    )
    
    return min(history["val_loss"])  # return the best val loss (valid due to early stopping)
