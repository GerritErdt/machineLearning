import optuna

from src.models import model
import src.data_loading.data_loading as dl
import src.helper_functions.helper_functions as hf

def main(trials=50, epochs=20):
    hf.set_all_seeds()
    # todo: reduce dataset for HPO
    train_loader, test_loader, val_loader, pos_weight = dl.get_stereo_clean_dataset(75000, batch_size=128)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: model.objective(
            trial=trial, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader, 
            epochs=epochs, 
            pos_weight=pos_weight),
        n_trials=trials
    )
    
    print("\n\nBest trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    trial = study.best_trial
    
    # TODO: final learning on full dataset with best hyperparameters

def just_train():
    hf.set_all_seeds()
    train_loader, test_loader, val_loader, pos_weight = dl.get_stereo_clean_dataset(75000, batch_size=128)
    
    ml_model = model.GNNModel(
        input_net_layer_count=2, 
        internal_dimensions=128,
        num_edge_convs=3,
        gnn_step_dropout_reduction=2,
        gnn_step_layer_count=2,
        classifier_layer_count=3,
        dropout_rate=0.5
    )
    
    trained_model, history = model.learn(
        model=ml_model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, 
        epochs=10,
        lr_start=1e-4,
        l2_reg=5e-1, 
        pos_weight=pos_weight,
    )
    
    hf.plot_history(history)
        
if __name__ == "__main__":
    # just_train()
    main()