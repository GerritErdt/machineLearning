import optuna
import src.models.model as model_def
import src.data_loading.data_loading_new as dl
import src.helper_functions.helper_functions as hf

def main(trials=50, trial_epochs=20, fraction_for_hpo=0.3, final_data_size=None, final_epochs=50):
    hf.set_all_seeds()
    
    # data loading
    train_loader, val_loader, test_loader, pos_weight, hpo_train_loader, hpo_val_loader = dl.get_stereo_clean_dataset(int(final_data_size) if final_data_size else None, batch_size=128, return_HPO_subset=True, fraction_for_hpo=fraction_for_hpo) 

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(
        lambda trial: model_def.objective(
            trial=trial, 
            train_loader=hpo_train_loader, 
            val_loader=hpo_val_loader, 
            test_loader=None, # No test loader during HPO to prevent data leakage
            epochs=trial_epochs, 
            pos_weight=pos_weight),
        n_trials=trials
    )
    
    # HPO results printing
    print("\n\nBest trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    print("\nStudy statistics:")
    print(f"  Total trials: {len(study.trials)}")

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"  Complete trials: {len(complete_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")

    print("\nParameter importances:")
    try:
        importances = optuna.importance.get_param_importances(study)
        for key, val in importances.items():
            print(f"  {key}: {val*100:.1f}%")
    except RuntimeError:
        print("  (Not enough complete trials to calculate importances.)")

    trial = study.best_trial
    
    print("\nStarting final training with best hyperparameters...")
    trained_model, history = model_def.learn(
        model=model_def.GNNModel(
            input_net_dropout=trial.params["input_net_dropout"],
            num_edge_convs=trial.params["num_edge_convs"],
            gnn_step_dropout=trial.params["gnn_step_dropout"],
            classifier_dropout=trial.params["classifier_dropout"]
        ), 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, 
        epochs=final_epochs,
        lr_start=trial.params["lr_start"],
        l2_reg=trial.params["l2_reg"], 
        pos_weight=pos_weight,
    )
    
    hf.plot_history(history)

def just_train():
    hf.set_all_seeds()
    # train_loader, test_loader, val_loader, pos_weight = dl.get_stereo_clean_dataset(None, batch_size=128)
    train_loader, val_loader, test_loader, pos_weight = dl.get_stereo_clean_dataset(1000, batch_size=128, train_split=0.7)
    
    ml_model = model_def.GNNModel(
        input_net_dropout=0.25,
        num_edge_convs=5,
        gnn_step_dropout=0.1,
        classifier_dropout=0.2
    )
    
    trained_model, history = model_def.learn(
        model=ml_model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, 
        epochs=5,
        lr_start=0.0012849329978680513,
        l2_reg=0.0014427968983024913,
        pos_weight=pos_weight,
    )
    
    hf.plot_history(history)
        
if __name__ == "__main__":
    # just_train()
    main()
    