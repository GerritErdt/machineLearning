import optuna
import src.models.model_dif_hp as model_def
import src.data_loading.data_loading_new as dl
import src.helper_functions.helper_functions as hf

def main(trials=60, trial_epochs=12, fraction_for_hpo=0.35, final_data_size=int(1e5)):
    hf.set_all_seeds()
    
    # data loading
    train_loader, val_loader, test_loader, pos_weight, hpo_train_loader, hpo_val_loader = dl.get_stereo_clean_dataset(int(final_data_size) if final_data_size else None, batch_size=128, return_HPO_subset=True, fraction_for_hpo=fraction_for_hpo) 
    warmup_epochs = int(trial_epochs * 0.4)  # the LR scheduler takes some time to ramp up, so we give it a warmup period before pruning can kick in
    final_epochs = int(2 * trial_epochs * len(hpo_train_loader) / len(train_loader)) # empirically found, provides a good scale-up
    print(f"Using {final_epochs} epochs for final training based on {trial_epochs} trial epochs and dataset size ratio.")
    
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=max(1, warmup_epochs),
        max_resource=trial_epochs,
        reduction_factor=3
    )
    
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

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
    except Exception as e:
        print("  (Not enough complete trials to calculate importances.)")

    trial = study.best_trial
    
    print("\nStarting final training with best hyperparameters...")
    trained_model, history = model_def.learn(
        model=model_def.GNNModel(
            input_net_dropout=0.1,
            num_edge_convs=8,
            gnn_step_dropout=0.1,
            classifier_dropout=0.1,
            activation_function=trial.params["activation_function"],
            aggr_combination=trial.params["aggr_combination"]
        ), 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, 
        epochs=final_epochs,
        lr_max=trial.params["lr_max"],
        l2_reg=0.00011058565255134606,
        pos_weight=pos_weight,
    )
    
    hf.show_history(history)
        
if __name__ == "__main__":
    main()