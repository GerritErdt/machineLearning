from src.models import model
import src.data_loading.data_loading as dl
import src.helper_functions.helper_functions as hf

def main():
    hf.set_all_seeds()
    train_loader, test_loader, val_loader, pos_weight_val = dl.get_stereo_clean_dataset(75000, batch_size=128)
    
    ml_model = model.GNNModel(
        input_net_layer_count=2, 
        internal_dimensions=128,
        num_edge_convs=3,
        gnn_step_dropout_reduction=2,
        gnn_step_layer_count=2,
        classifier_layer_count=3,
    )
    
    trained_model, history = model.learn(ml_model, train_loader, test_loader, val_loader, 20, pos_weight=pos_weight_val)
    
    hf.plot_history(history)
        
if __name__ == "__main__":
    main()