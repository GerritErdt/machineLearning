from src.models import model
import src.dataLoading.data_loading as dl
import src.helperFunctions.helper_functions as hf

def main():
    hf.set_all_seeds()
    train_loader, test_loader, pos_weight_val = dl.get_stereo_clean_dataset(50000, batch_size=128)
    
    ml_model = model.GNNModel()
    trained_model, history = model.learn(ml_model, train_loader, test_loader, 10, pos_weight=pos_weight_val)

    batch_data = next(iter(test_loader))
    ml_model.save(
        batch_data.x,
        batch_data.edge_index,
        batch_data.batch,
        batch_data.num_graphs,
        path="./model.onnx"
    )
        
    hf.plot_history(history)
        
if __name__ == "__main__":
    main()