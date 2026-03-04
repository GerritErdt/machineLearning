from src.models import model
import src.dataLoading.data_loading as dl
import src.helperFunctions.helper_functions as hf

def main():
    hf.set_all_seeds()
    train_loader, test_loader, pos_weight_val = dl.get_stereo_clean_dataset(75000, batch_size=128)
    
    ml_model = model.GNNModel()
    trained_model, history = model.learn(ml_model, train_loader, test_loader, 25, pos_weight=pos_weight_val)
        
    hf.plot_history(history)
        
if __name__ == "__main__":
    main()