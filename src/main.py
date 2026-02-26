from models import model
import dataLoading.dataLoading as dl

def main():
    m1_train, m1_test = dl.get_stereo_clean_dataset(20000)
    # mlModel = model.train_model_m1(m1_train, m1_test, 30)
    model.train_model_m1(m1_train, m1_test, num_epochs=30, batch_size=64)


if __name__ == "__main__":
    main()