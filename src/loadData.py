import pandas as pd
import numpy as np

# Definition of the column names for Hillas parameters and image data, to be extracted from the parquet file(s)

HILLAS_PARAMETERS_COLUMN_NAMES = [
    "hillas_length_m1",
    "hillas_width_m1",
    "hillas_delta_m1",
    "hillas_size_m1",
    "hillas_cog_x_m1",
    "hillas_cog_y_m1",
    "hillas_sin_delta_m1",
    "hillas_cos_delta_m1",
    "hillas_length_m2",
    "hillas_width_m2",
    "hillas_delta_m2",
    "hillas_size_m2",
    "hillas_cog_x_m2",
    "hillas_cog_y_m2",
    "hillas_sin_delta_m2",
    "hillas_cos_delta_m2",
]

IMAGE_COLUMN_NAMES = [
    "clean_image_m1",
    "clean_image_m2",
]


def loadData(filePath):
    # Returns (hilas_parameters, image data telescope 1, image data telescope 2)
    df = pd.read_parquet(filePath)
    
    hillasParameters = df[HILLAS_PARAMETERS_COLUMN_NAMES].to_numpy(dtype=np.float32)
    imageDataM1 = np.stack(df[IMAGE_COLUMN_NAMES[0]].to_numpy())
    imageDataM2 = np.stack(df[IMAGE_COLUMN_NAMES[1]].to_numpy())
    
    return hillasParameters, imageDataM1, imageDataM2

def printDataSample(filePath, sampleIndex=0):
    hillasParameters, imageDataM1, imageDataM2 = loadData(filePath)
    
    print("Hillas Parameters Sample:")
    print(hillasParameters[sampleIndex])
    
    # Print some general info about the image, without showing it
    print("\nImage Data Telescope 1 Sample Info:")
    print(f"Shape: {imageDataM1[sampleIndex].shape}, Min: {imageDataM1[sampleIndex].min()}, Max: {imageDataM1[sampleIndex].max()}")
    print("\nImage Data Telescope 2 Sample Info:")
    print(f"Shape: {imageDataM2[sampleIndex].shape}, Min: {imageDataM2[sampleIndex].min()}, Max: {imageDataM2[sampleIndex].max()}")
    

if __name__ == "__main__":
    sampleFilePath = "./data/magic-gammas-new-1.parquet"
    printDataSample(sampleFilePath, sampleIndex=0)