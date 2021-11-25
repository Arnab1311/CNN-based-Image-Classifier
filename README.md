# CNN-based-Image-Classifier
Simple CNN based image classifier which classifies an input image as Dog or cat.


![This is an image](/Images/imp_image.png)

## Download an Image dataset from
1. [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
2. [Tensorflow](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

## How to create a custom dataset
1. Google search cat and dog.
2. Add [Image assistant Batch Image downloader](https://chrome.google.com/webstore/detail/imageassistant-batch-imag/dbjbempljhcmhlfpfacalomonjpalpko?hl=en) to chrome extension. 
3. Use the extension to download all the images.
4. Clean the dataset
  - Remove unwanted images
  - Run the [script](image_finder.py) to identify smaller resolution images.
 
## Dataset Preparation
Run dataset.py for preparing the dataset.

## Training the model
1. Train the model using train.py
2. Configure the network

## Inference
Run predict.py to see the results for binary classification of dogs and cats


