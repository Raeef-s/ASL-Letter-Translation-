# ASL Webcam Letter Translation Project
## Overview
This project is designed to detect American Sign Language (ASL) letters from a webcam. It uses a Feedforward Neural Network, recieving input from Mediapipe Hand Landmark detection.  

# Features
- Classifies 28 different hand symbols, including letter A-Z along with Space, and Delete
- Model architecture trained with Keras in Tensorflow
- Displays current predictions in pop up window of frame using OpenCV

# Libraries
- OpenCV 4.11.0
- Numpy 2.0.1
- Tensorflow 2.18.0
- Mediapipe 0.10.21
- Pandas 2.2.3

## Dataset
ASL Alphabet by Akash:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
(87 000 training images)

## Preprocessing
To preprocess the data, each image training example was fed through Mediapipe Hand Landmark Detection, to generate an array of positional data in the hand. 
Each inference generated 21 landmarks, of which contained 5 different attributes (X position, Y position, Z position, Visibility, Presence) for a total of 104 data points. 

More information about the Hand Landmark Detection can be found here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Each feature was then appended to seperate lists within a larger list (referred to as dataColumns in DataToCSV), with each individual list acting as a column for each data type. 
The classification of the image was also added as a separate column, bringing the total number of columns to 106. This process was repeated, until every training image had been added.

The list was then converted into a Pandas DataFrame, to save as "MediapipePredictions.csv". No preprocessing was done to the DataFrame to remove empty columns, as to preserve the original length. 

## Training and Testing

The dataset was split 80-10-10 into training, test, and cross validation sets. 

To train the model, all empty rows of predictions were dropped. This resulted in an uneven distribution of training examples, which was accounted for by calculating the weights of each class. All 105 value columns were passed in as an input vector into the model. 

The model architecture consisted of three layers, including two inital 128 neuron dense layers with a ReLU activation and a final softmax activation for each classification. In addition, the first two dense layers used batch normalization and 40% dropout.

### Accuracy

The model was trained for 100 epochs, with a consistent batch size of 30. The model does not appear to completely converge, but shows minimal return for additional computing. 

#### Epoch 100 Training Accuracy: 0.9177
#### Epoch 100 Training Loss: 0.2826
#### Epoch 100 Validation Accuracy: 0.9549
#### Epoch 100 Validation Loss: 0.1302
#### Test Accuracy: 0.9539
#### Test Loss: 0.1372



