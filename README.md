# Vehicle-Image-Recognition

This script trains a convolutional neural network model to recognise a vehicle's make and model. This was done using Google's [Tensorflow](https://www.tensorflow.org/) and [TFlearn](http://tflearn.org/), a deep learning library built on top of Tensorflow. 

## Dataset
The dataset for this project was downloaded on [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/data). It contains XXXX vehicle images of identical size. There are a total of XXXX vehicles in the dataset, each of them containing 16 photos of the car, taken from different angles.

A 'metadata.csv' file is also provided, which contains the labelled make and model of each car.

## Installing software and libraries

This model is computationally demanding and requires to use TensorFlow on GPU (as opposed to CPU). An NVIDIA graphics card is thus necessary. This model was trained using a GeForce GTX 730 graphics card with XXXX GB of memory.

I followed [this tutorial](https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/) to install TensorFlow. 
NB: TensorFlow-GPU runs on Python 3.5 (but not for above versions as of 17/05/2018).

1. Instal graphics card:
The first step is to install a graphics card and the corresponding drivers on your computer

2. Then CUDA needs to be installed. CUDA is a parallel computing platform and application programming interface model created by Nvidia. 

3. Then CuDNN (CUDA Deep Neural Network library) needs to be downloaded and "installed".

4. Install TensorFlow-gpu and tflearn in an Anaconda environment running Python 3.5 

```
conda install -c anaconda tensorflow-gpu
python -m pip install tflearn
```

5. Install additional libraries
```
conda install matplotlib
conda install pandas
conda install -c anaconda scikit-learn 
```

## Data pre-processing

The images in the dataset are very big (1918 x 1280). This is causing memory issues for the model. The images were thus resized to something more managable that nonetheless allows the visual recognition of a car make and model ( XXXX x XXXX). The top part of the image diosplayuing the name of the used car dealers company was cropped.

The 

NB: do not use conda forge to install any library (such as scikit-learn) as this will break your Anaconda and will lead to Spyder not working.
