# Data Card

This dataset of traffic signs was created with multi-class classification in mind. It includes pictures of 58 distinct types of traffic signs. Every image has RGB colours and is in the ".jpg" format. This collection includes a number of classes, including yield signs, stop signs, speed limit signs, pedestrian crossings, and more. This dataset will be used to create a machine learning or deep learning model for the real-time recognition of traffic signs. All of the real-world situations were taken into consideration when compiling the dataset. Lighting conditions, darkness, angles, and occlusions are some of its variations.

## Source
- We got this data from Kaggle.
  - Link: https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification

## Purpose
- The purpose of this dataset is to classify traffic signs and address a critical part of Intelligent Transporation System and Autonomous driving.
1. Autonomous Driving: Self Driving cars must detect and classify the signs so that they follow the traffic rules and regulation and ensure road safety. The classification helps vehicles adapt to various speed limits, stops and prevent potential accidents.
2. Traffic Authorities can use Artificial Intelligence systems to monitor compliance with traffic laws.
3. Models trained on this dataset can be used in camera based system to identify the signs in real-time.

In this project, we have made a classification model each and then using real-time scenarios and situations the images in the dataset are transformed, distorted. The model is then trained on it as well as tested against these images. Idea behind this is to create a robust and strong classification model which can classify and detect the traffic signs with high accuracy and precision.


## About Dataset
- This dataset has images of 58 different traffic signs. 
- For each sign we have close to 120 images each. 
- In total we have 6,960 images. 
- The dataset contains 58 classes. 
- Each class has 120 images. 
```text
Data/
    DATA/
        0
        1
        2
        .
        .
        57
    labels.csv
```


## Preprocessing
We used the lables.csv and DATA folder to create two lists: 
- image_paths : 
  -   ['./Data/DATA/50/050_0013.png',
 './Data/DATA/50/050_0007.png',
 './Data/DATA/50/050_1_0019.png',
 './Data/DATA/50/050_1_0025.png',
 './Data/DATA/50/050_1_0024.png']
- Image_labels :
  - ['Go Right', 'Speed limit (30km/h)', 'Bicycles crossing', 'Horn', 'No entry']
In This way we assigned the traffic sign to it's respective image.
During the preprocessing, transformations like Gaussian Blur, Jitters, Random Rotation were applied to diversify the images and train the data on real-time scenarios.

Dataset was divided into train and test data. The size of each was:
Training set size: 3336
Testing set size: 834


## Author
- ALURU V N M HEMATEJA

## Owner
- ALURU V N M HEMATEJA

## License

- https://creativecommons.org/publicdomain/zero/1.0/


