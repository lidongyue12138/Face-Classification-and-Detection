# Project 1: Face Classification and Detection

### TO DO

1. Face Classification
   - Data Processing
     - Generate positive samples: by the annotation of the dataset
     - Generate negative samples: by resizing or by sliding windows
     - Extract HOG features
     - Generate training set and testing set
   - Model Building
     - Logistic Model
       - SGD
       - Langevin dynamics
     - Fisher Model
     - SVM Model
     - CNN Model
   - Visualization
     - Visualize bounding boxes of positive and negative samples on images. 
     - Visualize the extracted HOG features 
     - Visualize face detection results and feature distribution 
2. Face Detection

### Tasks

- [x] Visualize bounding boxes of positive and negative samples on images. 
- [x] Visualize the extracted HOG features 
- [x] Learning a logistic model for classification: SGD & Langevin
  - [ ]   Report the accuracy 
- [ ] Learning a fisher model for classification 
  - [ ] Report the accuracy, the intra-class variance and the inter-class variance 
- [ ] Learning SVMs for classification 
  - [ ] Report the accuracy 
  - [ ] List samples of support vectors (i.e., the samples whose the margin =1) 
- [ ] Learning convolutional neural networks for classification 
- [ ] Visualize face detection results and feature distribution 

### ToDo

- resplit training and testing set

- fix each model performance
- create api for face detection in each model: predict()