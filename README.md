## BDA-Project-Team-06  
### Title : Smile Classification using Deep Learning Model  
 Dataset link : https://drive.google.com/drive/folders/1bzdrClsZCwuvQriAJatiayeVTeW4nMix?usp=sharing

### Project Title : Smile Classification
### Team : 06

### Story of the project:
With the advent of digital communication (Zoom, Google meat) it is hard to monitor and figure out the response of an individual who is on the hearing side (Audience in a show, students in a classroom). Deep learning models can help to monitor and find out the response of individuals which in turn can help to improve the content of the presenter or help how effectively the information is communicated to the audience. The motivation of this project is to develop a deep learning model to classify a given image as a positive smile or a negative smile (Fake). The dataset contains a total of 2000 samples for each class, to train a deep learning model from scratch using such high dimensional data(faces) from scratch it is not possible to get a modest generalization (test) accuracy. We opted for transfer learning which addresses the problem of data scarcity problem. This model is retrained using our dataset by freezing all the layers except for the final layer. The model is trained for a few epochs. We provided the plots of train accuracy and test accuracy. We created an interface, which randomly shows a few faces, the user is then allowed to choose any image to detect whether the smile of the person is positive smile or negative smile.

### Description :
We have considered a model named EfficientNet4 (trained ny Imagenet dataset publicly available) and used it for training our dataset that was collected from the internet. So, our dataset consists of images related to positive smile, negative smile, and no smile. We also have the data divided and labelled into train data and test data with the three class names. We used openCV, the widely used library for computer vision for displaying the facial recognition done in our project. 

### Datasets:
Imagenet and Cifar10 trained model  
Positive Smile Vs Negative Smile (Dataset)  
Positive Smile Vs No Smile (Dataset)

### Project Snippets :
Below are the screenshots of working project. Once the application is launched , it shows the images which are displayed with the smile type.  

![7](https://user-images.githubusercontent.com/79431387/146981222-f49775ea-2860-4ecc-9674-bf3c622124f9.png)  

![9](https://user-images.githubusercontent.com/79431387/146981536-00bcff0c-4ac4-4d19-ad46-98123a061e2c.png)  

![10](https://user-images.githubusercontent.com/79431387/146981595-ea71aa5e-5ec9-4a28-b44f-531b0ab7e35c.png)  

![13](https://user-images.githubusercontent.com/79431387/146981645-715138b0-e4c0-4679-9134-449c30cc4553.png)  

#### Video Link: https://youtu.be/iJuEFY_vTMI  

### Module Sharing:

### Data, Code and Reports:  
#### Ravali Bellamkonda (16307934) 
#### Munni Vidiyala (16296757)

### Presentation and front end:
#### Aparna Vennamaneni  
#### Lakshmi Avinash
