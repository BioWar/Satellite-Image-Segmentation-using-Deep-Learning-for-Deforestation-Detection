# Dataset exploration

## Background 

Dataset was created by parsing areas of Ukrainian forestries using PyAutoGui and Google Earth Pro, the task required high-quality images and usage of such datasets as Landsat was not good enough. The program has gathered 322 images from three different regions. Images contain not only areas with forest or deforestation, but areas with roads, villages located inside the forest, open sources of water like rivers, and ponds. To make our model more robust and predict areas of deforestation more accurately. Without the third class, our prediction would be correct only with images that contain forest, and deforestation. Images were cropped to the size 512x512 pixels and saved with the corresponding masks in Google Cloud Storage.

## EDA

![Class distribution](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Images/Dataset_EDA/distribution_of_classes_log_diagram.png "Fig.1")

From Fig. 1 it is visible that the amount of pixel from the ''Forest'' class is one order bigger than the number of pixels in ''Deforestation'' class, and ''Other'' class. This means that the model could predict the ''Forest'' class all the time and get high accuracy. 


![Distribution of class areas sizes](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Images/Dataset_EDA/Whiskers_area_distribution.png "Fig.2")

Fig. 2 shows that for good accuracy, the model should predict very small areas of deforestation. Small areas of forest might be single trees, but it’s still important to differentiate them from other classes. The forest area takes 77\% taking into account Fig. 3 which means that blind prediction of our model that all pixels belonging to the forest will give it 77\% accuracy.

![Average percentage of class area per image](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Images/Dataset_EDA/Average_percentage_of_class_area_per_image.png "Fig.3")

The forest area takes 77\% taking into account Fig. 3 which means that blind prediction of our model that all pixels belonging to the forest will give it 77\% accuracy.

## Example of data

![Image example](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Images/Dataset_EDA/image.png "Fig.4")
![Mask example](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Images/Dataset_EDA/mask.png "Fig.5")
