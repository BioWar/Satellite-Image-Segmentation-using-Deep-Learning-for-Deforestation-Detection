# Dataset exploration

Dataset was created by parsing areas of Ukrainian forestries using PyAutoGui and Google Earth Pro, the task required high-quality images and usage of such datasets as Landsat was not good enough. The program has gathered 322 images from three different regions. Images contain not only areas with forest or deforestation, but areas with roads, villages located inside the forest, open sources of water like rivers, and ponds. To make our model more robust and predict areas of deforestation more accurately. Without the third class, our prediction would be correct only with images that contain forest, and deforestation. Images were cropped to the size 512x512 pixels and saved with the corresponding masks in Google Cloud Storage.

