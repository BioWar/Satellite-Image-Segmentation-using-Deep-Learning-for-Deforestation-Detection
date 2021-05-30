# How to use pre-trained model

1. Copy code for model creation in [Model_testing.ipynb](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Model_Testing.ipynb)
2. Use next line of code to create your model: 
```Python
model = build_unet(input_shape=(512, 512, 3),
                   filters=[2 ** i for i in range(5, int(np.log2(2048) + 1))],
                   batchnorm=False, transpose=False, dropout_flag=False)
```
3. Load model weights (Use model weights from current directory, download them using `git lfs pull`):
```Python
path_to_load = "U6_E:1201-F1:0.7134-IOU:0.6555.h5"
model.load_weights(path_to_load)
```
4. Exctract validation images from validation files: [first file](https://drive.google.com/file/d/1dWox7KNDmfg1Mpn5APPlRaaSh8QG194x/view?usp=sharing), [second file](https://drive.google.com/file/d/1yO2iPtvEXhSvjXRy09Lt3h7WbzYt8oea/view?usp=sharing). Using code from testing notebook.
5. Get an image by index (There are 23x2 images in every validation file from step 4):
```Python
image, mask = get_example(validation_filename_path, index_of_image)
```
6. Predict using code (Last steps are required to match colors of classes): 
```Python
image = image[np.newaxis, ...]
prediction = model.predict(image)
prediction_class1 = np.copy(prediction[..., 0]) # Forest
prediction_class2 = np.copy(prediction[..., 1]) # Deforest
prediction[..., 0] = prediction_class2 # RED - Deforest
prediction[..., 1] = prediction_class1 # GREEN - Forest
```
7. Visualize predictions:
```Python
plt.imshow(prediction[0])
```
                    
