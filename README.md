# Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection

[![Open TPU_Training_Strategy in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/blob/main/Model_Training_TPU_Strategy.ipynb)<br>

**Dataset link to Google Drive:** https://drive.google.com/drive/folders/1z96jnNwC0NSxggXsBcfnmoe-FiniOr-t?usp=sharing

**Pretrained model link to Google Drive:** https://drive.google.com/file/d/1Z0oIc3lc7VFNDclUnOTma2eZg4UTJmBg/view?usp=sharing

## Abstract

The past years automation process of various tasks using Deep
Learning techniques was proved to be successful, in this paper this approach
was used to create an image segmentation model for monitoring the
deforestation process, and efficiently prevent illegal deforestation. As a model
architecture, the U-Net architecture was taken. For such a specific task the
dataset was created which contains 322 images and corresponding masks of a
forest, deforestation areas, and areas that are none of the above. All materials
about this research will be hosted online to have an opportunity to recreate the
results of this paper and improve them.

## Keywords

Deep learning, convolutional neural network, TensorFlow, TPU,
image segmentation, U-Net, deforestation.

## Inspiration

1. Laurence Moroney guide for distributed training: https://www.coursera.org/lecture/custom-distributed-training-with-tensorflow/tpu-strategy-etSkY
2. U-Net: Convolutional Networks for BiomedicalImage Segmentation: https://arxiv.org/pdf/1505.04597.pdf
3. AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volumeSegmentation of Head and Neck Anatomy: https://arxiv.org/pdf/1808.05238.pdf
4. Tfrecords tutorial: https://www.tensorflow.org/tutorials/load_data/tfrecord
5. Chest X-Ray Analysis of Tuberculosis by Deep Learning with Segmentation and Augmentation: https://arxiv.org/ftp/arxiv/papers/1803/1803.01199.pdf


## Dataset

To display images saved in Dataset/deforestation_dataset.tar.gz you should extract tfrecords files from it and run next code example with tfrecord specifications.

Example of code to load an image from the tfrecord file:
```Python
def get_example(dataset_path, image_num):
  """
    dataset_path : path to tfrecord file;
    image_num : idx of the image you want to display
  """
  raw_image_dataset = tf.data.TFRecordDataset(dataset_path)
  # Create a dictionary describing the features.
  image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
  }

  def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

  parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
  for image_features in parsed_image_dataset:
    if image_num != 0:
      image_raw = image_features['image'].numpy()
      mask_raw = image_features['mask'].numpy()
      image_num -= 1
    else:
      break
  image = tf.image.decode_png(image_raw, channels=3)
  image = tf.image.resize(image, [512, 512])
  image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
  mask = tf.io.decode_raw(mask_raw, out_type="float")
  mask = tf.reshape(mask, [512, 512, 3])
  mask = tf.cast(mask, tf.float32)
  return image, mask
```
