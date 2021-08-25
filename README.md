# CLIP training

This repository contains code to train [CLIP](https://github.com/openai/CLIP) on [MS-COCO](https://cocodataset.org/#home) captions. 
Can be easily modified to train on other multi-modal datasets (OpenImages, Conceptual captions, ...).

## Requirements
* Use **python > 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.7.0 CUDA 10.2**

* Other requirements from 'requirements.txt'

**To setup environment**
```
# create new env clip_train
$ conda create -n clip_train python=3.8.5

# activate clip_train
$ conda activate clip_train

# install pytorch, torchvision
$ conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

# install other dependencies
$ pip install -r requirements.txt
```
## Training
### Preparing training dataset
MS-COCO training set images and their captions are used for training the CLIP model. 
To download the dataset :

```
# create directory in data/
$ mkdir data/mscoco

# download images
$ wget http://images.cocodataset.org/zips/train2017.zip -O data/mscoco/train2017.zip
$ unzip data/mscoco/train2017.zip -d data/mscoco


# download annotations 
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco/annotations_trainval2017.zip
$ unzip data/mscoco/annotations_trainval2017.zip -d data/mscoco
```

To check and update training parameters, model config and dataset paths please see the following config files : 
```
trainer/train_config.yaml   # training parameters
model/model_config.yaml     # CLIP model config
dataloader/data_config.yaml # training dataset path

``` 

### To train : 
Take dataset paths from 'dataloader/data_config.yaml'
```
$ python train.py 
```

OR, give dataset path as cl args
```
$ python train.py --train_img_dir <path to training images directory> --train_annotation_file <path to annotation file>
```
### Training setting : 
* Model config : Since MS-COCO is relatively small dataset, I used ResNet50 as image encoder instead of Vision Transformer. Further, I also reduced the number of transformer layers to 6 in text encoder. Detailed model config is here : [model_config.yaml](/model/model_config.yaml)

* Batch size : 256. I trained using 4 GTX1080 GPUs (64 batch size per gpu).   

* Optimizer : Adam optimizer with weight decay.

* Scheduler : Cosine Scheduler with warmup for first 20% of gradient update steps.
  Detailed training config is here : [train_config.yaml](/trainer/train_config.yaml)

* Temperature parameter clipping : Added temperature clipping as mentioned in the paper for training stability. The learnable temperature parameter is clipped to prevent scaling the logits by more than 100.

## Zero-shot classification :
For zero-shot classification, first all class names are converted into sentences using templates (like "a photo of a {class name}") and their text embeddings are computed using CLIP. Then to classify an image, first image embedding is computed using CLIP and then its cosine similarity with all the class sentences embeddings is computed to predict the class with the highest cosine similarity.

### Zero-shot demo :

**Trained weights** : 

- Download trained checkpoint from google drive : [link](https://drive.google.com/file/d/1BVEY4WeFmQb3wv0A6RaLyVjnc7qmChH2/view?usp=sharing) 
- Or use gdown to download it : 
  ```
  # first install gdown
  $ pip install gdown

  # then download trained weights at 'saved_checkpoints/trained_checkpoint.pt'
  $ mkdir saved_checkpoints
  $ gdown --id 1BVEY4WeFmQb3wv0A6RaLyVjnc7qmChH2 -O saved_checkpoints/trained_checkpoint.pt  
  ```

To classify image(s) into CIFAR100 classes, run the following

```
# to classify a single image
$ python zero_shot_demo.py --checkpoint_path <path_to_trained_checkpoint.pt> --img_path <path_to_img.jpg> --show_prediction

# to classify all images images in a directory
$ python zero_shot_demo.py --checkpoint_path <path_to_trained_checkpoint.pt> --img_dir <path_to_img_directory> --show_prediction

# --show_prediction flag is to save a prediction figure with class probabilities
# NOTE : Please put even number of images in img_directory to get a nice prediction figure
```

Example to run zero-shot demo:
```
# first put trained weights at saved_checkpoints/trained_checkpoint.pt 

# for single image
$ python zero_shot_demo.py --checkpoint_path saved_checkpoints/trained_checkpoint.pt --img_path test_images/bicycle.jpeg --show_prediction

# for an image directory
$ python zero_shot_demo.py --checkpoint_path saved_checkpoints/trained_checkpoint.pt --img_dir test_images --show_prediction

# view prediction figure in "demo_output" directory
```

### Zero-shot evaluation on vision datasets + observations :
For evaluation results and instructions on how to run eval code, check this : [Observations and Eval results](/zero_shot_eval_output/coco_trained_clip_observations.md)
