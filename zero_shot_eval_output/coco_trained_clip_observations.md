# Observations : CLIP trained on MS-COCO Captions

## COCO-trained CLIP learns img-text alignment
Even when the CLIP model is trained on a relatively small dataset of MS-COCO Captions containing 118k images, it is able to learn image-text alignment as seen from the following figures.

Img-text similarity calculated using COCO-trained CLIP on a few COCO val set images-captions :
![img-text similarity from COCO trained CLIP](/figures/img_text_sim_coco_val.png)


Img-text similarity calculated using COCO-trained CLIP on few other images
![img-text similarity from COCO trained CLIP](/figures/img_text_sim.png)

## COCO-trained CLIP performs reasonably well zero-shot classification for simple images
To perform zero-shot classification, I created sentences from 100 CIFAR-100 classes using the template "a photo of a _{class name}_." and found img-text similarity score using CLIP for all the class sentences for a given image. The class having the highest similarity with the image is the predicted one. 

COCO-trained CLIP does perform reasonably well for simples images as seen in the following figure : 

![img-text similarity from COCO trained CLIP](/figures/demo.png)

## COCO-trained CLIP does not perform well when used for zero-shot classification on images from Imagenet (and other datasets)
MS-COCO is a very small dataset compared to WebImageText (118K vs 400 Million). MS-COCO is also not as diverse as WebImageText. While the vastly diverse WebImageText possibly covers all kinds of images from the internet (natural scenes, sketches, blurry images, low-res images, texts, websites, etc.), MS-COCO mostly covers natural and day-to-day scenes. Further, a large number of categories in imagenet (e.g. stingray, tarantula, mousetrap, etc.) are not present in MS-COCO images and their caption texts. Thus the CLIP model trained on a small dataset like MS-COCO does not perform well on zero-shot imagenet classifications. CLIP needs to be trained on large and diverse datasets.

## zero-shot classification results on different datasets :

Dataset          | Top-1 Accuracy | Top-5 Accuracy
------------     | -------------  | ----------------
Imagenet Val set | 4.09  %        | 10.77 %
ImagenetV2       | 4.14  %        | 10.84 %
ImageNet-A       | 1.61  %        | 6.85  %
ImageNet-R       | 3.84  %        | 10.99 %
ImageNet-sketch  | 0.66  %        | 2.43  %
CIFAR10          | 15.07 %        | 66.8  %
CIFAR100         | 2.61  %        | 11.64 %

Class wise accuracies can be found in "zero_shot_eval_output" directory as .txt files.

## To run evaluation code : 

**Trained weights** : [link](https://drive.google.com/file/d/1BVEY4WeFmQb3wv0A6RaLyVjnc7qmChH2/view?usp=sharing)

```
python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt> --dataset_type <'cifar' or 'imagenet'> --data_dir <path to dataset eval images directory>
```
Check "zero_shot_eval_output" directory for evaluated class wise accuracies (as .txt files) once you run the eval code

#### For Imagenet-A
```
# download data
$ wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -O data/imagenet-a.tar
$ tar -xf data/imagenet-a.tar -C data

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type imagenet --data_dir data/imagenet-a
```

#### For Imagenet-r
```
# download data
$ wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -O data/imagenet-r.tar
$ tar -xf data/imagenet-r.tar -C data

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type imagenet --data_dir data/imagenet-r
```

#### For ImagenetV2
```
# download data
$ wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz -O data/imagenetv2-matched-frequency.tar.gz
$ tar -xf data/imagenetv2-matched-frequency.tar.gz -C data

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type imagenet --data_dir data/imagenetv2-matched-frequency-format-val
```

#### For Imagenet-sketch
```
# first install gdown to download dataset from google drive
$ pip install gdown

# download data
$ gdown --id 1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA -O data/ImageNet-Sketch.zip
$ unzip data/ImageNet-Sketch.zip -d data

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type imagenet --data_dir data/Imagenet-sketch
```

#### For CIFAR10
```
# create directory for CIFAR10 dataset
$ mkdir data/CIFAR10

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type cifar --data_dir data/CIFAR10
```

#### For CIFAR100
```
# create directory for CIFAR100 dataset
$ mkdir data/CIFAR100

# run code to eval
$ python zeroshot_eval.py --checkpoint_path <path_to_trained_checkpoint.pt>  --dataset_type cifar --data_dir data/CIFAR100
```
