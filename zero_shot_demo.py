import os
import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model.model import CLIP

from utils.simple_tokenizer import SimpleTokenizer
from utils import set_seed, mkdir, setup_logger, load_config_file

import argparse
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt


MODEL_CONFIG_PATH = 'model/model_config.yaml'

def tokenize(texts, tokenizer, context_length=77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    '''
    Creates texts for each class using templates and extracts their text embeddings.
    '''
    print("Getting text features from classnames")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(' '.join(classname.split('_'))) for template in templates] #format with class
            print("class texts :")
            print(texts)
            texts = tokenize(texts, tokenizer).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights



def predict_class(model, images, image_names, dataset_classes, tokenizer, device):
    '''
    Classifies images by predicting their classes from "dataset_classes"
    '''
    with torch.no_grad():
        
        classnames = [classname for classname in dataset_classes]
        
        templates = ["a photo of a {}."]
        zeroshot_weights = zeroshot_classifier(model, classnames, templates, tokenizer, device)
        # print("zeroshot_weights.shape", zeroshot_weights.shape)
        predictions = []

        for image, image_name in zip(images, image_names):
            image_input = image.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity_scale = 35.0
            similarity = (similarity_scale * image_features @ zeroshot_weights).softmax(dim=-1)
            
            # top 5 predictions
            values, indices = similarity[0].cpu().topk(5)
            print("------------------------")
            print("img : ", image_name)
            print("predicted classes :")
            for value, index in zip(values, indices):
                print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")
            print("------------------------")   

            predictions.append((values, indices))     
    
    return predictions                     

def show_predictions(images, predictions, dataset_classes, save_dir):
    '''
    To give predictions in a nice figure
    '''

    if len(images) == 1:
        # zero-shot demo on a single image only
        image = images[0]
        top_probs =  [prediction[0] for prediction in predictions]
        top_labels = [prediction[1] for prediction in predictions]

        plt.figure(figsize=(8, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        y = np.arange(top_probs[0].shape[-1])
        plt.grid()
        plt.barh(y, top_probs[0])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset_classes[index] for index in top_labels[0].numpy()])
        plt.xlabel("probability")

        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "single_img_demo.png"))
        plt.show()

        return

    # for images in a directory
    plt.figure(figsize=(16, 2*(len(images))))
    top_probs = [prediction[0] for prediction in predictions]
    top_labels = [prediction[1] for prediction in predictions]

    for i, image in enumerate(images):
        plt.subplot(len(images)//2, 4, 2 * i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(len(images)//2, 4, 2 * i + 2)
        y = np.arange(top_probs[i].shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset_classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "demo.png"))
    plt.show()

def zero_shot_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="path of saved weights")
    parser.add_argument("--img_dir", default="test_images", type=str, required=False, help="directory containing test images. Please have even number of images for a nice demo figure")
    parser.add_argument("--img_path", default=None, type=str, required=False, help="Path of an image to classify")
    parser.add_argument("--show_predictions", action='store_true', help="To show predictions in a figure")

    args = parser.parse_args()

    demo_output_dir = "demo_output"
    # creating directory to store demo result
    mkdir(path=demo_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_config = load_config_file(MODEL_CONFIG_PATH)

    # Image transform and text tokenizer
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
    ])

    transform_no_norm = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    tokenizer = SimpleTokenizer()

    # CIFAR100 classes
    dataset_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    # print(dataset_classes)

    # creating RN50 CLIP model
    model_params = dict(model_config.RN50)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = CLIP(**model_params)

    # loading trained weights
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    images = []
    image_names = []
    raw_images = []

    if args.img_path:
        # zero-shot demo on a single image only
        img_path = args.img_path
        image_name = os.path.split(img_path)[-1]
        image = transform(Image.open(img_path)).unsqueeze(0)
        raw_image = transform_no_norm(Image.open(img_path))        
        raw_images.append(raw_image)
        images.append(image)
        image_names.append(image_name)
    
    else :
        # zero-shot demo for images in a directory
        for img_path in glob(args.img_dir + '/*'):
            image_name = os.path.split(img_path)[-1]
            image = transform(Image.open(img_path)).unsqueeze(0)

            # un normalized image for display
            raw_image = transform_no_norm(Image.open(img_path))        
            raw_images.append(raw_image)
            images.append(image)
            image_names.append(image_name)
    
    predictions = predict_class(model, images, image_names, dataset_classes, tokenizer, device)

    if args.show_predictions:
        if args.img_path:
            
            try :
                show_predictions(raw_images, predictions, dataset_classes, demo_output_dir)
                print("==========")
                print(f"Please check the following for zero-shot prediction demo figure")
                print(" -- ", os.path.join(demo_output_dir, "single_img_demo.png"))
            except:
                print("Some error while generating demo figure for a single image.")

        else :
            # zero-shot demo for images in a directory
            try :
                show_predictions(raw_images, predictions, dataset_classes, demo_output_dir)
                print("==========")
                print(f"Please check the following for zero-shot prediction demo figure")
                print(" -- ", os.path.join(demo_output_dir, "demo.png"))
            except:
                print("Some error while generating demo figure. Please try putting even number of images in images directory for a nice demo figure.")

if __name__ == "__main__":
    zero_shot_demo()