import io
import os

import json

import torch
from torchvision import transforms

from PIL import Image, ImageDraw

import random
import urllib.request
import zipfile
import shutil
import argparse
from typing import List, Dict


# Define a class for COCO dataset
class COCO_dataset:
    def __init__(self, img_file: str, annotation_file: str, transformer: str):
        """
           Initialize COCO_dataset object.
           Args:
           img_file (str): path to image folder.
           annotation_file (str): path to COCO annotations file.
           transformer (str): data augmentation transformer to use.
           """
        self.img_file = img_file
        self.transformer = transformer
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)
        self.annotations = annotations
        self.new_annotations = {'info': self.annotations['info'],
                                'licenses': self.annotations['licenses'],
                                'images': [],
                                'annotations': [],
                                'categories': self.annotations['categories']
                                }

    def __getitem__(self, index: int):
        """
           Get a specific image and its annotations.
           Returns:
           A tuple containing:
               - The augmented image (PIL Image object).
               - The augmented image's annotations (dict).
               - The augmented annotations (list of dict).
           Args:
           index (int): index of the image and annotations to retrieve.
           """
        img_ann_img = self.annotations['images'][index]
        img_id = img_ann_img['id']
        img_filename = img_ann_img['file_name']

        img_ann_ann = [item for item in self.annotations['annotations'] if item['image_id'] == img_id]

        img_path = os.path.join(self.img_file, img_filename)
        img = Image.open(img_path).convert('RGB')

        if self.transformer is not None:
            # Apply the specified image augmentation

            augmented_annotation_img = {
                'id': img_id,
                'license': img_ann_img['license'],
                'coco_url': img_ann_img['coco_url'],
                'flickr_url': img_ann_img['flickr_url'],
                'width': img_ann_img['width'],
                'height': img_ann_img['height'],
                'file_name': img_filename,
                'date_captured': img_ann_img['date_captured'],

            }

            if self.transformer == 'flip':
                augmented_img, augmented_annotation_ann = flip(img, img_ann_ann)
                augmented_annotation_img['file_name'] = 'Aug_flip_' + str(img_filename)

            elif self.transformer == 'scale':
                augmented_img, augmented_annotation_ann = scale(img, img_ann_ann)
                augmented_annotation_img['file_name'] = 'Aug_scale_' + str(img_filename)

                # update the width and height after scaling
                new_width, new_height = augmented_img.size
                augmented_annotation_img['width'] = new_width
                augmented_annotation_img['height'] = new_height

            elif self.transformer == 'noise_injection':
                augmented_img = noise_injection(img)
                augmented_annotation_img['file_name'] = 'Aug_noise_' + str(img_filename)
                augmented_annotation_ann = img_ann_ann

        return augmented_img, augmented_annotation_img, augmented_annotation_ann

    def saveImage(self, augmented_img, augmented_annotation_img: Dict, augmented_annotation_ann: List[dict],
                  output_img_path: str, output_ann_path: str):
        # Saving the augmented image and annotations to the specified output paths

        self.new_annotations['images'].append(augmented_annotation_img)
        for item in augmented_annotation_ann:
            self.new_annotations['annotations'].append(item)

        name = augmented_annotation_img['file_name']
        augmented_img_path = os.path.join(output_img_path, name)
        augmented_img.save(augmented_img_path)

        with open(output_ann_path, 'w') as f:
            json.dump(self.new_annotations, f)

    def __len__(self):
        # Return the number of files in the img_file directory
        return len([name for name in os.listdir(self.img_file) if os.path.isfile(os.path.join(self.img_file, name))])

    def __iter__(self):
        # Return an iterator that yields COCO_dataset objects
        for i in range(len(self)):
            yield self[i]


def flip(img, img_ann_ann: List[dict]):
    # Define a flip transformation
    flip_transform = transforms.RandomHorizontalFlip(p=1)
    augmented_img = flip_transform(img)

    # Apply the flip transformation to the annotations and updating bounding boxes
    new_anns = []
    for item in img_ann_ann:
        box = item['bbox']
        img_width, img_height = img.size
        x, y, w, h = box
        new_x = img_width - x - w  # Compute the new x-coordinate
        new_bbox = [new_x, y, w, h]
        item['bbox'] = new_bbox
        new_anns.append(item)
    return augmented_img, new_anns


def scale(img, img_ann_ann: List[dict]):

    # define the size range
    min_height = 250
    max_height = 650
    min_width = 350
    max_width = 850

    # randomly choose the height and width
    crop_height = random.randint(min_height, max_height)
    crop_width = random.randint(min_width, max_width)

    # Define a scaling transformation
    scaling_transform = transforms.Resize((crop_height, crop_width))
    augmented_img = scaling_transform(img)

    # Apply the scale transformation to the annotations and updating bounding boxes
    new_anns = []
    for item in img_ann_ann:
        box = item['bbox']
        x, y, w, h = box
        x_scale, y_scale = augmented_img.size[0] / img.size[0], augmented_img.size[1] / img.size[1]
        x_new, y_new, w_new, h_new = x * x_scale, y * y_scale, w * x_scale, h * y_scale
        new_bbox = [x_new, y_new, w_new, h_new]
        item['bbox'] = new_bbox
        new_anns.append(item)

    return augmented_img, new_anns


def noise_injection(img):
    # Add Gaussian noise to the image
    noise_factor = 0.1
    im = transforms.ToTensor()(img)
    noisy = im + torch.randn_like(im) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    augmented_img = transforms.ToPILImage()(noisy)

    return augmented_img


def main(transformer: str, task: str, download):

    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)

    if download == 'true':

        # Download the annotation file from a URL and unzip it
        print('Downloading dataset, please wait...')
        url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        response = urllib.request.urlopen(url)
        data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zip_ref:
            zip_ref.extractall(dir_path)

        # Download the images file from a URL and unzip it
        url = 'http://images.cocodataset.org/zips/val2017.zip'
        response = urllib.request.urlopen(url)
        data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zip_ref:
            zip_ref.extractall(dir_path)

    img_path = os.path.join(dir_path, 'val2017')
    annotation_path = os.path.join(dir_path, 'annotations/instances_val2017.json')

    # creating the output folders

    output_img_path = os.path.join(dir_path, "augmented_images")
    output_ann_path = os.path.join(dir_path, "augmented_annotation")

    shutil.rmtree(output_img_path)
    shutil.rmtree(output_ann_path)

    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)

    if not os.path.exists(output_ann_path):
        os.mkdir(output_ann_path)

    output_ann_path = os.path.join(output_ann_path, "annotations.json")

    # Create a COCO dataset object
    coco = COCO_dataset(img_path, annotation_path, transformer)

    # If task is 'all', augment all images of the dataset and their annotations and save them
    if task == 'all':
        for item in coco:
            augmented_img, augmented_annotation_img, augmented_annotation_ann = item
            coco.saveImage(augmented_img, augmented_annotation_img, augmented_annotation_ann, output_img_path,
                           output_ann_path)

    # If task is 'example', augment the first 10 images and their annotations as an example and save them
    elif task == 'example':
        for i in range(10):
            augmented_img, augmented_annotation_img, augmented_annotation_ann = coco[i]
            coco.saveImage(augmented_img, augmented_annotation_img, augmented_annotation_ann, output_img_path,
                           output_ann_path)


if __name__ == "__main__":

    # Add command-line arguments for the task and transformer
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--transformer',  type=str)
    parser.add_argument('-o', '--download', help='an optional argument', default='true')
    args = parser.parse_args()

    main(args.transformer, args.task, args.download)
