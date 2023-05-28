import os
import argparse
from glob import glob
import numpy as np
import multiprocessing as mp
from PIL import Image
from torchvision.transforms import ToTensor, Lambda, RandomHorizontalFlip, ColorJitter, RandomAffine
import torchvision.transforms.functional as TF
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--jpg_path', type=str, required=True, help='Path to JPEG images directory.')
parser.add_argument('--png_path', type=str, required=True, help='Path to PNG images directory.')
parser.add_argument('--angle', type=int, required=True, help='Angle to rotate images.')

args = parser.parse_args()

inpf_path_jpg = args.jpg_path
inpf_path_png = args.png_path

list_wkk_jpg = sorted(glob(os.path.join(inpf_path_jpg, '*.jpg')))
list_wkk_png = sorted(glob(os.path.join(inpf_path_png, '*.png')))

flipper = RandomHorizontalFlip(p=1)
jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.04)

shear_value = 0.2  # シアーの値を固定

def shear_image(image, shear_value):
    width, height = image.size

    # アフィン変換行列を作成
    matrix = (1, shear_value, 0, 0, 1, 0)

    # シアー変換を適用
    sheared_image = image.transform((width, height), Image.AFFINE, matrix, Image.BICUBIC)

    return sheared_image

def process_files(files):
    file_jpg, file_png = files
    filename_jpg = os.path.splitext(os.path.basename(file_jpg))[0]
    filename_png = os.path.splitext(os.path.basename(file_png))[0]
    rotation_angle = np.random.uniform(-args.angle, args.angle)

    # Open image
    image_jpg = Image.open(file_jpg)
    image_png = Image.open(file_png)

    # Apply transformations
    for image, inpf_path, filename, fstring_suffix in [(image_jpg, inpf_path_jpg, filename_jpg, '.jpg'),
                                                       (image_png, inpf_path_png, filename_png, '.png')]:
        # Save original image
        image_orig = image.copy()
        image_flipped = flipper(image_orig)

        # For original image
        image_rot = image.rotate(rotation_angle)  # Rotate
        fstring_rot = filename + "-rot{}".format(args.angle) + fstring_suffix
        image_rot.save(os.path.join(inpf_path, fstring_rot))

        # Apply color jitter (for jpg only) and save
        if fstring_suffix == '.jpg':
            image_jitter = jitter(image)
            image_jitter.save(os.path.join(inpf_path, filename + "-jitter" + fstring_suffix))
        else:  # Save the png image without applying color jitter
            image.save(os.path.join(inpf_path, filename + "-jitter" + fstring_suffix))

        # Apply shear
        image_shear = shear_image(image, shear_value)
        image_shear.save(os.path.join(inpf_path, filename + "-shear" + fstring_suffix))

        # For flipped image
        image_flipped.save(os.path.join(inpf_path, filename + "-flip" + fstring_suffix))

        image_rot = image_flipped.rotate(rotation_angle)  # Rotate
        fstring_rot = filename + "-rot{}-flip".format(args.angle) + fstring_suffix
        image_rot.save(os.path.join(inpf_path, fstring_rot))

        # Apply color jitter (for jpg only) and save
        if fstring_suffix == '.jpg':
            image_jitter = jitter(image_flipped)
            image_jitter.save(os.path.join(inpf_path, filename + "-flip-jitter" + fstring_suffix))
        else:  # Save the png image without applying color jitter
            image_flipped.save(os.path.join(inpf_path, filename + "flip-jitter-" + fstring_suffix))

        # Apply shear
        image_shear = shear_image(image_flipped, shear_value)
        image_shear.save(os.path.join(inpf_path, filename + "-flip-shear" + fstring_suffix))

num_processes = mp.cpu_count()

with mp.Pool(num_processes) as p:
    list(tqdm(p.imap(process_files, zip(list_wkk_jpg, list_wkk_png)), total=len(list_wkk_jpg)))