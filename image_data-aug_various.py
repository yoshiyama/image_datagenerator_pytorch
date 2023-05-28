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
shearer = RandomAffine(degrees=0, shear=20)

def process_files(files):
    file_jpg, file_png = files
    filename_jpg = os.path.splitext(os.path.basename(file_jpg))[0]
    filename_png = os.path.splitext(os.path.basename(file_png))[0]
    rotation_angle = np.random.uniform(-args.angle, args.angle)

    # rotation For jpg
    image_jpg = Image.open(file_jpg)
    image_jpg_orig = image_jpg.copy()  # Save the original for further transformations
    image_jpg = ToTensor()(image_jpg)
    image_jpg = Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)(image_jpg)  # Ensure 3-channel for image
    image_jpg = TF.rotate(image_jpg, rotation_angle)
    image_jpg = TF.to_pil_image(image_jpg)
    fstring_jpg_rot = filename_jpg + "-rot{}.jpg".format(args.angle)
    image_jpg.save(os.path.join(inpf_path_jpg, fstring_jpg_rot))

    # For png
    image_png = Image.open(file_png)
    image_png_orig = image_png.copy()  # Save the original for further transformations
    image_png = image_png.rotate(rotation_angle)  # Rotate the image
    fstring_png_rot = filename_png + "-rot{}.png".format(args.angle)
    image_png.save(os.path.join(inpf_path_png, fstring_png_rot))

    # Apply horizontal flip and save for jpg and png
    image_jpg = flipper(image_jpg_orig)
    image_jpg.save(os.path.join(inpf_path_jpg, filename_jpg + "-flip.jpg"))
    image_png = flipper(image_png_orig)
    image_png.save(os.path.join(inpf_path_png, filename_png + "-flip.png"))

    # Apply color jitter and save for jpg and png
    image_jpg = jitter(image_jpg_orig)
    image_jpg.save(os.path.join(inpf_path_jpg, filename_jpg + "-jitter.jpg"))
    image_png = image_png_orig.copy()  # We only save the copy with the same name but no color jitter is applied
    image_png.save(os.path.join(inpf_path_png, filename_png + "-jitter.png"))

    # Apply shear and save
    image_jpg = shearer(image_jpg_orig)
    image_jpg.save(os.path.join(inpf_path_jpg, filename_jpg + "-shear.jpg"))
    image_png = shearer(image_png_orig)
    image_png.save(os.path.join(inpf_path_png, filename_png + "-shear.png"))

num_processes = mp.cpu_count()

with mp.Pool(num_processes) as p:
    list(tqdm(p.imap(process_files, zip(list_wkk_jpg, list_wkk_png)), total=len(list_wkk_jpg)))