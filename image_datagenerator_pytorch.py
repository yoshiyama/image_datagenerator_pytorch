import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, RandomRotation
import torchvision.transforms.functional as TF
from tqdm import tqdm

'''
    PyTorchはNumPyと同じ形式（[高さ, 幅, チャンネル]）で画像を扱いますが、Tensorは[チャンネル, 高さ, 幅]の形式でデータを管理します。そのため、前処理時にこの変換を行う必要があります。
    torchvision.transformsモジュールを用いて画像の拡張処理を行います。
    このコードは画像がRGBであることを前提としています。もし1チャネルの画像を扱う場合は、適宜修正してください。
    PIL.Imageのmodeは"L" (8-bit pixels, black and white) を前提としています。これは異なる場合、適宜修正してください。
    画像の読み込みと保存はPILライブラリを使用しています。
    このコードでは、torchvision.transformsモジュールを利用してランダムな回転の拡張を適用しています。また、出力された画像は同じディレクトリ内の新しいファイルとして保存されます。
'''

# Transform for image (jpg)
image_transforms = Compose([
    ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Ensure 3-channel for image
    RandomRotation(180)
])

# Transform for mask (png)
mask_transforms = Compose([
    ToTensor(),
    RandomRotation(180)
])
#jpeg images
inpf_path_jpg=r"/mnt/c/Users/survey/Documents/GitHub/ImageDataGenerator/JPEGImages+rot"
#mask images
inpf_path_png=r"/mnt/c/Users/survey/Documents/GitHub/ImageDataGenerator/SegmentationClass+rot"

list_wkk_jpg=os.listdir(inpf_path_jpg)
list_wkk_png=os.listdir(inpf_path_png)

# for file_jpg, file_png in zip(list_wkk_jpg, list_wkk_png):
for file_jpg, file_png in tqdm(zip(list_wkk_jpg, list_wkk_png), total=len(list_wkk_jpg)):
    filename=os.path.splitext(file_jpg)[0]
    fstring_jpg = filename + "-rot180.jpg"
    fstring_png = filename + "-rot180.png"

    # Generate a random rotation angle
    rotation_angle = np.random.uniform(-180, 180)

    # Process jpg file
    image_jpg = Image.open(os.path.join(inpf_path_jpg, file_jpg))
    image_jpg = ToTensor()(image_jpg)
    image_jpg = Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)(image_jpg)  # Ensure 3-channel for image
    image_jpg = TF.rotate(image_jpg, rotation_angle)  # Rotate the image
    image_jpg = TF.to_pil_image(image_jpg)
    image_jpg.save(os.path.join(inpf_path_jpg, fstring_jpg))

    # Process png file
    image_png = Image.open(os.path.join(inpf_path_png, file_png))
    palette = image_png.getpalette()  # Save original palette for png
    image_png = ToTensor()(image_png)
    image_png = TF.rotate(image_png, rotation_angle)  # Rotate the mask with the same angle
    image_png = TF.to_pil_image(image_png)
    image_png.putpalette(palette)
    image_png.save(os.path.join(inpf_path_png, fstring_png))