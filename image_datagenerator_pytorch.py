import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, RandomRotation
import torchvision.transforms.functional as TF

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
inpf_path_jpg=r"C:\Users\d18036\Desktop\ImageDataGenerator_hayashi\JPEGImages+shear"
#mask images
inpf_path_png=r"C:\Users\d18036\Desktop\ImageDataGenerator_hayashi\SegmentationClass+shear"

list_wkk_jpg=os.listdir(inpf_path_jpg)
list_wkk_png=os.listdir(inpf_path_png)

for file_jpg, file_png in zip(list_wkk_jpg, list_wkk_png):
    filename=os.path.splitext(file_jpg)[0]
    fstring_jpg = filename + "-rot180.jpg"
    fstring_png = filename + "-rot180.png"

    # Process jpg file
    image_jpg = Image.open(os.path.join(inpf_path_jpg, file_jpg))
    image_jpg = image_transforms(image_jpg)
    image_jpg = TF.to_pil_image(image_jpg)
    image_jpg.save(os.path.join(inpf_path_jpg, fstring_jpg))

    # Process png file
    image_png = Image.open(os.path.join(inpf_path_png, file_png))
    palette = image_png.getpalette()  # Save original palette for png
    image_png = mask_transforms(image_png)
    image_png = TF.to_pil_image(image_png)
    image_png.putpalette(palette)
    image_png.save(os.path.join(inpf_path_png, fstring_png))