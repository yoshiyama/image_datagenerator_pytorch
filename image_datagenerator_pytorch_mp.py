
'''
    PyTorchはNumPyと同じ形式（[高さ, 幅, チャンネル]）で画像を扱いますが、Tensorは[チャンネル, 高さ, 幅]の形式でデータを管理します。そのため、前処理時にこの変換を行う必要があります。
    torchvision.transformsモジュールを用いて画像の拡張処理を行います。
    このコードは画像がRGBであることを前提としています。もし1チャネルの画像を扱う場合は、適宜修正してください。
    PIL.Imageのmodeは"L" (8-bit pixels, black and white) を前提としています。これは異なる場合、適宜修正してください。
    画像の読み込みと保存はPILライブラリを使用しています。
    このコードでは、torchvision.transformsモジュールを利用してランダムな回転の拡張を適用しています。また、出力された画像は同じディレクトリ内の新しいファイルとして保存されます。
    python image_datagenerator_pytorch_mp.py --jpg_path "/path/to/jpeg" --png_path "/path/to/png" --angle 180
'''
import os
from PIL import Image
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import argparse
from glob import glob

# パーサーを作成
parser = argparse.ArgumentParser(description='Process some images.')

# #jpeg images
# inpf_path_jpg=r"/mnt/c/Users/survey/Documents/GitHub/ImageDataGenerator/JPEGImages+rot"
# #mask images
# inpf_path_png=r"/mnt/c/Users/survey/Documents/GitHub/ImageDataGenerator/SegmentationClass+rot"

# コマンドライン引数を定義
parser.add_argument('--jpg_path', type=str, required=True, help='Path to JPEG images directory.')
parser.add_argument('--png_path', type=str, required=True, help='Path to PNG images directory.')
parser.add_argument('--angle', type=int, required=True, help='Angle to rotate images.')

args = parser.parse_args()

#jpeg images
inpf_path_jpg=args.jpg_path
#mask images
inpf_path_png=args.png_path

list_wkk_jpg=sorted(glob(os.path.join(inpf_path_jpg, '*.jpg')))
list_wkk_png=sorted(glob(os.path.join(inpf_path_png, '*.png')))

# list_wkk_jpg=os.listdir(inpf_path_jpg)
# list_wkk_png=os.listdir(inpf_path_png)

# def process_files(file_jpg, file_png):
def process_files(files):
    file_jpg, file_png = files
    # filename=os.path.splitext(file_jpg)[0]
    filename_jpg = os.path.splitext(os.path.basename(file_jpg))[0]  # Update here
    filename_png = os.path.splitext(os.path.basename(file_png))[0]  # Update
    # fstring_jpg = filename + "-rot180.jpg"
    # fstring_png = filename + "-rot180.png"
    fstring_jpg = filename_jpg + "-rot{}.jpg".format(args.angle)
    fstring_png = filename_png + "-rot{}.png".format(args.angle)

    # Generate a random rotation angle
    # rotation_angle = np.random.uniform(-180, 180)
    rotation_angle = np.random.uniform(-args.angle, args.angle)

    # Process jpg file
    # image_jpg = Image.open(os.path.join(inpf_path_jpg, file_jpg))
    image_jpg = Image.open(file_jpg)  # Update here
    image_jpg = ToTensor()(image_jpg)
    image_jpg = Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)(image_jpg)  # Ensure 3-channel for image
    image_jpg = TF.rotate(image_jpg, rotation_angle)  # Rotate the image
    image_jpg = TF.to_pil_image(image_jpg)
    image_jpg.save(os.path.join(inpf_path_jpg, fstring_jpg))

    # Process png file
    # image_png = Image.open(os.path.join(inpf_path_png, file_png))
    image_png = Image.open(file_png)  # Update here
    palette = image_png.getpalette()  # Save original palette for png
    image_png = ToTensor()(image_png)
    image_png = TF.rotate(image_png, rotation_angle)  # Rotate the mask with the same angle
    image_png = TF.to_pil_image(image_png)
    image_png.putpalette(palette)
    image_png.save(os.path.join(inpf_path_png, fstring_png))  # Update here
    # image_png.save(os.path.join(inpf_path_png, fstring_png))

# Define the number of processes to spawn
num_processes = mp.cpu_count()

# Create a Pool of subprocesses
with mp.Pool(num_processes) as p:
    list(tqdm(p.imap(process_files, zip(list_wkk_jpg, list_wkk_png)), total=len(list_wkk_jpg)))
# with mp.Pool(num_processes) as p:
#     list(tqdm(p.imap(process_files, list_wkk_jpg, list_wkk_png), total=len(list_wkk_jpg)))