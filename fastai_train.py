from fastai.vision.all import *
from PIL import Image
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_image_files_from_folder(folder):
    image_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_files.append(os.path.join(folder, filename))
    return image_files

def get_images_and_labels():
    folder_paths = ["do", "re", "mi", "fa", "sol", "la", "ti"]
    image_files = []
    labels = []
    for folder in folder_paths:
        image_files.extend(get_image_files_from_folder(folder))
        labels.extend([folder] * len(get_image_files_from_folder(folder)))
    return image_files, labels

image_files, labels = get_images_and_labels()
data = list(zip(image_files, labels))
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data, valid_data = data[:split_idx], data[split_idx:]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=lambda _: image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
)

dls = dblock.dataloaders('', bs=32, num_workers=0) 

learn = vision_learner(dls, resnet18, metrics=error_rate)

learn.fine_tune(3)

learn.show_results()

for img_path in image_files:
    pred, _, _ = learn.predict(img_path)
    print(f"Prediction for {img_path}: {pred}")
