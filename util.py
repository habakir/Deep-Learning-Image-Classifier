from PIL import Image
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images


def getImagesAndLables():
    folder_paths = ["do", "re", "mi", "fa", "sol", "la", "ti"]
    images = []
    labels = []
    for folder in folder_paths:
        newList = load_images_from_folder(folder)
        images += newList
        labels += [folder for _ in newList]
    return images, labels