from PIL import Image
from dinoproof.utils.image_utils import split_image
import os

# Specify folder containing raw images
foldername = "./screenshots/RAW_DATASETS/test_set_1"
resize_dim = 512

# Iterate over images in foldername
for image in os.listdir(foldername):
    if not os.path.isfile(os.path.join(foldername, image)):
        continue
    image_obj = Image.open(os.path.join(foldername, image))

    left_image, right_image = split_image(image_obj)
    left_image = left_image.resize((resize_dim, resize_dim), Image.BICUBIC)
    right_image = right_image.resize((resize_dim, resize_dim), Image.BICUBIC)

    if not os.path.exists(os.path.join(foldername, f"left_{resize_dim}")):
        os.makedirs(os.path.join(foldername, f"left_{resize_dim}"))
    if not os.path.exists(os.path.join(foldername, f"right_{resize_dim}")):
        os.makedirs(os.path.join(foldername, f"right_{resize_dim}"))

    left_image.save(os.path.join(foldername,f"left_{resize_dim}/left-{image}"))
    right_image.save(os.path.join(foldername,f"right_{resize_dim}/right-{image}"))
