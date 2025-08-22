from PIL import Image
from dinoproof.utils.image_utils import pad_image, split_image
import os

# Specify folder containing raw images
foldername = "./screenshots/false_positive"

# Iterate over images in foldername
for image in os.listdir(foldername):
    
    image_obj = Image.open(os.path.join(foldername, image))

    left_image, right_image = split_image(image_obj)
    left_image = left_image.resize((392, 392), Image.BICUBIC)
    right_image = right_image.resize((392, 392), Image.BICUBIC)

    if not os.path.exists(os.path.join(foldername, "left")):
        os.makedirs(os.path.join(foldername, "left"))
    if not os.path.exists(os.path.join(foldername, "right")):
        os.makedirs(os.path.join(foldername, "right"))

    left_image.save(os.path.join(foldername,f"left/left-{image}"))
    right_image.save(os.path.join(foldername,f"right/right-{image}"))



# image = pad_image(image)

# # Resize with bicubic interpolation
# image = image.resize((392, 392), Image.BICUBIC)

# image.save(f"./screenshots/padded-{image_name}")


