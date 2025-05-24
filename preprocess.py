from PIL import Image

from image_utils import pad_image, split_image

image_name = '2025-05-21-18-55-05.png'
image = Image.open(f"./screenshots/{image_name}")


left_image, right_image = split_image(image)
left_image = left_image.resize((480, 480), Image.BICUBIC)
right_image = right_image.resize((480, 480), Image.BICUBIC)
left_image.save(f"./screenshots/left-{image_name}")
right_image.save(f"./screenshots/right-{image_name}")



image = pad_image(image)

# Resize with bicubic interpolation
image = image.resize((480, 480), Image.BICUBIC)

image.save(f"./screenshots/padded-{image_name}")


