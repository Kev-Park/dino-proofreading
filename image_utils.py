from PIL import Image

# Pad images with black on the top and bottom to create a square image, then resize to nxn
def pad_image(image):
    width, height = image.size
    if width > height:
        new_height = width
        new_image = Image.new("RGB", (width, new_height), (0, 0, 0))
        new_image.paste(image, (0, (new_height - height) // 2))
    else:
        new_width = height
        new_image = Image.new("RGB", (new_width, height), (0, 0, 0))
        new_image.paste(image, ((new_width - width) // 2, 0))

    return new_image

def split_image(image):
    width, height = image.size
    width = width // 2

    image1 = image.crop((0, 0, width, height))
    image2 = image.crop((width, 0, width * 2, height))

    return image1, image2