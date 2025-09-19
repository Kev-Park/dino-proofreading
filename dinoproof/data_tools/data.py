from ngllib import Environment
import time
import os
import sys
from PIL import Image
from dinoproof.utils.image_utils import split_image
from dinoproof.utils.annotate import *
from dinoproof.utils.augment import augment_folder

class NGLData():
    def __init__(self):
        self.output_path = None

    def process(self, resize_dim, output_path):
        """
        Split Neuroglancer views and resize them to the desired dimension.
        """

        for image in os.listdir(output_path):
            if not os.path.isfile(os.path.join(output_path, image)):
                continue
            image_obj = Image.open(os.path.join(output_path, image))

            left_image, right_image = split_image(image_obj)
            left_image = left_image.resize((resize_dim, resize_dim), Image.BICUBIC)
            right_image = right_image.resize((resize_dim, resize_dim), Image.BICUBIC)

            if not os.path.exists(os.path.join(output_path, f"left_{resize_dim}")):
                os.makedirs(os.path.join(output_path, f"left_{resize_dim}"))
            if not os.path.exists(os.path.join(output_path, f"right_{resize_dim}")):
                os.makedirs(os.path.join(output_path, f"right_{resize_dim}"))

            left_image.save(os.path.join(output_path,f"left_{resize_dim}/left-{image}"))
            right_image.save(os.path.join(output_path,f"right_{resize_dim}/right-{image}"))

    def collect(self):
        """
        Start a Neuroglancer session and collect screenshots.
        """

        options = {
            'euler_angles': True,
            'resize': False,
            'add_mouse': False,
            'fast': True,
            'image_path': None
        }

        # Get Neuroglancer config path
        config_path = input("Enter the NGL config JSON path (enter to look in current directory for config.json or 'exit' to quit): ")
        if config_path.lower() == 'exit':
            sys.exit(0)
        elif config_path == '':
            config_path = 'config.json'

        # Get output path
        output_path = input("Enter the output directory for screenshots (enter to use ./screenshots/ or 'exit' to quit): ")
        if output_path.lower() == 'exit':
            sys.exit(0)
        elif output_path == '':
            output_path = './screenshots/'
        else:
            if not output_path.endswith('/'):
                output_path += '/'
        self.output_path = output_path

        # Start session
        print("Starting Neuroglancer session. Type 'exit' to quit or 'ss' to take a screenshot.")
        env = Environment(headless=False, config_path=config_path, verbose=False)
        env.start_neuroglancer_session()

        # Collect images in a loop
        while True:
            cmd = input(">>> ").strip()

            if cmd.lower() == "exit":
                env.end_session()
                break
            elif cmd.lower() == "ss":

                ss_name = time.strftime("%Y-%m-%d-%H-%M-%S")
                ss_name = output_path + ss_name + ".png"
                env.get_screenshot(save_path=ss_name,resize=False, mouse_x=None, mouse_y=None, fast=False)

        resize_dim = input("Enter the resize dimension (enter to skip splitting and resizing, or 'exit' to quit): ")

        if resize_dim.lower() == 'exit':
            sys.exit(0)
        elif resize_dim != '':
            self.process(resize_dim=int(resize_dim), output_path=self.output_path)

            annotate = input("Do you want to annotate (yes to continue, enter to skip, exit/no to quit): ")
            if annotate.lower() == 'exit' or annotate.lower() == 'no':
                sys.exit(0)
            elif annotate.lower() == 'yes':
                annotate_terminate(input_path = self.output_path + f"right_{resize_dim}/")

        augment = input("Do you want to augment (yes to continue, enter to skip, exit/no to quit): ")
        if augment.lower() == 'exit' or augment.lower() == 'no':
            sys.exit(0)
        elif augment.lower() == 'yes':
            augment_folder(input_folder=self.output_path + f"right_{resize_dim}/", output_folder=self.output_path + f"right_{resize_dim}/")
            #augment_folder(input_folder=self.output_path + f"left_{resize_dim}/", output_folder=self.output_path + f"left_{resize_dim}_augmented/")

if __name__ == "__main__":
    data_collector = NGLData()
    data_collector.collect()
    #data_collector.annotate_terminate("./screenshots/RAW_DATASETS/test_set_1_augmented_512")