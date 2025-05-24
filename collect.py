from ngllib import Environment
import time

options = {
        'euler_angles': True,
        'resize': False,
        'add_mouse': False,
        'fast': True,
        'image_path': None
}

env = Environment(headless=False, config_path="config.json", verbose=False)

env.start_session(**options)

while True:
    cmd = input(">>> ").strip()

    if cmd.lower() == "exit":
        env.end_session()
        break
    elif cmd.lower() == "ss":

        ss_name = time.strftime("%Y-%m-%d-%H-%M-%S")
        ss_name = "./screenshots/" + ss_name + ".png"
        env.get_screenshot(save_path=ss_name,resize=False, mouse_x=None, mouse_y=None, fast=False)