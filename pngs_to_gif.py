# given a directory of pngs, convert them to a gif

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helper_functions import pngs_to_gif


if __name__ == '__main__':
    # get the home path
    with open('home_path.txt', 'r') as f:
        home_path = f.readlines()[0].strip()
    # get the directory of the pngs
    png_dir = home_path + 'results/old_results/'
    # get the name of the gif
    gif_name = 'gif1'
    # convert the pngs to a gif
    pngs_to_gif(png_dir, gif_name)