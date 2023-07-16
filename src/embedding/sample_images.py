import os
import random
import shutil
from tqdm import tqdm

class SampleImages:
    """
    Sample images
    """
    def __init__(self, args):
        self.input_path = args.input_image_path
        self.output_path = args.output_image_path
        self.r = args.image_sampling_ratio

    """
    The wrapper function called by main.py
    """
    def sample_images(self):
        # Generate the directory specified by `self.output_path`
        # if it does not already exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Iterate over the sub-directories in the input directory
        input_sub_directories = os.listdir(self.input_path)
        with tqdm(total=len(input_sub_directories)) as pbar:
            for sub_dir in input_sub_directories:
                sub_dir_path = os.path.join(self.input_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    # Create the corresponding sub-directory in the output directory
                    output_sub_dir_path = os.path.join(self.output_path, sub_dir)
                    if not os.path.exists(output_sub_dir_path):
                        os.makedirs(output_sub_dir_path)

                    # Get the list of image files in the subdirectory
                    image_files = os.listdir(sub_dir_path)

                    # Calculate the number of images to sample
                    num_samples = int(len(image_files) * self.r)

                    # Randomly select the images to copy
                    sampled_images = random.sample(image_files, num_samples)

                    # Copy the sampled images to the output subdirectory
                    for image_file in sampled_images:
                        source_path = os.path.join(sub_dir_path, image_file)
                        destination_path = os.path.join(output_sub_dir_path, image_file)
                        shutil.copyfile(source_path, destination_path)

                pbar.update(1)