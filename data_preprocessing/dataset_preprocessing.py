import os
import argparse
from tqdm import tqdm
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, default='preprocessed')
    parser.add_argument('--resize_shape', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--crop_type', default='center', choices=['center'])
    args = parser.parse_args()
    return args

def process_sample(sample_path, args):
    img = Image.open(sample_path)
    img = img.resize((args.resize_shape, args.resize_shape))

    left = (args.resize_shape - args.crop_size) / 2
    top = (args.resize_shape - args.crop_size) / 2
    right = (args.resize_shape + args.crop_size) / 2
    bottom = (args.resize_shape + args.crop_size) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

def process_dataset(args):
    # create preprocessed folder 
    output_folder = f'{args.output_folder}_{args.data_folder}'
    os.makedirs(output_folder, exist_ok=True)

    for data_class in os.listdir(args.data_folder)[3:4]: 
        class_folder = os.path.join(args.data_folder, data_class)
        os.makedirs(os.path.join(output_folder, data_class), exist_ok=True)
        for sample in tqdm(os.listdir(class_folder)):
            sample_path = os.path.join(args.data_folder, data_class, sample)
            img = process_sample(sample_path, args)
            output_path = os.path.join(output_folder, data_class, f'preprocessed {sample}')
            img.save(output_path)

if __name__ == '__main__':
    args = parse_arguments()

    process_dataset(args)
