import sys
import argparse

sys.path.append("..")
from sampling.Sampler import Sampler

increment = 1
max_percentage = 20

parser = argparse.ArgumentParser()
parser.parse_args()

my_sampler = Sampler()

for i in range(1, 20, increment):
    # get known training dataset and private dataset.
    known_training_dataset, private_dataset = my_sampler.sample(i)
    # perform inference on the model.
    # calculate loss
    # test if losses follow gaussian distribution.
    # calculate mean and std
    # classify images.
    # evaluate.
