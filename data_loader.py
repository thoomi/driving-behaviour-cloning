"""Udacity P3 - Behavioral Cloning - Loads data from provided csv files"""
import csv


def load_data_samples(folders=[]):
    """Load data from provided data folders"""
    recorded_base_path = './data/recorded/'
    recorded_log_file = 'driving_log.csv'
    recorded_image_path = '/IMG/'

    samples = []
    for folder in folders:
        log_file = recorded_base_path + folder + '/' + recorded_log_file

        with open(log_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                line[0] = recorded_base_path + folder + recorded_image_path + line[0].split('/')[-1]
                line[1] = recorded_base_path + folder + recorded_image_path + line[1].split('/')[-1]
                line[2] = recorded_base_path + folder + recorded_image_path + line[2].split('/')[-1]

                samples.append(line)

    return samples
