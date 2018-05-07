import numpy as np
import glob
from utils_data_augmentation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_dir',
    type=str,
    default='../../data/train/audio/*/*.wav',
    help='Where to prepare (load and write) the train .wav files.')
parser.add_argument(
    '--noise_dir',
    type=str,
    default='../../data/train/audio/_background_noise_/',
    help='Where to load noise .wav files.')
flags, args = parser.parse_known_args()

train_dir = flags.train_dir
noise_dir = '../../data/train/audio/_background_noise_/'
noises = ['white_noise', 'pink_noise', 'exercise_bike', 'doing_the_dishes', 'running_tap']
rolling_list_pad = [-1600, -800, 800, 1600]
rolling_list = [-800, 800]
stretch_list = [0.8, 1.2]
rolling_padding = True
rolling_no_padding = False
stretch = True
amplitude = True


for idx, filename in enumerate(glob.iglob(train_dir)):

    folder = filename.split('/')[-2]
    if folder != '_background_noise_':
        data = load_audio_file(filename)

        if rolling_padding: # roll with padding 0
            for roll in rolling_list_pad:
                roll_data_pad = shift_sound_padding(data, roll)
                operation = '_roll_pad_' + str(roll)
                new_filename = write_file(roll_data_pad, filename, operation)
                n = noises[np.random.randint(0, len(noises))]
                n_data = add_noise(roll_data_pad, n, noise_dir)
                write_file(n_data, new_filename, '_n')

        if rolling_no_padding: # roll without padding
            for roll in rolling_list:
                roll_data = shift_sound(data, roll)
                operation = '_roll_' + str(roll)
                new_filename = write_file(roll_data, filename, operation)
                n = noises[np.random.randint(0, len(noises))]
                n_data = add_noise(roll_data, n, noise_dir) # noise
                write_file(n_data, new_filename, '_n')

        if stretch: # stretch
            for rate in stretch_list:
                stretch_data = stretch_sound(data, rate)
                operation = '_stretch_' + str(rate).replace('.', '_')
                new_filename = write_file(stretch_data, filename, operation)
                n = noises[np.random.randint(0, len(noises))]
                n_data = add_noise(stretch_data, n, noise_dir) # noise
                write_file(n_data, new_filename, '_n')

        # noise
        n = noises[np.random.randint(0, len(noises))]
        n_data = add_noise(data, n, noise_dir)
        write_file(n_data, filename, '_n')

        # amplitude
        if amplitude:
            data_amplitude = scale_amplitude_transform(data, 0.1, 1)
            write_file(data_amplitude, filename, '_amp_down')
            data_amplitude = scale_amplitude_transform(data, 10, 1)
            write_file(data_amplitude, filename, '_amp_up')
