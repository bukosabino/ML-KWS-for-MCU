"""Set of functions useful for data augmentation techniques in speech
recognition problems.
"""

import numpy as np
import scipy.io.wavfile
import cv2
import random
import matplotlib.pyplot as plt


def add_noise(data, noise, noise_dir):
    """Add noise to sound.

    Keyword arguments:

    noise -- Choose betwen different noises:
        {white_noise, pink_noise, exercise_bike, doing_the_dishes, running_tap}

    noise_dir -- Where to load noise .wav files.
    """
    if noise == 'white_noise':
        filename = 'white_noise.wav'
        p = 0.01
    elif noise == 'pink_noise':
        filename = 'pink_noise.wav'
        p = 0.015
    elif noise == 'exercise_bike':
        filename = 'exercise_bike.wav'
        p = 0.1
    elif noise == 'doing_the_dishes':
        filename = 'doing_the_dishes.wav'
        p = 0.09
    elif noise == 'running_tap':
        filename = 'running_tap.wav'
        p = 0.07

    noise_data = load_audio_file(noise_dir + filename)
    data_with_noise = (data * (1-p)).astype('int16') + (noise_data * p).astype('int16')
    return data_with_noise

def scale_amplitude_transform(wave, scale_limit=0.1, u=0.5):
    """Scale amplitude of wav file.
    """
    if random.random() < u:
        scale = random.uniform(-scale_limit, scale_limit)
        wave = scale*wave
    return wave.astype('int16')

def shift_sound(data, roll=1600):
    """Roll sound in the time.
    Elements that roll beyond the last position are re-introduced at the first.
    """
    input_length = 16000
    data = np.roll(data, roll)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def shift_sound_padding(data, roll=1600):
    """Roll sound in the time.
    It use 0s to pad the gaps.
    """
    input_length = 16000
    data = _roll_zeropad(data, roll)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def stretch_sound(wav, speed_rate):
    """Time-stretch an wav file by a fixed rate.

    Keyword arguments:

    speed_rate -- float > 0:
        If speed_rate > 1, the the signal is speed up.
        If rate < 1, then the signal is slowed down.
    """
    sr = 16000
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    if len(wav_speed_tune) < 16000:
        pad_len = 16000 - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))].astype('int16')
    else:
        cut_len = len(wav_speed_tune) - 16000
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
    return wav_speed_tune


def load_audio_file(file_path):
    input_length = 16000
    data = scipy.io.wavfile.read(file_path)[1]
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def save_audio_file(path, data):
    sr = 16000
    scipy.io.wavfile.write(path, sr, data)


def write_file(data, filename, operation):
    new_filename = _update_path(filename, operation)
    save_audio_file(new_filename, data)
    return new_filename


def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


def _update_path(path_file, operation):
    path = path_file.split("/")
    path_waw = path[-1].split(".")
    path[-1] =  path_waw[0] + str(operation) + "." + path_waw[1]
    return "/".join(path)


def _roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.
    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
