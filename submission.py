r"""Runs a trained audio graph against a dir WAVE files and reports the results.

The model, labels and .wav files specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console.

Here's an example of running it:

python submission.py --graph=tmp/my_frozen_graph.pb \
--labels=tmp/speech_commands_train/conv_labels.txt \
--wav=tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    output = ''
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        # print('%s (score = %.5f)' % (human_string, score))
        output += str(human_string) + ", " + str(score) + ", "
        # output = human_string

    return output


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
    """Loads the model and labels, and runs the inference to print predictions."""
    if not wav or not tf.gfile.Exists(wav):
        tf.logging.fatal('Audio file does not exist %s', wav)

    if not labels or not tf.gfile.Exists(labels):
        tf.logging.fatal('Labels file does not exist %s', labels)

    if not graph or not tf.gfile.Exists(graph):
        tf.logging.fatal('Graph file does not exist %s', graph)

    labels_list = load_labels(labels)

    with open(wav, 'rb') as wav_file:
        wav_data = wav_file.read()

    return run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    '--labels',
    type=str,
    default='Pretrained_models/labels.txt',
    help='What labels (.pb) to use')
parser.add_argument(
    '--graph',
    type=str,
    default='tmp/my_frozen_graph.pb',
    help='What graph (.pb) to use')
parser.add_argument(
    '--test_dir',
    type=str,
    default='../../data/test/audio/*.wav',
    help='Where to load the test .wav files.')
parser.add_argument(
    '--output',
    type=str,
    default='submission.csv',
    help='Where to write the results.')

flags, args = parser.parse_known_args()

graph = flags.graph
labels = flags.labels
test_dir = flags.test_dir
output = flags.output
input_name = 'wav_data:0'
output_name = 'labels_softmax:0'
how_many_labels = 3

# load graph, which is stored in the default session
load_graph(graph)

submission = {}
for idx, filename in enumerate(glob.iglob(test_dir)):
    label = label_wav(filename, labels, graph, input_name, output_name, how_many_labels)
    submission[filename.split('/')[-1]] = label

# write submission
with open(output, 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))
