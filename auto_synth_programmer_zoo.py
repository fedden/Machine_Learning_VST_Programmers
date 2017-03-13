import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'mlp'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mlp_recursive'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cnn_to_rnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'rnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'rnn_recursive'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ga'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulated_annealing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from mlp import MLP
from mlp_recursive import RecursiveMLP
from rnn import LSTM
from ga import GeneticAlgorithm
from plugin_feature_extractor import PluginFeatureExtractor
import tensorflow as tf
import numpy as np
from utility_functions import *
from tqdm import trange

# Load VST.
extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   #desired_features=[i for i in range(8, 21)],
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore", normalise_audio=True)
path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():
    extractor.fit_normalisers(2000000)

# Get training and testing batch.
train_batch_x, train_batch_y, test_batch_x, test_batch_y = get_batches(1, 1, extractor)

# Load models.
features_cols = train_batch_x[0].shape[0]
features_rows = train_batch_x[0].shape[1]
parameter_size = train_batch_y[0].shape[0]
features = tf.placeholder(tf.float32, [None, features_cols, features_rows])
patches = tf.placeholder(tf.float32, [None, parameter_size])
prob_keep_input = tf.placeholder(tf.float32)
prob_keep_hidden = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32)

model = GeneticAlgorithm(extractor=extractor, population_size=400,
                         percent_elitism_elites=5, percent_elitist_parents=5,
                         dna_length=(parameter_size), target_features=test_batch_x,
                         feature_size=(features_cols * features_rows))

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

errors = []
for _ in trange(5, desc="Optimising model(s)"):
    model.optimise()
    prediction = model.prediction()
    stats = get_stats(extractor, prediction, test_batch_x, test_batch_y)
    errors += [stats[0]]

write_wavs(stats[1]['predicted_patch'], stats[1]['actual_patch'], extractor)
display_stats(stats)
plot_error(errors)


    # error = sess.run(model.error, { features: test_batch_x,
    #                                 patches: test_batch_y,
    #                                 batch_size: test_batch_size })
    # print "error: " + str(error)
    #
    # for _ in range(1):
    #     sess.run(model.optimise, { features: train_batch_x,
    #                                patches: train_batch_y,
    #                                batch_size: train_batch_size })
