import os
import pickle
import warnings
import numpy as np
import scipy.io.wavfile
from tqdm import trange
import librenderman as rm
from sklearn import preprocessing
from sklearn.externals import joblib
from pyAudioAnalysis import audioFeatureExtraction as fe

class PluginFeatureExtractor:

    def __init__(self, **kwargs):
        self.sample_rate = kwargs.get('sample_rate', 44100)
        self.buffer_size = kwargs.get('buffer_size', 512)
        self.midi_note = kwargs.get('midi_note', 40)
        self.midi_velocity = kwargs.get('midi_velocity', 127)
        self.note_length_secs = kwargs.get('note_length_secs', 1.0)
        self.render_length_secs = kwargs.get('render_length_secs', 2.5)
        self.frame_step_ms = kwargs.get('frame_step_ms', 25)
        self.frame_size_ms = kwargs.get('frame_size_ms', 50)
        self.pickle_path = kwargs.get('pickle_path', "")
        self.warning_mode = kwargs.get('warning_mode', "always")
        self.normalise_audio = kwargs.get('normalise_audio', False)
        self.desired_features_indices = kwargs.get('desired_features', [i for i in range(21)])
        self.frame_size_samples = int(self.frame_size_ms * self.sample_rate / 1000.0)
        self.frame_step_samples = int(self.frame_step_ms * self.sample_rate / 1000.0)
        self.loaded_plugin = False
        self.rendered_patch = False

    def load_plugin(self, plugin_path):
        self.engine = rm.RenderEngine(self.sample_rate,
                                      self.buffer_size,
                                      2048)
        if plugin_path == "":
            print "Please supply a non-empty path"
            return

        if self.engine.load_plugin(plugin_path):
            self.loaded_plugin = True
            self.generator = rm.PatchGenerator(self.engine)
            print "Successfully loaded plugin."
        else:
            print "Unsuccessful loading of plugin: is the path correct?"

    def set_patch(self, plugin_patch):
        if self.loaded_plugin:
            self.engine.set_patch(plugin_patch)
            self.engine.render_patch(self.midi_note,
                                     self.midi_velocity,
                                     self.note_length_secs,
                                     self.render_length_secs)
            self.rendered_patch = True
            return True
        else:
            print "Please load plugin first"

    def get_features_from_patch(self, plugin_patch):
        if self.pickle_files_exist():
            files = self.get_file_paths()
            if self.set_patch(plugin_patch):
                int_audio_frames = self.float_to_int_audio(np.array(self.get_audio_frames()))
                feature_vector = self.get_desired_features(int_audio_frames)
                contains_nan = np.isnan(feature_vector).any()
                if contains_nan:
                    feature_vector = np.zeros_like(feature_vector)
                normalisers = [joblib.load(files[i]) for i in range(len(files))]
                normalised_features = []
                index = 0
                for i in range(len(normalisers)):
                    if i in self.desired_features_indices:
                        normalised_features.append(normalisers[i].transform(feature_vector[index]))
                        index += 1
                norm_features = np.array(normalised_features)
                return norm_features.T
            else:
                return None
        else:
            print "Please train normalisers using PluginFeatureExtractor.fit_normalisers()."

    def add_patch_indices(self, patch):
        tuple_patch = []
        for i in range(len(patch)):
            tuple_patch += [(i, float(patch[i]))]
        return tuple_patch

    def remove_patch_indices(self, patch):
        return np.array([parameter[1] for parameter in patch])

    def list_patch(self):
        print self.engine.get_plugin_parameters_description()

    def get_audio_frames(self):
        if self.rendered_patch:
            audio = np.array(self.engine.get_audio_frames())
            if self.normalise_audio:
                return audio / np.max(np.abs(audio), axis=0)
            return audio
        else:
            print "Please set and render a patch before trying to get audio frames."

    def write_to_wav(self, path):
        if self.rendered_patch:
            float_audio = np.array(self.get_audio_frames())
            int_audio = self.float_to_int_audio(float_audio)
            scipy.io.wavfile.write(path, 44100, int_audio)
        else:
            print "Render a patch first before writing to file!"

    def float_to_int_audio(self, float_audio_frames):
        float_audio_frames *= 32768
        return np.clip(float_audio_frames, -32768, 32767).astype(np.int16)

    def get_random_example(self):
        if self.loaded_plugin:
            random_patch_list_tuples = self.generator.get_random_patch()
            random_patch = np.array([p[1] for p in random_patch_list_tuples])
            self.set_patch(random_patch_list_tuples)
            int_audio_frames = self.float_to_int_audio(np.array(self.get_audio_frames()))
            feature_vector = self.get_desired_features(int_audio_frames)
            contains_nan = np.isnan(feature_vector).any()
            if contains_nan:
                return (np.zeros_like(feature_vector.T), random_patch)

            return (feature_vector.T, random_patch)
        else:
            print "Please load plugin first."


    def get_desired_features(self, int_audio_frames):
        feature_vector = fe.stFeatureExtraction(int_audio_frames,
                                                self.sample_rate,
                                                self.frame_size_samples,
                                                self.frame_step_samples)
        output = []
        for desired_index in self.desired_features_indices:
            output.append(feature_vector[desired_index])
        return np.array(output)

    def get_file_paths(self):
        if not self.pickle_path.endswith('/') and self.pickle_path != "":
            self.pickle_path += "/"
        files = [self.pickle_path + "zero_crossing_rate.pkl",
                 self.pickle_path + "energy.pkl",
                 self.pickle_path + "energy_entropy.pkl",
                 self.pickle_path + "spectral_centroid.pkl",
                 self.pickle_path + "spectral_spread.pkl",
                 self.pickle_path + "spectral_entropy.pkl",
                 self.pickle_path + "spectral_flux.pkl",
                 self.pickle_path + "spectral_rolloff.pkl",
                 self.pickle_path + "mfccs0.pkl",
                 self.pickle_path + "mfccs1.pkl",
                 self.pickle_path + "mfccs2.pkl",
                 self.pickle_path + "mfccs3.pkl",
                 self.pickle_path + "mfccs4.pkl",
                 self.pickle_path + "mfccs5.pkl",
                 self.pickle_path + "mfccs6.pkl",
                 self.pickle_path + "mfccs7.pkl",
                 self.pickle_path + "mfccs8.pkl",
                 self.pickle_path + "mfccs9.pkl",
                 self.pickle_path + "mfccs10.pkl",
                 self.pickle_path + "mfccs11.pkl",
                 self.pickle_path + "mfccs12.pkl"]
        return files

    def pickle_files_exist(self):
        def files_exist(file_paths):
            for file_path in file_paths:
                if not file_path or not os.path.isfile(file_path):
                    return False
            return True
        files = self.get_file_paths()
        return files_exist(files)

    def need_to_fit_normalisers(self):
        return not self.pickle_files_exist()

    def get_random_normalised_example(self):
        with warnings.catch_warnings():
            warnings.simplefilter(self.warning_mode)
            if self.pickle_files_exist():
                files = self.get_file_paths()
                normalisers = [joblib.load(files[i]) for i in range(len(files))]

                features, patch = self.get_random_example()

                normalised_features = []
                index = 0
                for i in range(len(normalisers)):
                    if i in self.desired_features_indices:
                        normalised_features.append(normalisers[i].transform(features.T[index]))
                        index += 1

                norm_features = np.array(normalised_features)

                assert norm_features.T.shape == features.shape
                return (norm_features.T, patch)
            else:
                print "Please train normalisers using PluginFeatureExtractor.fit_normalisers()."

    def fit_normalisers(self, amount):
        if len(self.desired_features_indices) != 21:
            print "Please set the feature extractor to extract all available features!"
            return
        with warnings.catch_warnings():
            warnings.simplefilter(self.warning_mode)
            path = os.path.dirname(os.path.abspath(__file__)) + "/" + self.pickle_path
            print "\nBeginning to fit normalisers in " + path
            f, _ = self.get_random_example()
            (y, x) = f.shape

            # Get the features for fitting and reshape them.
            all_features = np.empty([amount, y, x])
            for i in trange(amount, desc="Rendering Examples"):
                features, _ = self.get_random_example()
                all_features[i] = features
            all_features = np.reshape(all_features, (x, amount, y))

            # Create the normalisers and fit them using the data.
            normalisers = [preprocessing.MinMaxScaler() for i in range(x)]
            for i in trange(x, desc="Fitting normalisers"):
                normalisers[i].fit_transform(all_features[i])
                print all_features[i].shape

            # Pickle the normalisers for future sessions.
            pickle_paths = self.get_file_paths()
            for i in range(x):
                if not os.path.exists(self.pickle_path):
                    os.makedirs(self.pickle_path)

                joblib.dump(normalisers[i], pickle_paths[i])
