from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys
import struct

def plot_error(errors):
    """Plot error from testing batches during optimisation."""
    plt.title("Average error over time")
    plt.plot(np.arange(len(errors)), errors)
    plt.show()

def plot_best_features(predicted, actual):
    """Plot color map representing features for easy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(actual, interpolation='nearest', cmap='magma')
    ax1.set_title('Actual Features')
    im = ax2.imshow(predicted, interpolation='nearest', cmap='magma')
    ax2.set_title('Predicted Features')
    ax1.axis('off')
    ax2.axis('off')
    fig.colorbar(im, orientation='vertical')
    plt.show()


def plot_best_patch(predicted, actual):
    """Plot the parameters used to make the closest prediction / actual feature pair."""
    fit, ax = plt.subplots()
    index = np.arange(len(predicted))
    bar_width = 0.3
    rects1 = plt.bar(index, predicted, bar_width, color='#6A2C70', label='predicted')
    rects2 = plt.bar(index + bar_width, actual, bar_width, color='#F08A5D', label='actual')
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Synth patch that produced closest features')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_error_histogram(all_errors):
    """Plot the distribution of errors across the current batch."""
    plt.hist(all_errors, bins=20)
    plt.title("Error Histogram")
    plt.xlabel("Average Error Over Batch")
    plt.ylabel("Frequency")
    plt.show()

def get_stats(extractor, predicted_patches, actual_features, actual_patches):
    """Get the difference between the predicted and actual batch of patches."""
    average_error = 0
    lowest_error = sys.float_info.max
    all_errors = []
    for i in range(len(predicted_patches)):
        patch = extractor.add_patch_indices(predicted_patches[i])
        predicted_features = extractor.get_features_from_patch(patch)
        error = np.add.reduce(np.abs(predicted_features - actual_features[i]).flatten())
        average_error += error

        if error < lowest_error:
            best_pairs = {
                'predicted_features' : predicted_features,
                'actual_features' : actual_features[i],
                'predicted_patch' : predicted_patches[i],
                'actual_patch' : actual_patches[i]
            }
        all_errors += [error]
    return ((average_error / len(predicted_patches)), best_pairs, all_errors)

def display_stats(stats):
    """Using the passed in stats, plot average error over the test batch,
       display best features and synth patches."""
    (average_error, best_pairs, all_errors) = stats
    plot_best_features(best_pairs['predicted_features'], best_pairs['actual_features'])
    plot_best_patch(best_pairs['predicted_patch'], best_pairs['actual_patch'])
    plot_error_histogram(all_errors)

def get_batches(train_batch_size, test_batch_size, extractor):
    (f, p) = extractor.get_random_normalised_example()
    f_shape = np.array(f).shape
    train_batch_x = np.zeros((train_batch_size, f_shape[0], f_shape[1]), dtype=np.float32)
    train_batch_y = np.zeros((train_batch_size, p.shape[0]), dtype=np.float32)
    for i in trange(train_batch_size, desc="Generating Train Batch"):
        (features, parameters) = extractor.get_random_normalised_example()
        train_batch_x[i] = features
        train_batch_y[i] = parameters

    test_batch_x = np.zeros((test_batch_size, f_shape[0], f_shape[1]), dtype=np.float32)
    test_batch_y = np.zeros((test_batch_size, p.shape[0]), dtype=np.float32)
    for i in trange(test_batch_size, desc="Generating Test Batch"):
        (features, parameters) = extractor.get_random_normalised_example()
        test_batch_x[i] = features
        test_batch_y[i] = parameters
    return train_batch_x, train_batch_y, test_batch_x, test_batch_y

# http://stackoverflow.com/questions/15576798/create-32bit-float-wav-file-in-python
def float32_wav_file(sample_array, sample_rate):
    byte_count = (len(sample_array)) * 4  # 32-bit floats
    wav_file = ""
    # write the header
    wav_file += struct.pack('<ccccIccccccccIHHIIHH',
        'R', 'I', 'F', 'F',
        byte_count + 0x2c - 8,  # header size
        'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
        0x10,  # size of 'fmt ' header
        3,  # format 3 = floating-point PCM
        1,  # channels
        sample_rate,  # samples / second
        sample_rate * 4,  # bytes / second
        4,  # block alignment
        32)  # bits / sample
    wav_file += struct.pack('<ccccI', 'd', 'a', 't', 'a', byte_count)
    for sample in sample_array:
        wav_file += struct.pack("<f", sample)
    return wav_file

def write_wavs(predicted_patch, actual_patch, extractor):
    """Normalises and writes audio data to wav file."""
    extractor.set_patch(extractor.add_patch_indices(predicted_patch))
    p_audio = extractor.get_audio_frames()
    p_audio /= np.max(np.abs(p_audio), axis=0)
    with open("predicted.wav", "w") as p_wav_file:
        p_wav_file.write(float32_wav_file(p_audio, 44100))
    extractor.set_patch(extractor.add_patch_indices(actual_patch))
    a_audio = extractor.get_audio_frames()
    a_audio /= np.max(np.abs(a_audio), axis=0)
    with open("actual.wav", "w") as a_wav_file:
        a_wav_file.write(float32_wav_file(a_audio, 44100))
