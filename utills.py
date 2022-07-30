import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
import ot

def read_data(path, n_motions=1000, time_steps=1001):

    data_inputs = []
    data_outputs = []
    # look for data files in path
    files = os.listdir(path)

    for file in files:
        data_dir = os.path.join(path, file)
        # allocate memory for data
        data_input = np.empty((n_motions, time_steps), dtype=np.float32)
        data_output = np.empty((n_motions, time_steps), dtype=np.float32)
        # read data
        with h5py.File(data_dir, "r") as f:
            data_input[:, :] = f["input"][:, :]
            data_output[:, :] = f["label"][:, :]
        # the last data is invalid one, so discard it
        data_input = data_input[:n_motions-1]
        data_output = data_output[:n_motions-1]
        # append
        data_inputs.append(data_input)
        data_outputs.append(data_output)

    # concat arrays
    data_inputs = np.concatenate(data_inputs)
    data_outputs = np.concatenate(data_outputs)

    # Shuffle
    shuffler = np.random.permutation(len(data_inputs))
    data_inputs = data_inputs[shuffler]
    data_outputs = data_outputs[shuffler]

    return data_inputs, data_outputs


def normalize(motions):

    normalized_motions = []
    for motion in motions:
        mean = np.mean(motion)
        std = np.std(motion)
        normalized_motion = (motion-mean)/std
        normalized_motions.append(normalized_motion)
    normalized_motions = np.array(normalized_motions)

    return normalized_motions


def make_probabililty(p_wave_signals):

    gaussian_distbs = []
    for p_wave_signal in p_wave_signals:
        sum = np.sum(p_wave_signal)
        gaussian_distb = p_wave_signal/sum
        gaussian_distbs.append(gaussian_distb)
    gaussian_distbs = np.array(gaussian_distbs)

    return gaussian_distbs


def one_hot(p_wave_signals):

    one_hots = []
    for p_wave_signal in p_wave_signals:
        index = np.argmax(p_wave_signal)
        one_hot = np.zeros(p_wave_signal.shape)
        one_hot[index] = 1
        one_hots.append(one_hot)
    one_hots = np.array(one_hots)

    return one_hots


def plot_result(prediction, test_dataset_y, test_dataset_x, dataset_id, time, save_dir):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 3.5))
    axs[0].plot(time, prediction[dataset_id, :], label='Prediction')
    axs[0].plot(time, test_dataset_y[dataset_id, :], label='Label')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Probability")
    axs[0].legend()
    axs[1].plot(time, test_dataset_x[dataset_id, :], label='text_x', alpha=0.5)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    fig.tight_layout()
    plt.show()
    plot_name = f"result-{dataset_id}.png"
    if save_dir == None:
        pass
    else:
        fig.savefig(os.path.join(save_dir, plot_name), format='png')


# Loss

# Loss
def avg_loss(ids, prediction, true, n_time_steps):
    Ws = []
    for i in ids:
        a = true[i].reshape((n_time_steps))  # label
        b = prediction[i].reshape((n_time_steps))  # prediction
        x = np.arange(n_time_steps, dtype=np.float64)
        M = ot.dist(x.reshape((n_time_steps, 1)), x.reshape((n_time_steps, 1)))
        M /= M.max()
        G0 = ot.emd(a, b, M)
        W = np.sum(G0*M)
        Ws.append(W)
    sum_loss = np.sum(Ws)
    avg_loss = sum_loss/n_time_steps

    return avg_loss, Ws