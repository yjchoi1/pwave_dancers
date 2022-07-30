from tensorflow.keras import layers, models
from keras.utils.vis_utils import plot_model

# model
def Conv1D(time_steps, dim, fig_dir):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 24, activation="relu", input_shape=(time_steps, dim)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 12, activation="relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(16, 6, activation="relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(time_steps, activation="softmax"))
    model.summary()
    plot_model(model, to_file=fig_dir, show_shapes=True, show_layer_activations=True)
    return model

def seq2seq(time_steps, dim, fig_dir):
    model = models.Sequential()
    model.add(
        layers.LSTM(units=32, return_sequences=True, input_shape=(time_steps, dim), activation='tanh'))
    model.add(
        layers.LSTM(units=16, return_sequences=True, activation='tanh'))
    model.add(layers.Flatten())
    model.add(
        layers.Dense(time_steps, activation="softmax"))
    model.summary()
    plot_model(model, to_file=fig_dir, show_shapes=True, show_layer_activations=True)
    return model