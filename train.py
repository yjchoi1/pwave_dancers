import os
import utills
from utills import read_data
from utills import normalize
from utills import make_probabililty
import numpy as np
import model
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

#%%
# inputs
path = 'dataset_broad'  # location of dataset
train_split = 0.80
val_split = 0.90
test_split = 1.00
dt = 0.01
output_base = 'outputs_broad'
model_option = "Conv1D"  # select from `Conv1D` or `seq2seq`
model_name = f'p-dancer_{model_option}.keras'
plot_dir = os.path.join(output_base, f'plots_{model_option}')
model_dir = os.path.join(output_base, f'model_{model_option}')

#%%
# make dirs
if not os.path.isdir(output_base):
    os.makedirs(output_base, exist_ok=True)
if not os.path.isdir(output_base):
    os.makedirs(output_base, exist_ok=True)
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir, exist_ok=True)

# Read data
# `data_inputs` are ground motions, and `data_outputs` are p-wave arrival probability distribution
data_inputs, data_outputs = read_data(path)

# See the data shape
n_datasets = data_inputs.shape[0]
n_time_steps = data_inputs.shape[1]

# Normalize ground motions
normalized_motions = normalize(data_inputs).reshape(n_datasets, n_time_steps, 1)

# Convert `data_outputs`to probability distribution
pwave_probabilities = make_probabililty(data_outputs).reshape(n_datasets, n_time_steps, 1)

# # Convert `data_outputs`to one_hot encoded array
# one_hots = utills.one_hot(data_outputs).reshape(n_datasets, n_time_steps, 1)

# train-test split
train_dataset_x = normalized_motions[:round(n_datasets*train_split)]
train_dataset_y = pwave_probabilities[:round(n_datasets*train_split)]
val_dataset_x = normalized_motions[round(n_datasets*train_split):round(n_datasets*val_split)]
val_dataset_y = pwave_probabilities[round(n_datasets*train_split):round(n_datasets*val_split)]
test_dataset_x = normalized_motions[round(n_datasets*val_split):round(n_datasets*test_split)]
test_dataset_y = pwave_probabilities[round(n_datasets*val_split):round(n_datasets*test_split)]


# make model
if model_option == 'Conv1D':
    model = model.Conv1D(
        time_steps=n_time_steps, dim=1, fig_dir=os.path.join(plot_dir, 'model_structure.png')
    )
elif model_option == 'seq2seq':
    model = model.seq2seq(
        time_steps=n_time_steps, dim=1, fig_dir=os.path.join(plot_dir, 'model_structure.png')
    )

# compile model
callbacks = [keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, model_name), save_best_only=False)]

if os.path.exists(os.path.join(model_dir, model_name)):
    model = keras.models.load_model(os.path.join(model_dir, model_name))
else:
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["mse"])
    history = model.fit(train_dataset_x,
                        train_dataset_y[:, :, 0],
                        batch_size=16,
                        epochs=50,
                        validation_data=(val_dataset_x, val_dataset_y),
                        callbacks=callbacks)
    model = keras.models.load_model(os.path.join(model_dir, model_name))

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Val loss")
    plt.legend()
    plt.show()
#%% Prediction
prediction_train = model.predict(train_dataset_x)
prediction_test = model.predict(test_dataset_x)
prediction_val = model.predict(val_dataset_x)

time = np.arange(0, dt*n_time_steps, dt)
ids = np.arange(len(test_dataset_x))
#%% Plot

# for i in ids:
#     utills.plot_result(
#         prediction=prediction_train, test_dataset_y=train_dataset_y, test_dataset_x=train_dataset_x,
#         dataset_id=i, time=time, save_dir=plot_dir)
# for i in ids:
#     utills.plot_result(
#         prediction=prediction_val, test_dataset_y=val_dataset_y, test_dataset_x=val_dataset_x,
#         dataset_id=i, time=time, save_dir=plot_dir)
for i in ids:
    utills.plot_result(
        prediction=prediction_test, test_dataset_y=test_dataset_y, test_dataset_x=test_dataset_x,
        dataset_id=i, time=time, save_dir=plot_dir)

#%% loss
id2s = [1, 19]
for i in id2s:
    utills.plot_result(
        prediction=prediction_test, test_dataset_y=test_dataset_y, test_dataset_x=test_dataset_x,
        dataset_id=i, time=time, save_dir=plot_dir)

loss_val, Ws = utills.avg_loss(ids=id2s, prediction=prediction_test, true=test_dataset_y, n_time_steps=n_time_steps)

with open(os.path.join(model_dir, 'loss.txt'), 'w') as f:
    f.write(f'{loss_val}')