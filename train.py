import numpy as np
import random
import os
from dataloader import Dataloader
import models

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import warnings
warnings.filterwarnings(action='ignore')

# ==== seed ==== #
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # np
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    tf.random.set_seed(seed) # tensorflow

my_seed = 777
my_seed_everywhere(my_seed)

# ==== gpu usage ==== #
gpu_id = 0
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) # check available gpu devices

# ==== model preparation ==== #
input_len = 24
n_features = 7
model_name = 'bilstm'  # model name for save
model = models.bilstm(input_len, n_features)

# ==== model train ===== #
# train -> 8 months for each year, test -> 4 months for each year
# After training a total of 12 models, the results from the same month are ensembled (averaged) to produce the results.
for m in range(1, 13):
    test_month = []
    for _ in range(4):
        if m > 12: n = m % 12
        else: n = m
        test_month.append(str(n).zfill(2))
        m += 1

    test_month = tuple(test_month)
    group_path = '../group_list/'
    dataset_path = '../storm/30m/'
    dataloader = Dataloader(group_path, input_len, mode='train', test_month=test_month, dataset_path=dataset_path)
    
    x_train, y_train, train_input_t, train_target_t = dataloader.make_dataset()
    # x_train --> (Bt, Bx, By, Bz, V, N, T)

    # custom loss function: weighted rmse
    import keras.backend as K
    def weighted_rmse(y_true, y_pred):
        x = 0
        for i in range(1, 13): x += i
        # weight = [(72-i)/x for i in range(72)]
        weight = []
        for i in range(12):
            weight.append((12 - i) / x)
        loss = K.square(y_pred - y_true)
        loss = loss * weight
        loss = K.mean(loss, axis=1)
        return loss

    # model train
    optimizers = Adam(lr=0.0001)
    model.compile(loss=weighted_rmse, optimizer=optimizers)
    # model.compile(loss='mse', optimizer=optimizers)

#    from tensorflow.keras import callbacks
#    callback = [callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    hist = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=1, verbose=1)

    # model save
    from tensorflow.keras.callbacks import ModelCheckpoint
    model_json = model.to_json()
    path = f'./models/{model_name}'
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{model_name} {test_month}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{path}/{model_name} {test_month}.h5")
    print("Saved model to disk", test_month)
