import numpy as np
import pandas as pd
import os
from dataloader import Dataloader
import glob

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import model_from_json

import warnings
warnings.filterwarnings('ignore')

model_name = 'bilstm'
models = sorted(glob.glob(f'./models/{model_name}/*.json'))
weights = sorted(glob.glob(f'./models/{model_name}/*.h5'))

group_path = '../group_list/'
dataset_path = '../storm/30m/'

group_path = '../group_list/'
dataset_path = '../storm/30m/'

total_df = pd.DataFrame()
for file in sorted(glob.glob(group_path+'*.csv')):
    df = pd.read_csv(file, engine='python').interpolate(direction='both')
    total_df = total_df.append(df, ignore_index=True)
bz = total_df['Bz'].values

for i, model in enumerate(models):
    weight = weights[i]
    json_file = open(model, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight)

    idx1 = model.find('(')
    idx2 = model.find(')')

    filename = model[idx1+1:idx2+2]
    test_month_ = model[idx1+1:idx2].split(', ')
    test_month = [int(t.replace("'", "")) for t in test_month_]

    group_path = '../group_list/'
    dataset_path = '../storm/30m/'
    n_features = 7
    input_len = 24
    dataloader = Dataloader(group_path, input_len, mode='test', test_month=test_month, dataset_path=dataset_path)
    x_test, y_test, test_input_t, test_target_t = dataloader.make_dataset()
    inp_bz = x_test[:, :, 3]
  
    y_pred = loaded_model.predict(x_test, batch_size=1)

    # save the results
    savepath = f'./npy result/{model_name}'
    os.makedirs(savepath, exist_ok=True)
    np.save(f'{savepath}/pred {filename}npy', y_pred)
    np.save(f'{savepath}/test {filename}npy', y_test)
    np.save(f'{savepath}/input {filename}npy', inp_bz)
    np.save(f'{savepath}/target_t {filename}npy', test_target_t)
    np.save(f'{savepath}/input_t {filename}npy', test_input_t)
    print(f'npys are saved {filename}')
