# Ploting packages
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Date wrangling
from datetime import datetime, timedelta

# Data wrangling
import pandas as pd 

# The deep learning class
from deep_model import DeepModelTS

# Reading the configuration file
import yaml

# Directory managment 
import os

# Reading the hyper parameters for the pipeline
with open(f'{os.getcwd()}\\conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

# Reading the data 
d = pd.read_csv('prepped_data_1390-1400.csv', index_col=0)
d['Datetime'] = [datetime.strptime(x, '%Y-%m-%d') for x in d['Datetime']]

# Drawing input Data
data = d
print(f"Number of rows with missing values: {d.isnull().any(axis=1).mean()}")
data = data.loc['2011/03/21': '2022/03/30']
end_train = '2019/01/01'
end_validation = '2020/12/12'
data_train = data.loc[: end_train, :]
data_val   = data.loc[end_train:end_validation, :]
data_test  = data.loc[end_validation:, :]
print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}")
print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}")

fig, ax = plt.subplots(figsize=(12, 4))
data_train.plot(ax=ax, label='train', linewidth=1)
data_val.plot(ax=ax, label='validation', linewidth=1)
data_test.plot(ax=ax, label='test', linewidth=1)
ax.set_title('Electricity demand')
ax.legend();

# Sorting the values
d.sort_values('Datetime', inplace=True)

# Initiating the class 
deep_learner = DeepModelTS(
    data=d,
    Y_var='12',
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    epochs=conf.get('epochs'),
    train_test_split=conf.get('train_test_split') # The share of data that will be used for validation
)

# Fitting the model 
model = deep_learner.LSTModel()

# Making the prediction on the validation set
# Only applicable if train_test_split in the conf.yml > 0
yhat = deep_learner.predict()

if len(yhat) > 0:

    # Constructing the forecast dataframe
    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    # Ploting the forecasts
    plt.figure(figsize=(12, 8))
    for dtype in ['12', 'forecast']:
        plt.plot(
            'Datetime',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()
    
# Forecasting n steps ahead   

# Creating the model using full data and forecasting n steps ahead
deep_learner = DeepModelTS(
    data=d,
    Y_var='12',
    lag=24,
    LSTM_layer_depth=64,
    epochs=10,
    train_test_split=0
)

# Fitting the model 
deep_learner.LSTModel()

# Forecasting n steps ahead
n_ahead = 365
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Constructing the forecast dataframe
fc = d.tail(10).copy()
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(days=x + 1) for x in range(n_ahead)], 
    '12': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

# Ploting the forecasts 
plt.figure(figsize=(12, 8))
for col_type in ['original', 'forecast']:
    plt.plot(
        'Datetime',
        '12',
        data=fc[fc['type']==col_type],
        label=col_type
        )

plt.legend()
plt.grid()
plt.show()