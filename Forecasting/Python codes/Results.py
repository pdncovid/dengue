import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf
import numpy as np
# Seed value
# Apparently you may use different seed values at each stage
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
from keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

#LSTM model
df = pd.read_csv(r'C:\Users\HP\Documents\4th year\7th sem\EE405\DATA\Colombo_Dengue.csv')
print(df)
df.info()
#df['Month'] = pd.to_Month(df['Month'], infer_Month_format=True)
#df.set_index('Month')[['Rainfall', 'Temp', 'pop,000', 'Cases']].plot(subplots=True)
#plt.show()
df_input = df[['Rainfall', 'Temp','Humidity', 'Cases']]
print(df_input)
print(df_input.describe())

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input)
print(data_scaled)


features = data_scaled
target= data_scaled[:, 3]
X= TimeseriesGenerator(features, target, length=12, sampling_rate=1)[0]
print(X)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)
print(X_train.shape)
print(X_test.shape)

train_generator = TimeseriesGenerator(X_train, y_train, length=12, sampling_rate=1)
test_generator = TimeseriesGenerator(X_test, y_test, length=12, sampling_rate=1)
print(train_generator[0])

model =tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(12, 4), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))
print(model.summary())

model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
hist = model.fit_generator(train_generator, epochs=300, validation_data=test_generator, shuffle=False)
model.evaluate_generator(test_generator, verbose=1)

predictions=model.predict_generator(test_generator)
print(predictions.shape[0])
print(predictions)

print(X_test)
x=X_test[:, :3][12:]
print(x)
df_pred=pd.concat([pd.DataFrame(x), pd.DataFrame(predictions)], axis=1)
rev_trans=scaler.inverse_transform(df_pred)
print(rev_trans)
df_final=df_input[predictions.shape[0]*-1:]
print(df_final.count())
df_final['predi_cases']=rev_trans[:, 3]
print(df_final)

plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

#FSL model
seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
from keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

df1 = pd.read_csv(r'C:\Users\HP\Documents\4th year\8th sem\EE406\Time_series_data.csv')
print(df1)
df1.info()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df1)
print(data_scaled)
# preparing independent and dependent features
def prepare_data(timeseries_data, n_features):
    X, y = [], []
    for i in range(len(timeseries_data)):
        # find the end of this pattern
        end_ix = i + n_features
        # check if we are beyond the sequence
        if end_ix > len(timeseries_data) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

#x = data_scaled.values.tolist()
data=[]
for i in range(0, 16):
    for j in range(0, len(data_scaled)):
        data.append(data_scaled[j][i])
print(data)

#define input sequence

timeseries_data = data
X_train, X_test= train_test_split(timeseries_data, test_size=0.20, random_state=123,
                                                        shuffle=False)
print(X_train)
print(X_test)
# choose a number of time steps
n_steps = 12
# split train set into samples
X1, y1 = prepare_data(X_train, n_steps)

print(X1), print(y1)
print(X1.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X1 = X1.reshape((X1.shape[0], X1.shape[1], n_features))

# split test set into samples
X2, y2 = prepare_data(X_test, n_steps)

print(X2), print(y2)
print(X2.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X2 = X2.reshape((X2.shape[0], X2.shape[1], n_features))
# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(12, 1), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse',  metrics=['accuracy'])
# fit model
hist=model.fit(X1, y1, epochs=40, validation_data=(X2, y2), shuffle=False, verbose=1)
plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
#plt.show()

df2 = pd.read_csv(r'C:\Users\HP\Documents\4th year\8th sem\EE406\infectious diseases data.csv')
print(df2)
scaler = MinMaxScaler()
data_scaled1 = scaler.fit_transform(df2)
print(data_scaled1)
data_In=[]
for i in range(0, 5):
    for j in range(0, len(data_scaled1)):
        data_In.append(data_scaled1[j][i])
print(data_In)
X_train, X_test= train_test_split(timeseries_data, test_size=0.20, random_state=123,
                                                        shuffle=False)
print(X_train)
print(X_test)
# choose a number of time steps
n_steps = 12
# split train set into samples
X1, y1 = prepare_data(X_train, n_steps)

print(X1), print(y1)
print(X1.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X1 = X1.reshape((X1.shape[0], X1.shape[1], n_features))


X2, y2 = prepare_data(X_test, n_steps)

print(X2), print(y2)
print(X2.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X2 = X2.reshape((X2.shape[0], X2.shape[1], n_features))
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(12, 1), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse',  metrics=['accuracy'])
hist=model.fit(X1, y1, epochs=40, validation_data=(X2, y2), shuffle=False, verbose=1)
plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

df3 = pd.read_csv(r'C:\Users\HP\Documents\4th year\8th sem\EE406\Dengue_ph.csv')
print(df3)
scaler = MinMaxScaler()
data_scaled1 = scaler.fit_transform(df3)
print(data_scaled1)
data_In=[]
for i in range(0, 14):
    for j in range(0, len(data_scaled1)):
        data_In.append(data_scaled1[j][i])
print(data_In)
X_train, X_test= train_test_split(timeseries_data, test_size=0.20, random_state=123,
                                                        shuffle=False)
print(X_train)
print(X_test)
# choose a number of time steps
n_steps = 12
# split train set into samples
X3, y3 = prepare_data(X_train, n_steps)

print(X3), print(y3)
print(X3.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X3 = X3.reshape((X3.shape[0], X3.shape[1], n_features))


X4, y4 = prepare_data(X_test, n_steps)

print(X4), print(y4)
print(X4.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X4 = X4.reshape((X4.shape[0], X4.shape[1], n_features))
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(12, 1), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse',  metrics=['accuracy'])
hist=model.fit(X3, y3, epochs=40, validation_data=(X4, y4), shuffle=False, verbose=1)
plt.figure(3)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')


df = pd.read_csv(r'C:\Users\HP\Documents\4th year\7th sem\EE405\DATA\Colombo_Dengue.csv')
print(df)
df.info()
#df['Month'] = pd.to_Month(df['Month'], infer_Month_format=True)
#df.set_index('Month')[['Rainfall', 'Temp', 'pop,000', 'Cases']].plot(subplots=True)
#plt.show()
df_input = df[['Rainfall', 'Temp','Humidity', 'Cases']]
print(df_input)
print(df_input.describe())
plt.figure(4)
plt.plot(df['Month'], df['Cases'])
plt.ylabel('No of Dengue Cases')
plt.xlabel('Year')
plt.xticks([1, 12, 24, 36, 48, 60, 72, 84], ["2010", "2011", "2012", "2013","2014", "2015", "2016", "2018"], rotation=40)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input)
print(data_scaled)


features = data_scaled
target= data_scaled[:, 3]
X= TimeseriesGenerator(features, target, length=12, sampling_rate=1)[0]
print(X)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)
print(X_train.shape)
print(X_test.shape)

train_generator = TimeseriesGenerator(X_train, y_train, length=12, sampling_rate=1)
test_generator = TimeseriesGenerator(X_test, y_test, length=12, sampling_rate=1)
print(train_generator[0])

model =tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(12, 4), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse',  metrics=['accuracy'])
hist = model.fit_generator(train_generator, epochs=50, validation_data=test_generator, shuffle=False)
model.evaluate_generator(test_generator, verbose=1)

predictions=model.predict_generator(test_generator)
print(predictions.shape[0])
print(predictions)

print(X_test)
x=X_test[:, :3][12:]
print(x)
df_pred=pd.concat([pd.DataFrame(x), pd.DataFrame(predictions)], axis=1)
rev_trans=scaler.inverse_transform(df_pred)
print(rev_trans)
df_final3=df_input[predictions.shape[0]*-1:]
print(df_final3.count())
df_final3['predi_cases3']=rev_trans[:, 3]
print(df_final3)


#VAR model
df = pd.read_csv(r'C:\Users\HP\Documents\4th year\7th sem\EE405\DATA\Colombo_Dengue.csv')
data = df[['Rainfall', 'Temp','Humidity', 'Cases']]
print(data)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled)
#creating the train and validation set
train = data_scaled[:int(0.875*(len(data)))]
valid = data_scaled[int(0.875*(len(data))):]
print(train)
print(valid)
#fit the model

model = VAR(endog=train)
model_fit = model.fit(maxlags=4)
print(model_fit.summary())
# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
print(prediction)
rev_trans=scaler.inverse_transform(prediction)
print(rev_trans)
predi=rev_trans[:, 3]
print(predi)
p=[]
for i in range(4, 12):
  p.append(predi[i])
print(p)

day_new = np.arange(5, 13)
day_pred = np.arange(5, 13)
fig, ax = plt.subplots()

ax.plot(day_new, df_final['Cases'], color='blue', marker='o', label='Actual')
ax.plot(day_pred, df_final3['predi_cases3'], color='red', marker='o',label='Few-Shot Learning Model ')
ax.plot(day_pred, df_final['predi_cases'], 'm--', marker='o', label='Naive LSTM Model')
ax.plot(day_pred, p, 'g--',marker='o', label='VAR Model')
#df_final[['Cases', 'predi_cases']].plot()
x_labels = ['May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
ax.set_xticks(range(5,13))
ax.set_xticklabels(x_labels)
plt.title("Dengue Cases Reported in Colombo in 2018")
plt.ylabel('Number of Dengue Cases')
plt.xlabel('Month')

plt.legend()

plt.show()