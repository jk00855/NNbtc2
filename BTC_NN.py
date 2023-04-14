# import libraries and modules needed for the project  
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the new.csv data into a pandas dataframe
df = pd.read_csv('new.csv')

# Convert the date column to a datetime type and set it as the index of the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select the close price as the target variable and the remaining columns as features
target_col = 'Close'
features = df.columns.drop(target_col)

# Normalize the data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df[target_col] = scaler.fit_transform(df[[target_col]])

# Split the data into training and test sets
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[:train_size], df[train_size:]

# Convert the data into sequences for the LSTM model
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def create_sequences(data, seq_length):
    data_sequences = []
    for i in range(len(data) - seq_length + 1):
        data_sequences.append(data[i:i+seq_length])
    return np.array(data_sequences)


# Set the sequence length and number of features
seq_length = 500
n_features = len(features) 


# Convert the training and test data into sequences
print("Converting data into sequences...")
X_train = create_sequences(train[features].values, seq_length)
print("X_train shape: {}".format(X_train.shape))
y_train = create_sequences(train[target_col].values, seq_length)
print("y_train shape: {}".format(y_train.shape))
X_test = create_sequences(test[features].values, seq_length)
print("X_test shape: {}".format(X_test.shape))
y_test = create_sequences(test[target_col].values, seq_length)
print("y_test shape: {}".format(y_test.shape))

# Reshape the data for the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], seq_length, n_features))
print("X_train shape: {}".format(X_train.shape))
X_test = np.reshape(X_test, (X_test.shape[0], seq_length, n_features))
print("X_test shape: {}".format(X_test.shape))



# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, n_features)))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1, shuffle=False)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test, batch_size=32)
print(f'Test loss: {test_loss:.4f}')


# Make predictions on the test data
y_pred = model.predict(X_test)

# Invert the normalization of the data
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))[-seq_length:]
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))[:seq_length]

# Get the dates for the real and predicted prices
dates_test = df.index[train_size:]
dates_pred = pd.date_range(dates_test[-1], periods=len(y_pred))

# Plot a graph showing the real and predicted prices
plt.plot(dates_test[-seq_length:], y_test, color='red', label='Real prices')
plt.plot(dates_pred, y_pred, color='blue', label='Predicted prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Real vs Predicted Prices')
plt.legend()
plt.show()


