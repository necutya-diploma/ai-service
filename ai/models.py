from keras.layers import Dropout, LSTM, Bidirectional
from tensorflow.python.keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
)
from tensorflow.python.keras.models import Sequential


def dense_model(vocab_size: int, max_len: int, embeding_dim: int, drop_value: float, *args: int):
    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_model(vocab_size: int, max_len: int, embeding_dim: int, drop_value: float, n_lstm: int):
    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(LSTM(n_lstm, dropout=drop_value, return_sequences=True))
    model.add(LSTM(n_lstm, dropout=drop_value, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def bi_lstm_model(vocab_size: int, max_len: int, embeding_dim: int, drop_value: float, n_lstm: int):
    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(n_lstm, dropout=drop_value, return_sequences=True)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
