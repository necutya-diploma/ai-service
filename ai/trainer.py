# імпорт залежностей
import keras
import seaborn as sns
import wordcloud
import warnings
import pickle
import pandas as pd

from typing import Any
from matplotlib import pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import ndarray
from keras.layers import Dropout, LSTM, Bidirectional
from tensorflow.python.keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
)
from tensorflow.python.keras.models import Sequential
from typing import Callable, List
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def visualize_train_result(model_name: str, history: Any):
    metrics = pd.DataFrame(history.history)

    metrics.rename(
        columns={'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'},
        inplace=True,
    )

    def plot_graphs1(var1, var2, string, model_name):
        metrics[[var1, var2]].plot()
        plt.title(f'{model_name}: training and Validation ' + string)
        plt.xlabel('Number of epochs')
        plt.ylabel(string)
        plt.legend([var1, var2])
        plt.show()

    plot_graphs1('Training_Loss', 'Validation_Loss', 'loss', model_name)
    plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy', model_name)


# функція попредньої обробки тексту
def preprocess_text(
        train_set: List[str],
        test_set: List[str],
        max_len: int,
        vocab_size: int,
        char_level: bool = False,
        oov_token: str = "<OOV>",
        trunc_type: str = "post",
        padding_type: str = "post"
) -> (ndarray, ndarray):
    tokenizer = Tokenizer(num_words=vocab_size, char_level=char_level, oov_token=oov_token)
    tokenizer.fit_on_texts(train_set)

    tot_words = len(tokenizer.word_index)
    print('There are %s unique tokens in training data. ' % tot_words)

    training_sequences = tokenizer.texts_to_sequences(train_set)
    training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(test_set)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_len,
                                   padding=padding_type, truncating=trunc_type)

    with open('model/tokenizer.pkl', 'wb') as output:
        pickle.dump(tokenizer, output, pickle.HIGHEST_PROTOCOL)

    return training_padded, testing_padded


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


def show_wordcloud(df, title):
    text = ' '.join(df['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)

    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords, background_color='black',
                                        colormap='viridis', width=800, height=600).generate(text)

    plt.figure(figsize=(10, 7), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()


def train(dataset_path: str, compile_models: List[Callable[[int, int, int, float, int], keras.Model]]):
    df = pd.read_csv(dataset_path, encoding='latin-1')

    data = df.copy()
    print(f'Value Count: {data.value_counts()}')

    sns.countplot(data['label'])
    plt.show()

    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    data['text'] = data['text'].map(lambda x: str(x))

    data_ham = data[data['label'] == 0].copy()
    data_spam = data[data['label'] == 1].copy()

    show_wordcloud(data_spam, "Spam messages")
    show_wordcloud(data_ham, "Ham messages")

    ham_msg_df = data_ham.sample(n=len(data_spam), random_state=44)
    spam_msg_df = data_spam

    msg_df = ham_msg_df.append(spam_msg_df).reset_index(drop=True)

    sns.countplot(msg_df['label'])
    plt.show()

    msg_df['text_length'] = msg_df['text'].apply(len)

    X = data['text'].values
    Y = data['label'].values

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=434)

    max_len = 25
    vocab_size = 500
    embeding_dim = 16
    drop_value = 0.2
    epoch_num = 20
    n_lstm = 20

    train_processed_x, test_processed_x = preprocess_text(
        train_x,
        test_x,
        max_len=max_len,
        vocab_size=vocab_size
    )

    for compile_model in compile_models:
        model = compile_model(vocab_size, max_len, embeding_dim, drop_value, n_lstm)
        print(model.summary())

        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(
            train_processed_x,
            train_y,
            epochs=epoch_num,
            validation_data=(test_processed_x, test_y),
            callbacks=[early_stop],
            verbose=1,
        )

        print(f"Train result: {model.evaluate(test_processed_x, test_y)}")
        visualize_train_result(compile_model.__name__, history)

        model.save(f'model/{compile_model.__name__}', 'wb')
