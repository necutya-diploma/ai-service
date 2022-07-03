import pickle
from typing import List

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import ndarray


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
    # tokenization
    tokenizer = Tokenizer(num_words=vocab_size, char_level=char_level, oov_token=oov_token)
    tokenizer.fit_on_texts(train_set)

    tot_words = len(tokenizer.word_index)
    print('There are %s unique tokens in training data. ' % tot_words)

    # Sequencing and padding on training and testing
    training_sequences = tokenizer.texts_to_sequences(train_set)
    training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(test_set)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_len,
                                   padding=padding_type, truncating=trunc_type)

    with open('model/tokenizer.pkl', 'wb') as output:
        pickle.dump(tokenizer, output, pickle.HIGHEST_PROTOCOL)

    return training_padded, testing_padded
