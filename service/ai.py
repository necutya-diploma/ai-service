import pickle

import keras
import numpy as np

import tensorflow
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


class AIService:

    def __init__(self, model_path: str, tokenizer_path: str, *args, **kwargs):
        self.tokenizer: Tokenizer = None
        self.model: keras.Model = tensorflow.keras.models.load_model(model_path)

        with open(tokenizer_path, 'rb') as output:
            self.tokenizer = pickle.load(output)

    def check_message(self, message) -> (str, bool, float):
        sequence = self.tokenizer.texts_to_sequences([message])
        processed = pad_sequences(
            sequence,
            maxlen=25,
            padding='post',
            truncating='post',
        )
        res = self.model.predict(processed)[0]

        generated_percent = round(res[np.argmax(res)].astype("float32").item(), 3) * 100
        is_generated = generated_percent > 50

        return message, is_generated, generated_percent
