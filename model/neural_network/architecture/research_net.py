import tensorflow as tf


class MalwareEntropyAnalyzer:
    @staticmethod
    def generate(input_shape=None, number_of_classes=2):
        if number_of_classes is None:
            raise Exception("[ERROR] Invalid number of classes.")

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(8, activation='tanh', input_shape=input_shape))
        model.add(tf.keras.layers.Dense(units=6, activation='relu'))
        model.add(tf.keras.layers.Dense(units=number_of_classes, activation='softmax'))

        return model
