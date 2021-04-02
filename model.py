import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_text as text


def get_model_type(index=0):
    preprocessors = ['https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                     'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                     'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3']
    modules = ['https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
               'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3',
               'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3']

    return preprocessors[index], modules[index]


def get_model_name(index=1):
    models = ['bert_en_uncased_L-12_H-768_A-12', 'bert_en_uncased_L-24_H-1024_A-16',
              'bert_en_cased_L-12_H-768_A-12']
    return models[index]


def get_model_parts(n_epochs=5, lr=3e-5):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    epochs = n_epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return loss, epochs, metrics, optimizer
