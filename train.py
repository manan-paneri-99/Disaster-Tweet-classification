import tensorflow_hub as hub

from model import *
from preprocessing import *


def train_and_evaluate_model(index=0):
    preprocessor_url, module_url = get_model_type(index)

    text_input = tf.keras.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocessor_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(module_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']

    # net = tf.keras.layers.Dense(128, activation='relu')(net)

    net = tf.keras.layers.Dense(64, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)

    return tf.keras.Model(text_input, net)


def train(train_size=0.8, n_epochs=5, lr=3e-5, index=0, save_path='./saved_models/'):
    loss, epochs, metrics, optimizer = get_model_parts(n_epochs=n_epochs, lr=lr)

    train_df, valid_df = get_batches(train_size=train_size)

    classifier_model = train_and_evaluate_model(index)
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_name = get_model_name(index)
    histories = {}

    print("Training model with ", model_name)
    histories[model_name] = classifier_model.fit(train_df['text'], train_df['target'],
                                                 validation_data=(valid_df['text'], valid_df['target']), epochs=n_epochs
                                                 )

    return classifier_model, histories
