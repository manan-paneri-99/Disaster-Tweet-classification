def get_model_parts(index=2):
    models = ['bert_en_uncased_L-12_H-768_A-12', 'bert_en_uncased_L-24_H-1024_A-16',
              'bert_en_cased_L-12_H-768_A-12']
    preprocessors = ['https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                     'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                     'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3']
    modules = ['', 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3',
               'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3']

    return preprocessors[index], modules[index]