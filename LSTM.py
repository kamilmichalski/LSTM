from word2vec_encoding import data, embedding_weights, w2v_vocabulary_dimension, w2v_vocabulary_size
from data_preprocessing import target

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import save_model

from sklearn.cross_validation import KFold

batch_size = 50
n_epoch = 50
sentence_length = 40

target = sequence.pad_sequences(target, maxlen=sentence_length, value=[0, 1])
data = sequence.pad_sequences(data, maxlen=sentence_length)

slicing_point = int(len(data)*0.9)

target_train = target[:slicing_point]
data_train = data[:slicing_point]
target_test = target[slicing_point+1:]
data_test = data[slicing_point+1:]


def create_model():

    inputs = Input(shape=(sentence_length,), dtype='int32')

    x = Embedding(
        input_dim=w2v_vocabulary_size,
        output_dim=w2v_vocabulary_dimension,
        input_length=sentence_length,
        mask_zero=True,
        weights=[embedding_weights]
    )(inputs)

    lstm_out = LSTM(200, return_sequences=True)(x)

    regularized_data = Dropout(0.3)(lstm_out)

    predictions = TimeDistributed(Dense(2, activation='sigmoid'))(regularized_data)

    model = Model(input=inputs, output=predictions)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def fit_model(X_train, y_train, X_test, y_test):
    hist = model.fit(X_train, y_train, batch_size=50, nb_epoch=50,
                 validation_data=(X_test, y_test), verbose=2)
    return hist

if __name__ == '__main__':

    kf = KFold(len(data_train), 10)

    for index, (train, test) in enumerate(kf):

        print '''

        Training {0} fold

        '''.format(index)

        X_train, X_test = data_train[train], data_train[test]
        Y_train, Y_test = target_train[train], target_train[test]
        model = None
        model = create_model()
        fit_model(X_train, Y_train, X_test, Y_test)
        model.save('model_' + str(index) + '.h5')
