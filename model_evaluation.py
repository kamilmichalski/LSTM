from LSTM import target_test, data_test

from keras.models import load_model
from glob import glob
import numpy as np

from sklearn.metrics import precision_recall_fscore_support


def one_hot_to_zero_ones(sequences_array):
    zero_ones = []
    for sentence in sequences_array:
        for encoded_word in sentence:
            zero_ones.append(encoded_word)
    zero_ones = [1 if i == [1, 0] else 0 for i in zero_ones]
    return zero_ones


if __name__ == '__main__':

    predictions = []
    scores = []
    precision = []
    recall = []
    fscore = []

    for filename in glob('*.h5'):
        model = load_model(filename)
        print 'loaded model {0}'.format(filename)
        predictions.append(model.predict(data_test, batch_size=50, verbose=2))
    for prediction in predictions:
        scores.append(
            precision_recall_fscore_support(one_hot_to_zero_ones(target_test),
                                            one_hot_to_zero_ones(prediction)))
    for score in scores:
        precision.append(score[0])
        recall.append(score[1])
        fscore.append(score[2])

    print '''
                Model evaluation

                    Sentiment    Non-sentiment
    Precision          {0}            {1}
    Recall             {2}            {3}
    F-score            {4}            {5}

    '''.format(np.mean(np.array(precision), axis=0)[0], np.mean(np.array(precision), axis=0)[1],
               np.mean(np.array(recall), axis=0)[0], np.mean(np.array(recall), axis=0)[1],
               np.mean(np.array(fscore), axis=0)[0], np.mean(np.array(fscore), axis=0)[1])


