import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.preprocessing import text, sequence
from keras.layers import LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier


train = pd.read_csv("./input/train.csv")
test  = pd.read_csv("./input/test.csv")

list_sentences_train = train["comment_text"].fillna(" ").values
list_sentences_test = test["comment_text"].fillna(" ").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


train_sentence_filtered = train[   (train.toxic == 1)   | (train.severe_toxic == 1)  | (train.obscene == 1) | (train.threat == 1) \
                                 | (train.insult ==  1) | (train.identity_hate == 1) ]
list_filtered_train = train_sentence_filtered["comment_text"].fillna(" ").values

#print(list_sentences_train.shape)
#print(list_filtered_train.shape, list_filtered_train[1:4])

max_features = 20000
maxlen = 100

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_filtered_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_train = sequence.pad_sequences(list_tokenized_train[0:20000], maxlen=maxlen)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
y = train[list_classes].values[0:20000]


# define the Keras model graph
def get_model():
    embed_size = 300
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(GRU(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=get_model, epochs=2, verbose=1)
#droprate = [0.1, 0.2, 0.3, 0.5]
#embedsize = [64, 128, 200, 256, 300]
#num_cell = [30, 50, 80, 100]
batch_size = [10, 20, 30, 40, 50, 60]
param_grid = dict(batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train, y)

# summarize grid search results
means = grid_result.cv_results_['split0_train_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

