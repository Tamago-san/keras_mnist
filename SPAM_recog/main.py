import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding


pd.set_option("display.max_colwidth", 100)


filename = './dataset/SMSSpamCollection'

dataset_df = pd.read_csv(filename,
                         sep='\t',
                         header=None)
                         
print(dataset_df)
dataset_df.rename({0: 'label', 1: 'text'},
                   axis=1,
                   inplace=True)
                   

print(dataset_df.head())
dataset_df['category'] = dataset_df.apply(lambda r: 1 if r['label'] == 'spam' else 0, axis=1)

print(dataset_df.head())


X_train, X_test, Y_train, Y_test = train_test_split(dataset_df[['text']],
                                                    dataset_df[['category']],
                                                    test_size=0.2,
                                                    random_state=0
                                                    )

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# (4457, 1) (1115, 1) (4457, 1) (1115, 1)
print(X_train.head())
print(Y_train.head())




max_len = 100  # 1メッセージの最大単語数 (不足分はパディング)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['text'])
print("与えられた文章の数 : ", tokenizer.document_count)
#print(tokenizer.word_index)


x_train = tokenizer.texts_to_sequences(X_train['text'])
x_test = tokenizer.texts_to_sequences(X_test['text'])

for text, vector in zip(X_train['text'].head(3), x_train[0:3]):
    print(text)
    print(vector)
# No I'm good for the movie, is it ok if I leave in an hourish?
# [38, 32, 56, 12, 5, 636, 9, 14, 47, 36, 1, 208, 8, 128, 3810]
# If you were/are free i can give. Otherwise nalla adi entey nattil kittum
# [36, 3, 204, 21, 51, 1, 29, 138, 949, 2527, 3811, 3812, 3813, 3814]
# Have you emigrated or something? Ok maybe 5.30 was a bit hopeful...
# [17, 3, 3815, 26, 185, 47, 404, 209, 740, 62, 4, 299, 3816]

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(x_train[0])


print(len(tokenizer.word_index))
vocabulary_size = len(tokenizer.word_index) + 1  # 学習データの語彙数+1

model = Sequential()

model.add(Embedding(input_dim=vocabulary_size, output_dim=32))#共分散行列みたいなもの
model.add(LSTM(16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

y_train = Y_train['category'].values
y_test = Y_test['category'].values

# 学習
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_test, y_test)
                    )