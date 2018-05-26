# -*- coding: utf-8 -*-
import pandas
import nltk
from nltk.corpus import stopwords
import string
from keras.models import Sequential
from keras.layers import Dense



data = pandas.read_csv("spam.csv",encoding="latin-1")
bag_of_words = set([])
stop = list(string.punctuation ) + stopwords.words('english')
dataword = [[word for word in nltk.word_tokenize(sentence) if word not in stop] for sentence in data.v2]
bag_of_words = set([])
for sentence in dataword:
	for word in sentence:
		bag_of_words.add(word)
print ("words count: %d" %len(bag_of_words))
bag_of_words = list(bag_of_words)
			
print ("read_data_finish")



def convertColumnToY(data):
	return [[0, 1] if x=='ham' else [1 ,0] for x in data ]

def convertColumnToX(bag_of_words,data):
	print ("Convert X data")
	
	
	X = []
	count = 0
	for sentence in data:
		vectorize = []
		print ('sentence: ' + str(count))
		for index_bag_of_words in range(len(bag_of_words)):
			if bag_of_words[index_bag_of_words] in sentence:
				
				vectorize.append(1)
			else:
				vectorize.append(0)
		X.append(vectorize)
		
		count = count + 1
	return X

train_data = data[:4000]
test_data = data[4000:]

	
X_train= convertColumnToX(bag_of_words,train_data.v2)
y_train = convertColumnToY(train_data.v1)
X_test= convertColumnToX(bag_of_words,test_data.v2)
y_test = convertColumnToY(test_data.v1)

model = Sequential()
model.add(Dense(units=en(bag_of_words), activation='relu', input_dim=100))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(x_train, y_train, epochs=5, batch_size=32)

classes = model.predict(X_test, batch_size=128)
accuracy = 0
for index in len(X_test):
	
	if classes[index] == y_test:
		accuracy +=1
print ("Accuracy score: %2f" %(accuracy/len(X_test)))