import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
import csv
data = pandas.read_csv("spam.csv",encoding="latin-1")
train_data = data[:4000]
test_data = data[4000:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer();
# train
vectorizer_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorizer_text, train_data.v1)

#metrics
vectorizer_text_test = vectorizer.transform(test_data.v2)
predict = classifier.predict(vectorizer_text_test)

score = accuracy_score(test_data.v1, predict,True)
print("accuracy_score: " + str(score))

csv_arr = []
for index, row in test_data.iterrows():
	answer = row[0]
	text = row[1]
	vectorizer_text = vectorizer.transform([text])
	predict = classifier.predict(vectorizer_text)[0]
	if (predict == answer):
		result = True
	else:
		result = False
	csv_arr.append([len(csv_arr), text, answer, predict, result])

with open('test_score.csv', 'w', newline='',encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

    for row in csv_arr:
        
        spamwriter.writerow(row)
	
    

		
