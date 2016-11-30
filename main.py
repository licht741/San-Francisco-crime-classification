# coding: utf-8

def prepare_data(data):
    days = pd.get_dummies(data.DayOfWeek)
    district = pd.get_dummies(data.PdDistrict)
    year = data.Dates.dt.year
    hour = data.Dates.dt.hour
    day = data.Dates.dt.day
    x = data.X
    y = data.Y
    out_data = pd.concat([year, day, hour, days, district, x, y], axis=1)
    return out_data

def estimate_solution(data, labels):
    from sklearn.metrics import log_loss
    [crime_train_data, crime_test_data, crime_train_labels, crime_test_labels] = cross_validation.train_test_split(data, labels, test_size=0.3)
    crime_rf = RandomForestClassifier()
    crime_rf.fit(crime_train_data, crime_train_labels)
    prediction = np.array(crime_rf.predict_proba(crime_test_data))
    return log_loss(crime_test_labels, prediction)

def output_result_csv(prediction, classes, path):
    result=pd.DataFrame(prediction, columns=classes)
    result.to_csv(path, index = True, index_label = 'Id' )


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 

train = pd.read_csv('D:\\temp\\train.csv', parse_dates = ["Dates"], index_col= False)
test = pd.read_csv('D:\\temp\\test.csv', parse_dates = ["Dates"], index_col= False)

from sklearn import preprocessing

le_crime = preprocessing.LabelEncoder()
le_crime.fit(train.Category)

train_label = le_crime.transform(train.Category)

train_data = prepare_data(train)

crime_rf = RandomForestClassifier()
crime_rf.fit(train_data, train_label)

test_data = prepare_data(test)
pred = crime_rf.predict_proba(test_data)
prediction = np.array(pred)

estimate_solution(train_data, train_label)

output_result_csv(prediction, le_crime.classes_ , 'D:\\temp\\testResult.csv')
