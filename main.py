def prepare_data(data):
    x = data.X
    y = data.Y
    out_data = pd.concat([x, y], axis=1)

    district_enc = LabelEncoder()
    out_data["PdDistrict"]= district_enc.fit_transform(data["PdDistrict"])
    
    out_data['WeekOfYear'] = data['Dates'].dt.weekofyear
    
    day_of_week_enc = LabelEncoder()
    out_data["DayOfWeek"] = day_of_week_enc.fit_transform(data["DayOfWeek"])
    
    out_data["Month"] = data.Dates.dt.month
    out_data["Hour"]  = data.Dates.dt.hour
    
    out_data["Year"] = data.Dates.dt.year - 2003
    
    out_data['StreetNo'] = data['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    out_data['Blocked'] = data['Address'].apply(lambda x: 1 if "Block" in x else 0)
    out_data['StreetCorner'] = data['Address'].apply(lambda x : 1 if '/' in x else 0)


    return out_data

def estimate_solution(data, labels):
    from sklearn import cross_validation
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
output_result_csv(prediction, le_crime.classes_ , 'D:\\temp\\testResult.csv')
