import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

adult_file_path = 'adult.csv'
adult_df = pd.read_csv(adult_file_path)
adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace('.', '_')
adult_df = adult_df.replace({'?': np.nan}).dropna()
adult_df.head()


adult_df.income = [1 if income == ">50K" else 0 for income in adult_df.income]
adult_df.sex = [1 if sex == "Male" else 0 for sex in adult_df.sex]
white = [1 if race == "White" else 0 for race in adult_df.race]
black = [1 if race == "Black" else 0 for race in adult_df.race]
native_american = [1 if native_country == "United-States" else 0 for native_country in adult_df.native_country]
single = [1 if marital_status == "Never-married" else 0 for marital_status in adult_df.race]
married = [1 if marital_status == "Married-civ-spouse" else 0 for marital_status in adult_df.marital_status]
separated = [1 if marital_status == "Separated" else 0 for marital_status in adult_df.marital_status]
divorced = [1 if marital_status == "Divorced" else 0 for marital_status in adult_df.marital_status]
widowed = [1 if marital_status == "Widowed" else 0 for marital_status in adult_df.marital_status]
high_degree = [1 if education in ['Masters', 'Doctorate'] else 0 for education in adult_df.education]
adult_df['white'] = white
adult_df['black'] = black
adult_df['native_american'] = native_american
adult_df['single'] = single
adult_df['married'] = married
adult_df['separated'] = separated
adult_df['divorced'] = divorced
adult_df['widowed'] = widowed
adult_df['high_degree'] = high_degree
adult_features = ['age', 'sex', 'education_num', 'hours_per_week', 'native_american', 'white', 'black', 'single', 'married',
                  'separated', 'divorced', 'widowed', 'high_degree', 'capital_gain', 'capital_loss', 'income']
adult_df = adult_df[adult_features]
adult_df.head()

X = adult_df.drop(['income'], axis=1).values
y = adult_df['income'].values
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)

x_train = X[:int(0.8*len(X))]
x_test = X[int(0.8*len(X)):int(0.9*len(X))]
x_valid = X[int(0.9*len(X)):]

rf_clf = RandomForestClassifier(n_estimators=100, random_state=1)
rf_clf.fit(train_X, train_y)
y_train_pred = rf_clf.predict(train_X)
y_valid_pred = rf_clf.predict(valid_X)
y_test_pred = rf_clf.predict(test_X)

train_accuracy_score = accuracy_score(train_y, y_train_pred)
valid_accuracy_score = accuracy_score(valid_y, y_valid_pred)
test_accuracy_score = accuracy_score(test_y, y_test_pred)

print(f"Train Classification Accuracy: {train_accuracy_score}")
print(f"Valid Classification Accuracy: {valid_accuracy_score}")
print(f"Test Classification Accuracy: {test_accuracy_score}")
