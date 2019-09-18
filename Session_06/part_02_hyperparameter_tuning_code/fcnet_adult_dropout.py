# # Data set description
# **Data income, classsification problem**
# The original data has attributes:  
# income (>50K, <=50K),  age( continuous),  workclass:( Private, Self-emp-not-inc...), fnlwgt: continuous, education (Bachelors,...), education-num (continuous), 
# marital-status (Married-civ-spouse...), occupation (Tech-support,..), relationship (Wife,...), race (White,...), sex (Female, Male) capital-gain (continuous) capital-loss (continuous), hours-per-week  (continuous) and native-country.
# 
# I removed attributes that contain just minor categories. I kept attributes that have larrge categories,  for example for race white and black are large categories and for native-country United States is the main caegory.
# 
# So my final attributes are:
# income, age, education-num, marital-status, sex, capital-gain, capital-loss, hours per week, native country.
# Here I cleaned the data set everything so it has just numerical variables.
# 
# https://archive.ics.uci.edu/ml/datasets/Adult 


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

adult_file_path = 'adult.csv'
adult_df = pd.read_csv('adult.csv')
adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace('.', '_')
adult_df = adult_df.replace({'?': np.nan}).dropna()
adult_df.head()

# Preprocessing
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
y = adult_df['income'].values.reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)

n_features = train_X.shape[1]
m_train = train_X.shape[0]
n_output = 1
n_hidden = 64
learning_rate = 0.001
n_epochs = 65
batch_size = 128
droup_out = 0.5

with tf.name_scope('placeholders'):
    X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
    y = tf.placeholder(shape=(None, n_output), dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden_layer'):
    W = tf.Variable(tf.random_normal((n_features, n_hidden)))
    b = tf.Variable(tf.zeros((n_hidden,)))
    X_hidden = tf.nn.relu(tf.matmul(X, W) + b)
    # Apply dropout
    X_hidden = tf.nn.dropout(X_hidden, keep_prob=keep_prob)

with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, n_output)))
    b = tf.Variable(tf.zeros((n_output,)))
    y_logit = tf.matmul(X_hidden, W) + b
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding p(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    # Sum all contributions
    l = tf.reduce_sum(entropy)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./nn_train', tf.get_default_graph())

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(1, n_epochs + 1):
        batch_pos = 0
        pos = 0
        while pos < m_train:
            X_batch = train_X[pos: pos + batch_size]
            y_batch = train_y[pos: pos + batch_size]
            feed_dict = {X: X_batch, y: y_batch, keep_prob: 0.5}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print(f"epoch {epoch}, step {step}, loss: {loss}")
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size
    # Make Predictions (set keep_pprob to 1.0 for predictions)
    y_train_pred = sess.run(y_pred, feed_dict={X: train_X, keep_prob: 1.0})
    y_valid_pred = sess.run(y_pred, feed_dict={X: valid_X, keep_prob: 1.0})
    y_test_pred = sess.run(y_pred, feed_dict={X: test_X, keep_prob: 1.0})

train_accuracy_score = accuracy_score(train_y, y_train_pred)
valid_accuracy_score = accuracy_score(valid_y, y_valid_pred)
test_accuracy_score = accuracy_score(test_y, y_test_pred)

print(f"Train Classification Accuracy: {train_accuracy_score}")
print(f"Valid Classification Accuracy: {valid_accuracy_score}")
print(f"Test Classification Accuracy: {test_accuracy_score}")
