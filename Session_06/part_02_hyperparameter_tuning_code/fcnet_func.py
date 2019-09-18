import numpy as np

np.random.seed(456)
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

tf.set_random_seed(456)
from sklearn.metrics import accuracy_score


def eval_adult_hyperparams(n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100):
    print("---------------------------------------------")
    print("Model hyperparameters")
    print("n_hidden = %d" % n_hidden)
    print("n_layers = %d" % n_layers)
    print("learning_rate = %f" % learning_rate)
    print("n_epochs = %d" % n_epochs)
    print("batch_size = %d" % batch_size)
    print("dropout_prob = %f" % dropout_prob)
    print("---------------------------------------------")

    d = 1024
    graph = tf.Graph()
    with graph.as_default():

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
        adult_features = ['age', 'sex', 'education_num', 'hours_per_week', 'native_american', 'white', 'black', 'single',
                          'married',
                          'separated', 'divorced', 'widowed', 'high_degree', 'capital_gain', 'capital_loss', 'income']
        adult_df = adult_df[adult_features]
        adult_df.head()

        X = adult_df.drop(['income'], axis=1).values
        y = adult_df['income'].values.reshape(-1, 1)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)

        n_features = train_X.shape[1]
        n_output = 1
        # Generate tensorflow graph
        with tf.name_scope('placeholders'):
            X = tf.placeholder(shape=(None, n_features), dtype=tf.float32)
            y = tf.placeholder(shape=(None, n_output), dtype=tf.float32)
            keep_prob = tf.placeholder(tf.float32)
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((n_features, n_hidden)))
                b = tf.Variable(tf.zeros((n_hidden,)))
                X = tf.nn.relu(tf.matmul(X, W) + b)
                X = tf.nn.dropout(X, keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, n_output)))
            b = tf.Variable(tf.zeros((n_output,)))
            y_logit = tf.matmul(X, W) + b
            y_one_prob = tf.sigmoid(y_logit)
            y_pred = tf.round(y_one_prob)

        with tf.name_scope("loss"):
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
            l = tf.reduce_sum(entropy)

        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()

        hyperparam_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d" % (d, n_hidden, learning_rate, n_epochs, batch_size,)
        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str,
                                             tf.get_default_graph())
        N = train_X.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos + batch_size]
                    batch_y = train_y[pos:pos + batch_size]
                    feed_dict = {X: batch_X, y: batch_y, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                    print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                    train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

            # Make Predictions (set keep_prob to 1.0 for predictions)
            valid_y_pred = sess.run(y_pred, feed_dict={X: valid_X, keep_prob: 1.0})

        score = accuracy_score(valid_y, valid_y_pred)
        print("Valid Classification Accuracy: %f" % score)
    return score


if __name__ == "__main__":
    score = eval_adult_hyperparams()
