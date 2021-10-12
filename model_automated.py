import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def get_model():
    #getting the data
    full_data = pd.read_csv('aug_train.csv')

    #getting the columns we want
    X = full_data[['city_development_index','enrolled_university','relevent_experience','education_level','last_new_job','experience', 'target']]
    #dropping nulls from the data
    X = X.dropna()
    X.reset_index(drop=True, inplace=True)
    #getting y
    y = X['target']

    X = X[['city_development_index','enrolled_university','relevent_experience','education_level','last_new_job','experience']]
    #one hot encoding the columns that need to be
    one_hot_x = pd.get_dummies(X, dtype=float)

    #train/test splitting the data
    train_x, test_x, train_y, test_y = train_test_split(one_hot_x, y, test_size=0.30, random_state = 1)
    print("Training the model...")
    print()
    #creating and training our model
    log_clf = LogisticRegression(random_state=42).fit(train_x, train_y)
    log_predict = log_clf.predict(test_x)
    print("The model precision is:")
    precision = precision_score(test_y, log_predict)
    print(precision)
    print()
    print("The model recall is:")
    recall = recall_score(test_y, log_predict)
    print(recall)
    print()
    print("We can now begin using the model in practice. Enter the candidate info below to receive a prediction.")

    return log_clf

def get_prediction(classifier,sample):
    full_data = pd.read_csv("aug_train.csv")
    X = full_data[['city_development_index','enrolled_university','relevent_experience','education_level','last_new_job','experience']]
    X = X.append(sample, sort=False, ignore_index=True)
    one_hot_x = pd.get_dummies(X, dtype=float)
    last_row = one_hot_x.iloc[-1:]
    class_prob = classifier.predict_proba(last_row)
    prob_stay = round(class_prob[0][0],2)
    prob_leave = round(class_prob[0][1],2)
    print("We predict the prob of staying to be: ", prob_stay)
    print("And the prob of leaving to be: ", prob_leave)
    if prob_leave >= 0.8:
        print("As we are trying to optimize for precision and the prob of leaving is above 80%, we recommend pursuing this candidate")
    else:
        print("As we are trying to optimize for precision and the prob of leaving is below 80%, we do not recommend pursuing this candidate")

