import pandas as pd
import numpy as np
# from statistics import *
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor
import time
import ray
from ray.train.sklearn import SklearnTrainer,SklearnPredictor,SklearnCheckpoint
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import roc_curve, auc

ray.init()
def retrieve__data():
    df = pd.read_excel("pistachio_data.xls", index_col=0)
    sns.countplot(data=df,y="Class")
    plt.show()
    y = df["Class"].values
    df.drop("Class", axis=1, inplace=True)
    return df, y


def retireve_wbc_data():
    df = pd.read_csv('data.csv')
    df.drop("Unnamed: 32", axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    sns.countplot(data=df,y="diagnosis")
    plt.show()

    y = df["diagnosis"].values
    df.drop('diagnosis', axis=1, inplace=True)
    return df, y



def retrieve_data(cat=False):
    # Change these and bounds in pso_custom to change dataset
    df, y = retireve_wbc_data()
#    df,y = retrieve__data()

    le = LabelEncoder()
    if cat:
        # print(df)
        X = df.values
        s = StandardScaler()

        X = s.fit_transform(X)

        y = le.fit_transform(y)

        return X, y


def select_features(size=10):
    X, y = retrieve_data(cat=True)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    fs = SelectKBest(score_func=f_regression, k=size)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    return X_selected, y


def xgbProbability(X, y, test, vals, tune=True):
    """
    4 parameters to tune
    """
    booster = ['gbtree', 'gblinear']
    base_score = [0.25, 0.5, 0.75, 0.9]
    n_estimators = [100, 500, 900, 1100, 1500]
    learning_rate = [0.05, 0.1, 0.15, 0.20]
    if tune:
        XGB = XGBClassifier(booster=booster[vals[0]], base_score=base_score[vals[1]],
                            n_estimators=n_estimators[vals[2]],
                            learning_rate=learning_rate[vals[3]],n_jobs=-1)
    else:
        XGB = XGBClassifier()

    XGB.fit(X,y)

    predictor = SklearnPredictor(estimator=XGB)
    XGB_pred = predictor.predict(test)
    return XGB_pred


def knnProbability(X, y, test, vals, tune=True):
    """
    3 parameters to tune
    """
    leaf_size = list(range(1, 50, 1))
    n_neighbors = list(range(1, 30, 1))
    p = [1, 2]
    ##### KNN
    if tune:
        KNN = KNeighborsClassifier(leaf_size=leaf_size[vals[0]], n_neighbors=n_neighbors[vals[1]], p=p[vals[2]],
                                   n_jobs=-1)
    else:
        KNN = KNeighborsClassifier()

    KNN.fit(X, y)
    KNN_pred = KNN.predict(test)
    return KNN_pred


def rfProbability(X, y, test, vals, tune=True):
    """
    7 parameters to tune
    """
    n_estimators = [100, 200, 300, 400, 500, 750, 950, 1100]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    lr = [0.0000000000001,0.0000000001,0.000001,0.0000000000001,0.0000000001,0.000001,0.00001,0.0001, 0.001, 0.01, 0.1, 1.0]
    if tune:
        ##### Random Forest
        RandomForest = RandomForestClassifier(n_estimators=n_estimators[vals[0]], max_features=max_features[vals[1]],
                                              max_depth=max_depth[vals[2]],
                                              min_samples_split=min_samples_split[vals[3]],
                                              min_samples_leaf=min_samples_leaf[vals[4]], bootstrap=bootstrap[vals[5]],n_jobs=-1)
    else:
        RandomForest = RandomForestClassifier()
        # RandomForest= AdaBoostClassifier(n_estimators=n_estimators[vals[0]],learning_rate=lr[vals[2]])
    RandomForest.fit(X, y)

    # RandomForest_pred = RandomForest.predict(test)
    RandomForest_pred = RandomForest.predict(test)
    return RandomForest_pred


@ignore_warnings(category=(ConvergenceWarning, UserWarning))
def lsvcProbability(X, y, test, vals, tune=True):
    """
    3 parameters to tune?
    # """
    learning_rates = [1e-15, 1e-8, 1e-4, 1e-2, 1e-1, 1]
    penalties = ["l2", "l2"]
    max_iters = [1000, 5000, 10000, 20000, 50000, 150000]
    #
    # param_grid = {'dual': [True,False], 'penalty': penalties, 'loss': penalties, 'C': [10. ** np.arange(-3, 3)]}


    if tune:
        # print(learning_rates[vals[0]],penalties[vals[1]],max_iters[vals[2]])
        with ignore_warnings(category=(ConvergenceWarning, FitFailedWarning)):
            Linear = LinearSVC(dual=False, C=learning_rates[vals[0]], penalty=penalties[vals[1]],
                               max_iter=max_iters[vals[2]])

    else:
        Linear = LinearSVC()
    Linear.fit(X,y)
    Linear_pred = Linear.predict(test)
    # Linear_pred = Linear.predict(test)

    return Linear_pred


def decodeValues(values):
    xgb_vals = values[5:9]
    knn_vals = values[9:12]
    rf_vals = values[12:18]
    lsvc_vals = values[18:]
    # print(xgb_vals,knn_vals,rf_vals,lsvc_vals)

    xgb_vals = [round(x * 10) for x in xgb_vals]
    knn_vals = [round(x * 10) for x in knn_vals]
    rf_vals = [round(x * 10) for x in rf_vals]
    lsvc_vals = [round(x * 10) for x in lsvc_vals]
    return xgb_vals, knn_vals, rf_vals, lsvc_vals


def sum_to_one(vector):
    sum = 0
    for i in range(len(vector)):
        sum += vector[i]
    for i in range(len(vector)):
        vector[i] /= sum
    return vector

def feature_importance(X,preds):
    from sklearn.inspection import partial_dependence
    from sklearn.dummy import DummyClassifier
    import matplotlib.pyplot as plt



    prediction_probability = preds
    dummy_model = DummyClassifier(strategy='constant', constant=prediction_probability)

    # Calculate partial dependence for a single feature
    feature_index = 0

    pdp, (x_axis,) = partial_dependence(dummy_model, X, feature_index)

    # Plot the partial dependence plot
    plt.plot(x_axis, pdp)
    plt.title(f'Partial dependence of {0}')
    # plt.xlabel(feature_name)
    plt.ylabel('Partial dependence')
    plt.show()


def initialise_models(vals):
    xgb_params, knn_params, rf_params, svc_params = decodeValues(vals)
    # with open('model_params_pistachio_2.csv', 'a') as fd:
    #     paramstr = ""
    #     for param in vals:
    #         paramstr+=","+str(param)
    #
    #     fd.write(str(paramstr) + "\n")
    #     fd.close()

    num_features = round(vals[4] * 100)
    # num_features = 27
    current_acc = 0
    current_precision = 0
    current_f1 = 0
    current_recall = 0
    print(str(num_features) + " Features.")
    splits = 2
    weights = sum_to_one(vals[0:4])
    X, y = select_features(num_features)
    kfold = StratifiedKFold(n_splits=splits)

    print(weights[0:4])
    start_time = time.perf_counter()
    tuneval = True

    for train_idx, test_idx in kfold.split(X, y):
        # pool = ThreadPoolExecutor(max_workers=4)

        # print("here")
        # XGB_pred = pool.submit(xgbProbability, X[train_idx], y[train_idx],X[test_idx],xgb_params,tune=tuneval).result()
        # KNN_pred = pool.submit(knnProbability, X[train_idx], y[train_idx],X[test_idx],knn_params,tune=tuneval).result()
        # RandomForest_pred = pool.submit(rfProbability, X[train_idx], y[train_idx],X[test_idx],rf_params,tune=tuneval).result()
        # Linear_pred = pool.submit(lsvcProbability, X[train_idx], y[train_idx], X[test_idx], svc_params, tune=tuneval).result()

        XGB_pred = xgbProbability(X[train_idx], y[train_idx], X[test_idx], xgb_params, tune=tuneval)
        # print("Decision Tree: " + str(metrics.accuracy_score(y[test_idx], DecisionTree_pred)))

        KNN_pred = knnProbability(X[train_idx], y[train_idx], X[test_idx], knn_params, tune=tuneval)
        # print("KNN: " + str(metrics.accuracy_score(y[test_idx], KNN_pred)))

        RandomForest_pred = rfProbability(X[train_idx], y[train_idx], X[test_idx], rf_params, tune=tuneval)
        # print("RandomForest: " + str(metrics.accuracy_score(y[test_idx], RandomForest_pred)))

        Linear_pred = lsvcProbability(X[train_idx], y[train_idx], X[test_idx], svc_params, tune=tuneval)

        # print(Linear_pred)

        #
        # print("XGB: " + str(metrics.accuracy_score(y[test_idx], XGB_pred)))


        weighted_average = (weights[0] * Linear_pred +
                            weights[1] * XGB_pred +
                            weights[2] * RandomForest_pred +
                            weights[3] * KNN_pred)



        y_pred = np.where(weighted_average > 0.5, 1, 0)
        fpr, tpr, thresholds = roc_curve(y[test_idx], weighted_average)

        # The histogram of scores compared to true labels
        fig_hist = px.histogram(
            x=weighted_average, color=y[test_idx], nbins=50,
            labels=dict(color='True Labels', x='Score')
        )

        fig_hist.show()

        # Evaluating model performance at various thresholds
        df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
        df.index.name = "Thresholds"
        df.columns.name = "Rate"

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        fig.show()
        # feature_importance(X[train_idx],y_pred)

        current_acc += (metrics.accuracy_score(y[test_idx], y_pred))
        current_recall += (metrics.recall_score(y[test_idx], y_pred))
        current_precision += (metrics.precision_score(y[test_idx], y_pred))
        current_f1 += (metrics.f1_score(y[test_idx], y_pred))
    current_acc = current_acc / splits
    current_recall = current_recall / splits
    current_precision = current_precision / splits
    current_f1 = current_f1 / splits

    # with open('pistachio_final.csv', 'a') as fd:
    #     fd.write(str(current_acc)+"\n")
    #     fd.close()

    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(str(current_acc) + "," + str(current_recall) + "," + str(current_f1) + "," + str(current_precision))
    # print(current_acc)

    return current_acc


#


EWO = [0.5722244,  0.19318053, 0.1405872 , 0.09400787 ,0.27     ]
#
wbc = [0.47217541685222847, 0.2193213728084335, 0.16326995470293024, 0.14523325563640768, 0.29, 0.0, 0.3, 0.09281399830047762, 0.17653993406147728, 1.9033391703053542, 2.726966217265016, 0.0, 0.16763269020425653, 0.03833469680266502, 0.9, 0.0, 0.2, 0.06409550264614769, 0.4093653266323241, 0.1, 0.0]
pistachio = [0.39689734 ,0.3093049 , 0.21483259, 0.07896518, 0.20699932, 0.06166435,
 0.17092975, 0.11055709, 0.12082741, 2.41478008, 0.83528844, 0.09106312,
 0.  ,       0.01256907, 0.67770853, 0.     ,    0.05593953, 0.04833007,
 0.35015515 ,0.00951135 ,0.33041401]
# initialise_models(wbc)



