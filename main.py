from typing import DefaultDict, Iterable, List, Tuple, Type
import logging
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from collections import defaultdict
from logging import debug

logging.basicConfig(level=logging.INFO)

# required to use LaTeX in labels and text when plotting
plt.rcParams['text.usetex'] = True

random_state=42


def load_data(Xy = False):
    raw = pd.read_excel('./data/Concrete_Data.xls')

    if not Xy:
        return raw
    
    raw_data = raw.to_numpy()
    X, y = raw_data[:,:8], raw_data[:,8]

    return X, y

def main():
    lasso_color = 'darkorange'
    ridge_color = 'yellowgreen'
    linear_color = 'firebrick'

    X, y = load_data(Xy = True)
    
    results = []

    for _ in range(30):
        results.append(base(X, y))

    accuracy, weights = defaultdict(list), defaultdict(list)
    alpha, poly_degree = defaultdict(list), defaultdict(list)
    # result should have formatting {'LinearRegression': dict, 'Ridge': dict, 'Lasso': dict}
    for result in results:
        for key, item in result.items():
            if key != 'LinearRegression':
                alpha[key].append(item['alpha'])
                poly_degree[key].append(item['poly_degree'])
            accuracy[key].append(item['acc'])
            # this affects the average 
            # TODO only look at linear feature weights
            weights[key].append(item['weights'][1:9])

    avg_weights = {key: np.average(weights[key], axis=0) for key in weights}

    # create relations: polydegree -> acc for ridge and lasso
    poly_acc_ridge = relate(poly_degree['Ridge'], accuracy['Ridge'])
    poly_acc_lasso = relate(poly_degree['Lasso'], accuracy['Lasso'])

    # crete relations: alpha -> acc for ridge and lasso
    alpha_acc_ridge = relate(alpha['Ridge'], accuracy['Ridge'])
    alpha_acc_lasso = relate(alpha['Lasso'], accuracy['Lasso'])

    # average the R^2 scores over all relations:
    for key, val in poly_acc_ridge.items():
        poly_acc_ridge[key] = [np.average(val)]
    for key, val in poly_acc_lasso.items():
        poly_acc_lasso[key] = [np.average(val)]
    for key, val in alpha_acc_lasso.items():
        alpha_acc_lasso[key] = [np.average(val)]
    for key, val in alpha_acc_ridge.items():
        alpha_acc_ridge[key] = [np.average(val)]

    # lower dimension of values to 1
    poly_acc_ridge = {k: v[0] for k, v in poly_acc_ridge.items()}
    poly_acc_lasso = {k: v[0] for k, v in poly_acc_lasso.items()}
    alpha_acc_ridge = {k: v[0] for k, v in alpha_acc_ridge.items()}
    alpha_acc_lasso = {k: v[0] for k, v in alpha_acc_lasso.items()}
    print(alpha_acc_lasso)
    
    print(f'Linear Regression average R^2: {np.average(accuracy["LinearRegression"])}')
    print(f'Linear Regression variance of R^2: {np.var(accuracy["LinearRegression"])}')
    print(f'Ridge Regression average R^2: {np.average(accuracy["Ridge"])}')
    print(f'Ridge Regression variance R^2: {np.var(accuracy["Ridge"])}')
    print(f'Lasso Regression average R^2: {np.average(accuracy["Lasso"])}')
    print(f'Lasso Regression variance of R^2: {np.var(accuracy["Lasso"])}')

    _, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title('Ridge Regression')
    axes[0].set_xlabel('Polydegree')
    axes[0].set_ylabel(r'$R^2$')
    axes[0].scatter(poly_degree['Ridge'], accuracy['Ridge'])
    axes[0].xaxis.set_ticks(np.arange(min(poly_degree['Ridge']), max(poly_degree['Ridge'])+1, 1))

    axes[1].set_title('Ridge Regression')
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel(r'$R^2$')
    axes[1].scatter(alpha['Ridge'], accuracy['Ridge'])

    _, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title('Lasso Regression')
    axes[0].set_xlabel('Polydegree')
    axes[0].set_ylabel(r'$R^2$')
    axes[0].scatter(poly_degree['Lasso'], accuracy['Lasso'])
    axes[0].xaxis.set_ticks(np.arange(min(poly_degree['Lasso']), max(poly_degree['Lasso'])+1, 1))

    axes[1].set_title('Lasso Regression')
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel(r'$R^2$')
    axes[1].scatter(alpha['Lasso'], accuracy['Lasso'])

    _, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title(r'Average $R^2$ over Polynomial Degree')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel(r'Average $R^2$')
    axes[0].plot(*zip(*sorted(poly_acc_lasso.items(), key=lambda x: x[0])), color=lasso_color, label='Lasso Regression')
    axes[0].plot(*zip(*sorted(poly_acc_ridge.items(), key=lambda x: x[0])), color=ridge_color, label='Ridge Regression')
    axes[0].xaxis.set_ticks(np.arange(min(min(poly_degree['Lasso']), min(poly_degree['Ridge'])), 
                                   max(max(poly_degree['Lasso'])+1, max(poly_degree['Ridge'])+1)))
    axes[0].legend()

    axes[1].set_title(r'Average $R^2$ over Alpha')
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel(r'Average $R^2$')
    axes[1].plot(*zip(*sorted(alpha_acc_lasso.items(), key=lambda x: x[0])), color=lasso_color, label='Lasso Regression')
    axes[1].plot(*zip(*sorted(alpha_acc_ridge.items(), key=lambda x: x[0])), color=ridge_color, label='Ridge Regression')
    axes[1].legend()

    _, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title('Alpha over Iterations')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel(r'$\alpha$')
    axes[0].plot(np.arange(1, 31), alpha['Lasso'], color=lasso_color, label='Lasso Regression')
    axes[0].plot(np.arange(1, 31), alpha['Ridge'], color=ridge_color, label='Ridge Regression')
    axes[0].legend()

    axes[1].set_title('Polynomial Degree over Iterations')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Polynomial Degree')
    axes[1].plot(np.arange(1, 31), poly_degree['Lasso'], color=lasso_color, label='Lasso Regression')
    axes[1].plot(np.arange(1, 31), poly_degree['Ridge'], color=ridge_color, label='Ridge Regression')
    axes[1].legend()

    _, ax = plt.subplots(constrained_layout=True)
    ax.set_title('Average weights of Features')
    labels = [x.replace('^', '\\^') for x in load_data().columns[:8]]
    l = np.arange(len(labels))
    width = 0.2
    r1 = ax.barh(l - width/2, avg_weights['LinearRegression'], width, color=linear_color, label='Linear Regression')
    r2 = ax.barh(l - 3*width/2, avg_weights['Ridge'], width, color=ridge_color, label='Ridge Regression')
    r3 = ax.barh(l + width/2, avg_weights['Lasso'], width, color=lasso_color, label='Lasso Regression')
        
    ax.set_yticks(l, labels)
    ax.bar_label(r1, padding=3)
    ax.bar_label(r2, padding=3)
    ax.bar_label(r3, padding=3)
    ax.legend()

    plt.show()

def describe_data():
    raw = load_data()
    print('data looks like')
    # gain information about the data
    series = raw.describe(include='all')
    series.to_csv('./data/description.csv')
    print(raw.columns)
    print(raw.head())


def relate(x: Iterable, y: Iterable):
    """ forms a 1:1 correspondance between x and y 
        
        returns:
            f: x -> y
    """
    d = defaultdict(list)
    for v,w in zip(x,y):
        d[v].append(w)

    return d

def train_model(model: LinearRegression | Lasso | Ridge, X_train: np.ndarray, y_train: np.ndarray, poly_degree: int = 1):
    """ Trains and returns a model in a Pipeline for the given input parameters. The pipeline handles data transformations all on its own.
        
        params: 
            `model`: model to be trained
            `X_train`: training data 
            `y_train`: training labels 
            `poly_degree`: degree of polynomial to be used. Defaults to 1.

        returns: 
            sklearn.pipeline.Pipeline with layers: MinmaxScaler, PolynomialFeatures and 
            given regression model `model`
    """
    pipe = Pipeline([('scaler', MinMaxScaler()), ('polytransform', PolynomialFeatures(poly_degree)), 
                     ('model', model)])
    pipe.fit(X_train, y_train)
    return pipe
    

def base(X: np.ndarray, y: np.ndarray):
    """ This is the base experiment as described on 'Übungsblatt 06' 
        
        params: 
            X: data 
            y: labels

        returns:
            Dict['LinearRegression': Dict['acc': float], 
                 'Ridge': Dict['acc': float, 'alpha': float, 'poly_degree': int],
                 'Lasso': Dict['acc': float, 'alpha': float, 'poly_degree': int]]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    linear = train_model(LinearRegression(), X_train, y_train)
    linear_acc = linear.score(X_test, y_test)

    _, (ridge_alpha, ridge_polydegree) = cross_validation(10, Ridge, X_train, y_train, alpha_range=np.array([0.05, 0.1, 0.5, 2, 5, 10, 25]))
    ridge = train_model(Ridge(ridge_alpha), X_train, y_train, ridge_polydegree)
    ridge_acc = ridge.score(X_test, y_test)

    _, (lasso_alpha, lasso_polydegree) = cross_validation(10, Lasso, X_train, y_train, alpha_range=np.array([0.05, 1, 0.1, 0.5, 1, 2, 5, 10, 25]))
    lasso = train_model(Lasso(lasso_alpha), X_train, y_train, lasso_polydegree)
    lasso_acc = lasso.score(X_test, y_test)

    return {'LinearRegression': {'acc': linear_acc, 'weights': linear.get_params()['model'].coef_}, 
            'Ridge': {'acc': ridge_acc, 'alpha': ridge_alpha, 
                      'poly_degree': ridge_polydegree, 'weights': ridge.get_params()['model'].coef_},
            'Lasso': {'acc': lasso_acc, 'alpha': lasso_alpha, 
                      'poly_degree': lasso_polydegree, 'weights': lasso.get_params()['model'].coef_}} 


def cross_validation(n: int, model: Type[Ridge] | Type[Lasso], 
                     X: np.ndarray, y:np.ndarray, degree_range: np.ndarray | list | range = np.arange(1, 6), 
                     alpha_range: np.ndarray | list | range = np.array([0.05, 0.1, 0.5, 2, 5, 10, 25])) -> Tuple[float, Tuple[float, int]]:
    """ Executes kfold-crossvalidation for a given n and a model class. Performs polynomial regression with a specific model. 
        
        params: 
            n: number of folds
            model: Model Class | either sklearn.linear_model.Ridge 
                or sklearn.linear_model.Lasso
            X: training features
            y: training labels
            degree_range: range, list or numpy.ndarray of 
                polynomial degrees to choose from 
            alpha_range: range, list or numpy.ndarray of 
                alphas to choose from

        returns: 
            Tuple(train_accuracy: float, (alpha: float, poly_degree: int))
    """
    kf = KFold(n, random_state=random_state)
    best = (0, (1, 1))
    for train, test in kf.split(X):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        a = np.random.RandomState(random_state).permutation(alpha_range)[0]
        d = np.random.RandomState(random_state).permutation(degree_range)[0]
        # for debugging purposes only
        debug(f'CV main.py: Training model {model.__name__} with alpha {a} and degree {d}.')
        m = model(alpha=a)
        pipe = train_model(m, X_train, y_train, d)
        score = pipe.score(X_test, y_test)
        if score > best[0]:
            best = (score, (a, d))

    return best


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        main()
    if sys.argv[1] == 'data':
        describe_data()
