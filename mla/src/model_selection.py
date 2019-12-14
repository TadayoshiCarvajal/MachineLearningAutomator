import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report
from random import randint
from time import time
import warnings
from utility import get_time_string

warnings.filterwarnings('ignore')

def get_promising_models(feature_selected_data, pipeline_automator):
    """
    Automates the process of selecting a good classifier, hyperparameter tuning for the classifier
    and training and testing the model. This process is very slow so we start by limiting the data
    that we want the models to be evaluated on. The current version takes 15 minutes or so to 
    run on using 1500 records which is acceptable for testing purposes all things considered (training dozens of
    classifiers on 1500 records and hundreds of features)
    """    
    print('\n\nModel Selection...')
    start = time()

    random_state = pipeline_automator.parameters['random_state']
    np.random.seed(random_state)

    n_records = feature_selected_data.shape[0]
    n_features = feature_selected_data.shape[1] - 1 # n_features = n_cols - 1, (last column is the target column)
    split_type = pipeline_automator.parameters['data_splitting']

    """
    1) Split our data up into development aka training set for the model selection and tuning data, and
    a validation set to see how well our models can classify never before seen examples after we tune them.
    We want to use a lot of data for developing good models, but we also want to set aside
    enough data to test how well our models can generalize to unseen examples. """

    print('Splitting Our Data into training and testing sets...')
    testing_set_size = pipeline_automator.parameters['testing_set_size']
    train_data, test_data = split_data(feature_selected_data, test_size=testing_set_size, type_=split_type, random_state=random_state)

    n_examples_in_training_set = len(train_data)

    pipeline_automator.metadata['training_data'] = train_data
    pipeline_automator.metadata['training_data_size'] = n_examples_in_training_set
    pipeline_automator.metadata['testing_data'] = test_data
    pipeline_automator.metadata['testing_data_size'] = len(test_data)
    print('total examples:', 
        len(feature_selected_data), 
        '\ttraining examples:',  
        len(train_data), 
        '\ttesting examples:', 
        len(test_data),end='\n\n')

    """
    2) Enforce the Model Selection Data Limit. Model selection is very slow. We presently have 13 classifier types.
    Each classifier is cloned cv_folds number of times and then trained on the training data. For cv=10, that's 130
    classifiers being trained and tested. If we trained all of those models on the entire data set the computational
    complexity grows in a combinatorial way, i.e., will not finish in a reasonable amount time. Therefore, we get around
    this by using a subset if we have more records than the model_selection_data_limit. If we do not have more than that many
    records in the training set, then we use the entire training set to select our models.
    """
    model_selection_data_limit = pipeline_automator.parameters['model_selection_data_limit']

    if model_selection_data_limit < n_examples_in_training_set :
        print('Splitting our data because it exceeds the model_selection_data_limit parameter...')
        rest_of_the_data_set, evaluation_set = split_data(feature_selected_data, test_size=model_selection_data_limit, type_=split_type, random_state=random_state)
        print(len(rest_of_the_data_set),'saved for later +',len(evaluation_set),'evaluation')
    else:
        rest_of_the_data_set = []
        evaluation_set = feature_selected_data

    """
    3) Split our training data up into feature matrix X and label vector y:
    """
    X, y = evaluation_set[:,:n_features], evaluation_set[:,n_features]
    X = X.astype(np.float64)
    y = y.reshape( (y.shape[0], 1) )
    y = np.array(y.T)[0]

    """
    4) Place all our generated classifier functions above into a list of callable functions that we iterate
    through and call one at a time on the training data X and y. Then each classifier trains the data 
    and uses cross validation in order to assess how good that particular classifier is at classifying the data.
    A confusion matrix for each is computed and calculates the accuracy, precision, recall, etc..
    Depending on the needs of the classifier(maximize precision vs. recall, balanced, etc.), we select the appropriate
    rating indicator, and rate the classifiers by their performance.
    """
    # list of callables used to generate each classifier:
    cv_folds = pipeline_automator.parameters['model_selection_cv_folds_count']
    classifiers_getters =  [
        get_sgd_classifier,
        get_knn_classifier,
        get_linear_svm_classifier,
        get_polynomial_svm_classifier,
        get_rbf_svm_classifier,
        get_decision_tree_classifier,
        get_random_forest_classifier,
        get_extra_trees_classifier,
        get_adaboost_forest_classifier,
        get_mlp_classifier
    ]
    classifiers = [ get_classifier(X, y, random_state=random_state, cv_folds=cv_folds) for get_classifier in classifiers_getters ]
    classifier_dictionary = {name: clf for name, clf, results in classifiers}

    """
    5) Choose the most promising models to send to the next phase: model tuning.
    """
    # Rank the classifiers by their F1 score (for now) we can decide which metric or even automate this in
    # a pipeline hyperparameter later on.
    performance_metric = pipeline_automator.parameters['model_selection_performance_metric']
    ranked_classifiers = sorted( classifiers, key=lambda x: x[-1][performance_metric], reverse=True )

    for name, classifier, results in ranked_classifiers:
        print(f"{name:<35s}\tF1-score:{results['f1_macro']:<10f}")
    print()

    # Filter the ones with low performance.
    min_performance = pipeline_automator.parameters['model_selection_min_performance']
    print('Filtering models that do not meet the model selection minimum performance: ', performance_metric, '>=', min_performance,'...')
    ranked_classifiers = list(filter(lambda x: x[-1][performance_metric] >= min_performance, classifiers))
    ranked_classifiers = sorted( ranked_classifiers, key=lambda x: x[-1][performance_metric], reverse=True )

    if len(ranked_classifiers) > 0:
        for name, classifier, results in ranked_classifiers:
            print(f"{name:<35s}\tF1-score:{results['f1_macro']:<10f}")
        print()
    else:
        print('None of the models met the minimum performance of', performance_metric, '>=', min_performance)

    n_promising_models_to_select = min(pipeline_automator.parameters['n_promising_models_to_select'], len(ranked_classifiers))
    
    promising_models = [ (promising__clf_name, promising__classifier, promising__results) for promising__clf_name, promising__classifier, promising__results in ranked_classifiers[:n_promising_models_to_select]]

    print('Selecting up to the top',len(promising_models),'models...')

    if len(ranked_classifiers) > 0:
        for name, classifier, results in promising_models:
            print(f"{name:<35s}\tF1-score:{results['f1_macro']:<10f}")
        print()
    else:
        print('None of the models met the minimum performance of', performance_metric, '>=', min_performance)

    promising_models = {promising__clf_name:promising__classifier for promising__clf_name, promising__classifier, promising__results in promising_models}

    stop = time()
    time_elapsed = get_time_string(stop - start)
    pipeline_automator.metadata['model_selection_time'] = time_elapsed

    return promising_models

def split_data(data, test_size = 0.2, random_state = None, type_='stratified'):
    """
    A function used to split the data up in to a test set and a train set.
    There are two ways of doing this which is controlled by the type_ parameter:
    
    type == 'random': randomly splits the data. This is fine if the data
    is evenly distributed / we have enough data. This is bad if there is not 
    enough data because the test set can end up having non-representative
    amounts of each class.

    type == 'stratified': uses k-folds cross validation and a stratified splitter
    to generate k different splits of train and test data
    where each split of test data is ensure to have the same ratio
    of the classes as is found in the overall dataset. Use this when random
    there isn't enough data for 'random' mode to achieve a representative split.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit

    if type_ == 'random':
        train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_set, test_set

    elif type_ == 'stratified':
        n_features = data.shape[1] - 1 # n_features = n_cols - 1, (last column is the target column)
        X, y = data[:,:n_features].astype(np.float64), data[:,n_features]

        stratas = StratifiedShuffleSplit(test_size=test_size, random_state=random_state)
        train_sets, test_sets, ratios = [], [], []

        for train_index, test_index in stratas.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_test_counting = np.array(y_test.T)
            unique, counts = np.unique(y_test_counting, return_counts=True)

            total = sum(counts)
            label_frequencies_in_test = dict(zip(unique, counts/total))

            train_set = np.column_stack( [X_train, y_train] )
            test_set = np.column_stack( [X_test, y_test] )
            
            train_sets.append(train_set)
            test_sets.append(test_set)
            ratios.append(label_frequencies_in_test)

        print('Data maintains the following label proportions:',ratios[0])
        print(len(train_sets),'stratified splits were generated')
        return train_sets[0], test_sets[0]

def score_model(name, clf, X_train, y_train, cv_folds, print_scores=True):
    """
    Performs k-folds cross validation using the specified classifier on the training
    data. Prints the performance of the model accross the following metrics:

        accuracy : number of correct predictions / total # of predictions
        balanced accuracy: takes into account the ratio of each class and uses that to weigh the accuracy.
        precision : number of correct positive class predictions / total # positive class predictions.
        recall : number of correct positive class predictions / total # positive class records.
        f1 - score : harmonic average of precision and recall (favors models with high values for both). Imbalanced
        models get  alower f1 score.
    """

    print('\nPerforming k-folds cross validation ( k=',cv_folds,') for ', name, '...')

    y_train_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    y_train_predictions = cross_val_predict(clf, X_train, y_train, cv=cv_folds)
    cm = confusion_matrix(y_train, y_train_predictions)

    accuracy = sum(y_train_scores)/cv_folds
    balanced_accuracy = balanced_accuracy_score(y_train, y_train_predictions)
    precision = precision_score(y_train, y_train_predictions, average='macro')
    recall = recall_score(y_train, y_train_predictions, average='macro')
    f1_macro = f1_score(y_train, y_train_predictions, average='macro')
    f1_micro = f1_score(y_train, y_train_predictions, average='micro')

    results = {
        'confusion matrix' : cm, 
        'accuracy' : accuracy,
        'balanced accuracy' : balanced_accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1_macro' : f1_macro,
        'f1_micro' : f1_micro
    }
    if print_scores:
        print(cm)
        print('accuracy:', accuracy)
        print('balanced accuracy:', balanced_accuracy)
        print( classification_report(y_train, y_train_predictions) )
        print()

    return results

def get_sgd_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    A classifier that performs classification based on stochastic gradient descent.
    X - a (n_training_examples, n_features) np.matrix containing the features of all training examples.
    y - a (n_training_examples, 1) np.array containing the labels.
    """
    from sklearn import linear_model

    name = 'SGD Classifier (log)'
    clf = linear_model.SGDClassifier(
        max_iter=1000, 
        tol=1e-3,
        loss='log', 
        early_stopping=True, 
        n_jobs=-1, 
        random_state=random_state)
    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_knn_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    A classifier that performs classification based on k-nearest neighbors.
    X - a (n_training_examples, n_features) np.matrix containing the features of all training examples.
    y - a (n_training_examples, 1) np.array containing the labels.
    random_state - does nothing in KNN since the decision algorithm is deterministic.
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    name = 'KNN Classifier'
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_linear_svm_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    A binary classifier that performs classification based on support vector machine.
    X - a (n_training_examples, n_features) np.matrix containing the features of all training examples.
    y - a (n_training_examples, 1) np.array containing the labels.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier

    name = 'SGD Classifier (linear SVM)'
    clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000, random_state=random_state)
    calibrated_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    calibrated_clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, calibrated_clf, results

def get_polynomial_svm_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    A classifier that performs classification based on support vector machine.
    X - a (n_training_examples, n_features) np.matrix containing the features of all training examples.
    y - a (n_training_examples, 1) np.array containing the labels.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    name = 'Polynomial SVM Classifier'
    clf = Pipeline((
        ("scaler", StandardScaler()),
        ("poly_svc", SVC(kernel='poly', 
                        degree=2,
                        coef0=1, 
                        C=5, 
                        probability=True, 
                        gamma='auto', 
                        random_state=random_state,))
    ))

    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_rbf_svm_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    A classifier that performs classification based on support vector machine.
    X - a (n_training_examples, n_features) np.matrix containing the features of all training examples.
    y - a (n_training_examples, 1) np.array containing the labels.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    

    name = 'RBF SVM Classifier'
    clf = Pipeline((
        ("scaler", StandardScaler()),
        ("rbf_svc", SVC(kernel='rbf', C=1, probability=True, gamma='auto', random_state=random_state))
    ))

    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_decision_tree_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    An ensemble classifier system that performs classification based on it's component decision trees.
    """
    from sklearn.tree import DecisionTreeClassifier

    name = 'Decision Tree Classifier'
    clf = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_random_forest_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    An ensemble classifier system that performs classification based on it's component decision trees.
    """
    from sklearn.ensemble import RandomForestClassifier

    name = 'Random Forest Classifier'
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=random_state)
    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_extra_trees_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    An ensemble classifier system that performs classification based on it's component decision trees and adds trees when needed.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    
    name = 'Extra Trees Classifier'
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=2, n_jobs=-1, random_state=random_state)
    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results
   
def get_adaboost_forest_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    An ensemble classifier system that performs classification based on voting 
    and uses boosting to improve the performance of its component estimators.
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    
    name = 'AdaBoosting Trees Classifier'
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=random_state)

    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results

def get_mlp_classifier(X_train, y_train, random_state=None, cv_folds=5):
    """
    An ensemble classifier system that performs classification based on voting 
    and uses boosting to improve the performance of its component estimators.
    """
    from sklearn.neural_network import MLPClassifier

    
    name = 'MLP Classifier'
    clf = MLPClassifier(alpha=0.001, early_stopping=False, hidden_layer_sizes=[150]*7, random_state=random_state)

    clf.fit(X_train, y_train)

    results = score_model(name, clf, X_train, y_train, cv_folds=cv_folds)

    return name, clf, results