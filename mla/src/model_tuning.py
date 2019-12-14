import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from time import time
from tqdm import tqdm
from utility import get_time_string

def tune_models(promising_models, pipeline_automator):
    print('\n\nModel Tuning...')
    start = time()
    # Take our dict of promising models and perform randomized search
    training_data = pipeline_automator.metadata['training_data']
    n_features = training_data.shape[1] - 1 # last column is the target column

    n_hyperparam_combos = pipeline_automator.parameters['n_hyperparam_combos']
    model_tuning_cv_folds_count = pipeline_automator.parameters['model_tuning_cv_folds_count']
    include_ensemble_voters = pipeline_automator.parameters['include_ensemble_voters']
    model_tuning_scoring = pipeline_automator.parameters['model_tuning_scoring']

    random_state = pipeline_automator.parameters['random_state']
    np.random.seed(random_state)

    X, y = training_data[:,:n_features], training_data[:,n_features]
    X = X.astype(np.float64)
    y = y.reshape( (y.shape[0], 1) )
    y = np.array(y.T)[0]

    tuned_models = {}
    ignored_models =    {
                            'Gaussian Process Classifier' : "This model doesn't need hyperparameter tuning because it's 'hyperparameters' were optimized at creation.",
                            'Naive Bayes Classifier' : "We have not determined if there are hyperparameters to tune for this model.",
                            'Discriminant Analysis Classifier' : "We have not determined if there are hyperparameters to tune for this model.",
                            'RBF SVM Classifier' : "The model autotunes gamma.",
                            'Polynomial SVM Classifier' : 'Tuned already'
                        }

    for model_name in tqdm(promising_models, 'Tuning Promising Models...'):
        model = promising_models[model_name]
        if model_name not in ignored_models:
            print( 'Beginning Tuning Process for ', model_name,'...' )
            param_distribution = get_parameter_distribution(model_name, random_state=random_state, n_iter=n_hyperparam_combos)
            random_search = RandomizedSearchCV(model, 
                            param_distribution, 
                            n_iter=n_hyperparam_combos, 
                            cv=model_tuning_cv_folds_count, 
                            scoring=model_tuning_scoring,
                            refit=True, 
                            n_jobs=-1, 
                            random_state=random_state)
            random_search.fit(X, y)
            clf = random_search.best_estimator_
            tuned_models[model_name] = clf
        else:
            print('Skipped tuning of',model_name,'for the following reason:', ignored_models[model_name])
            tuned_models[model_name] = model
            model.fit(X, y)
            clf = model
        print('Best Tuning:')
        print(clf)

    tuned_models_list = tuned_models.items()

    stop = time()
    time_elapsed = get_time_string(stop-start)
    pipeline_automator.metadata['model_tuning_time'] = time_elapsed

    return tuned_models

def get_parameter_distribution(model_name, random_state=None, n_iter=10):
    """
    returns a dictionary containing key=classifier_name:value=parameter_distribution_dictionary
    """
    from scipy.stats import randint as sp_randint
    from sklearn.gaussian_process.kernels import RBF
    from scipy.stats import expon as sp_expon
    
    np.random.seed(random_state)

    parameter_distributions =   {
                                    'SGD Classifier (log)'             : {
                                                                                'alpha': sp_expon(scale=0.1),
                                                                            },
                                    'KNN Classifier'                   :    { 
                                                                                'n_neighbors' : [3,4,5]
                                                                            },
                                    'SGD Classifier (linear SVM)'      :    {
                                                                                'base_estimator__alpha': sp_expon(scale=3e-5),
                                                                            },
                                    'Polynomial SVM Classifier'        :    { 
                                                                                'poly_svc__C': [1,3,5,7,9],
                                                                            },
                                    'RBF SVM Classifier'               :    { 
                                                                                'rbf_svc__C': [1,10,100,1000],
                                                                                'rbf_svc__gamma': sp_expon(scale=0.01)
                                                                            },
                                    'Decision Tree Classifier'         :    {  
                                                                                "max_depth"         : sp_randint(3, 15),
                                                                                "max_features"      : sp_randint(3, 15),
                                                                                "min_samples_split" : sp_randint(2, 10),
                                                                                "min_samples_leaf"  : sp_randint(1, 10),
                                                                            },
                                    'Random Forest Classifier'         :    {  
                                                                                "n_estimators"      : sp_randint(2, 150), 
                                                                                "max_depth"         : sp_randint(3, 15),
                                                                                "max_features"      : sp_randint(3, 15),
                                                                                "min_samples_split" : sp_randint(2, 10),
                                                                                "min_samples_leaf"  : sp_randint(1, 10),
                                                                            },
                                    'Extra Trees Classifier'           :    {  
                                                                                "n_estimators"      : sp_randint(2, 150),
                                                                                "max_depth"         : sp_randint(3, 15),
                                                                                "max_features"      : sp_randint(3, 15),
                                                                                "min_samples_split" : sp_randint(2, 10),
                                                                                "min_samples_leaf"  : sp_randint(1, 10),
                                                                            },
                                    'AdaBoosting Trees Classifier'     :    {
                                                                                "n_estimators"                      : sp_randint(2, 150), 
                                                                                "base_estimator__max_depth"         : sp_randint(3, 15),
                                                                                "base_estimator__max_features"      : sp_randint(3, 15),
                                                                                "base_estimator__min_samples_split" : sp_randint(2, 10),
                                                                                "base_estimator__min_samples_leaf"  : sp_randint(1, 10),
                                                                            },
                                    'MLP Classifier'     :                  {
                                                                                'hidden_layer_sizes'    : get_dist_mlp_layers(
                                                                                                                int(n_iter//3),
                                                                                                                min_num_layers=4, 
                                                                                                                max_num_layers=8,
                                                                                                                min_neurons_per_layer=50, 
                                                                                                                max_neurons_per_layer=200
                                                                                                            )
                                                                            },                                                                                                                                                                            
                                }

    return parameter_distributions[model_name]

def get_ensemble_voting_classifier(X_train, y_train, component_classifiers, voting = 'hard'):
    """
    An ensemble classifier system that performs classification based on voting of component classifiers called estimators
    """
    from sklearn.ensemble import VotingClassifier
    
    name = 'Voting Classifier' + ' (' + voting + ')'
    clf = VotingClassifier(
        estimators=component_classifiers, 
        voting=voting) #soft voting applies confidence weighting

    clf.fit(X_train, y_train)

    return name, clf

def get_dist_mlp_layers(n_iter, min_num_layers=3, max_num_layers=10, min_neurons_per_layer=100, max_neurons_per_layer=200):
    from scipy.stats import randint as sp_randint

    num_layers_dist = sp_randint(min_num_layers, max_num_layers)
    num_neurons_dist = sp_randint(min_neurons_per_layer, max_neurons_per_layer)

    rtn = [ [num_neurons_dist.rvs() for layer in range(num_layers_dist.rvs())] for iteration in range(n_iter) ]
    return rtn