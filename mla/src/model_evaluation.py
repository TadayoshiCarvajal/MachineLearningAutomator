from model_selection import score_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report
from time import time
from utility import get_time_string

def evaluate_best_models(tuned_models, pipeline_automator):
    import numpy as np
    performance_metric = pipeline_automator.parameters['final_min_performance_metric']
    min_performance = pipeline_automator.parameters['final_min_performance']

    # This is used to control reproducibility in the output.
    random_state = pipeline_automator.parameters['random_state']
    np.random.seed(random_state)

    print('\n\nModel Evaluation...')
    start = time()
    # Prepare features matrix and label vector for final evaluation.
    test_data = pipeline_automator.metadata['testing_data']
    n_features = test_data.shape[1] - 1 # last column is the target column
    
    X, y = test_data[:,:n_features], test_data[:,n_features]
    X = X.astype(np.float64)
    y = y.reshape( (y.shape[0], 1) )
    y = np.array(y.T)[0] 

    # For each model in tuned models, calculate the F1-score on the test data and select the model with best F1-score. These models are already trained.
    classifiers = []
    
    for model_name in tuned_models:
        model = tuned_models[model_name]
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        balanced_accuracy = balanced_accuracy_score(y, y_pred,)
        precision = precision_score(y, y_pred, pos_label='Complaint', average='macro')
        recall = recall_score(y, y_pred, pos_label='Complaint', average='macro')
        f1 = f1_score(y, y_pred, pos_label='Complaint', average='macro')

        results = {
            'confusion matrix' : cm, 
            'balanced accuracy' : balanced_accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1_macro' : f1
        }
        print('Final Evaluation of Tuned', model_name,':')
        print(cm)
        print('balanced accuracy:', balanced_accuracy)
        print( classification_report(y, y_pred) )
        print()
        classifiers.append( (model_name, model, results) )

    ranked_classifiers = sorted( classifiers, key=lambda x: x[-1]['f1_macro'], reverse=True )

    for name, classifier, results in ranked_classifiers:
        print(f"{name:<35s}\tF1-score:{results['f1_macro']:<10f}")
    print()
    
    best_model_name, best_model, best_model_results = ranked_classifiers[0]

    if best_model_results[performance_metric] >= min_performance:
        print('The best model Pipeline Automator found was the', best_model_name,'.')
        print('results:')
        print(best_model_results['confusion matrix'])
        print('balanced accuracy:', best_model_results['balanced accuracy'])
        print('precision:', best_model_results['precision'])
        print('recall:', best_model_results['recall'])
        print('f1-score:', best_model_results['f1_macro'])
        print()
    else:
        print('No model had minimum performace of', performance_metric, '>=', min_performance,'. Please adjust the pipeline parameters and try again.\n')
        best_model_name, best_model, best_model_results = None, None, None
    stop = time()
    time_elapsed = get_time_string(stop-start)
    pipeline_automator.metadata['model_evaluation_time'] = time_elapsed

    return best_model_name, best_model, best_model_results