class PipelineAutomator:
    """
    A class that automates the machine learning pipeline used for generating good models that can 
    perform supervised, unsupervised, and reinforcement learning tasks.
    """
    def __init__(self, data_file_path, parameters):
        """
        Construct a PipelineAutomator. Right now the Automator is barebones:
        It simply reads in a data_file_path and a machine learning / AI model type that 
        you would like to generate using the provided data. Classifier is the only
        model type that the automator currently supports.
        """
        from file_handling import FileHandler

        self.file_handler = FileHandler('input_files', 'models')

        self.parameters = self._validate_parameters(self, parameters)

        self.metadata = {
            'data_file_path' : data_file_path,
            'number_of_records' : None,
            'text' : None,
            'words_count':None,
            'lemmatized_records': None,
            'lemmatized_corpus' : None,
            'lemmas' : None,
            'lemma_count' : None,
            'bigram_collocations' : None,
            'bigram_collocation_count' : None,
            'trigram_collocations' : None,
            'trigram_collocation_count' : None,
            'terms' : None,
            'term_count' : None,
            'ngrams' : {'words' : None, 'bigrams' : None, 'trigrams' : None},
            'features' : None,
            'features_count' : None,
            'selected_features': None,
            'selected_features_count':None,
            'zero_row_vector_count':None,
            'feature_selected_matrix_shape':None,
            'feature_selected_matrix': None,
            'training_data' : None,
            'training_data_size' : None,
            'testing_data': None,
            'testing_data_size' : None,
            'preprocessing_time' : None,
            'feature_selection_time' : None,
            'model_selection_time' : None,
            'model_tuning_time' : None,
            'model_evaluation_time' : None
        }

    def generate_model(self):
        """
        Initiates the pipeline by getting the data in the form of a .csv file and performs: 
        preprocessing, feature selection, model selection, model tuning, and model evaluation, 
        and finally deploys and computes statistical information if the user is happy with the 
        generated machine learning model. Return the model to be saved in the calling program.
        """
        from preprocessing import preprocess
        from feature_selection import feature_select
        from model_selection import get_promising_models
        from model_tuning import tune_models
        from model_evaluation import evaluate_best_models
        import numpy as np

        # used for reproducibility of output
        random_state = self.parameters['random_state']
        np.random.seed(random_state)

        # the data we wish to train our model on.
        data = self.file_handler.get_data( 
            self.metadata['data_file_path'],
            self.parameters['feature_col_name'], 
            self.parameters['label_col_name'],
            min_records=self.parameters['min_records_per_class'])

        data = self.separate_data(data)
        self.metadata["number_of_records"] = len(data)

        # the data void of junk words with lemmatization applied.
        cleaned_data = preprocess(data, self)

        # the cleaned data simplified by only including the selected features.
        feature_selected_data = feature_select(cleaned_data, self)

        # the best models the pipeline was able to generate given the data and pipeline parameters.
        promising_models = get_promising_models(feature_selected_data, self)
        
        # the best model after it's been trained on the entire dataset and had its parameters tuned.
        tuned_models = tune_models(promising_models, self)

        # the data from the best of the best models.
        best_model_data = evaluate_best_models(tuned_models, self)

        # the pipeline decides what to do based on results.
        self.evaluate_decision(best_model_data)

    def separate_data(self, data):
        from utility import split_data_ignore_include
        random_state = self.parameters['random_state']
        n_records = len(data) # n rows
        split_type = self.parameters['data_splitting']
        number_of_records_to_include = self.parameters['number_of_records_to_include']
        dont_use_all_records = (type(number_of_records_to_include)==float \
                                    and number_of_records_to_include<1.0) \
                                or (type(number_of_records_to_include)==int \
                                    and number_of_records_to_include<n_records)

        if dont_use_all_records:
            # For development: it is much more efficient to develop on a smaller sample of the data.
            print('Splitting our data for faster debugging...')
            if type(number_of_records_to_include) == float:
                number_of_records_to_include = round(number_of_records_to_include * n_records)
                print(str(self.parameters['number_of_records_to_include'])+'% * ',
                    '=',number_of_records_to_include)
                ignored_data, data = split_data_ignore_include(data, 
                    test_size=number_of_records_to_include, 
                    type_=split_type, 
                    random_state=random_state)
            elif number_of_records_to_include < n_records : 
                # int and less than the n_records
                ignored_data, data = split_data_ignore_include(data, 
                    test_size=number_of_records_to_include, 
                    type_=split_type, 
                    random_state=random_state)
            else:
                # int and greater than or equal to n_records so use all of the data.
                ignored_data = []
            print(len(ignored_data),'data is being ignored and',len(data),'records will be used...')
        return data

    @staticmethod
    # FOR FUTURE DEVELOPERS: You can play around with different pipeline automator setting quickly by changing 
    # them directly here or writing a bash script to automate many pipeline automator tests until that is 
    # incorporated into this code along with an interface for modifying these settings.
    def get_parameter_defaults_and_options():
        parameter = {
            'default': {
                # Developer Parameters
                'model_type' : 'classifier', # To-Do: this is all that is supported at the moment
                'feature_col_name' : None,
                'label_col_name' : None,
                'verbosity' : 1, # To-Do: not implemented
                'number_of_records_to_include' : 0.05,
                'min_records_per_class' : 5000,
                'random_state': 5, # a random seed used to replicate results.
                # Preprocessing Parameters
                'use_spell_checking' : True, # not implemented
                'title_proper_nouns' : True,
                'uppercase_acronyms' : True,
                'remove_sub_terms' : False,
                #-----Lemma Control Parameters
                'lemmatize_unigrams' : True, #not implemented
                'lemmas_are_alpha' : True,
                'lemmas_are_not_stop' : True,
                'lemmas_are_dictionary' : True,
                'lemmas_allow_number' : False,
                'lemmas_allow_alphanum' : False,
                'lemmas_allow_acronyms' : False,
                'lemmas_allow_proper_nouns': False,
                'lemmas_freq_requirement' : 20,
                'lemmas_len_requirement' : 3,
                #-----Bigram Collocation Control Parameters
                'bi_cols_term_count' : 300,
                'bi_cols_are_alpha' : True,
                'bi_cols_are_dictionary' : True,
                'bi_cols_are_not_stop' : True,
                'bi_cols_allow_number' : False,
                'bi_cols_allow_alphanum' : False,
                'bi_cols_allow_acronyms' : False,
                'bi_cols_allow_proper_nouns': False,
                'bi_cols_freq_requirement' : 20,
                #-----Trigram Collocation Control Parameters
                'tri_cols_term_count' : 200,
                'tri_cols_are_alpha' : True,
                'tri_cols_are_dictionary' : True,
                'tri_cols_are_not_stop' : True,
                'tri_cols_allow_number' : False,
                'tri_cols_allow_alphanum' : False,
                'tri_cols_allow_acronyms' : False,
                'tri_cols_allow_proper_nouns': False,
                'tri_cols_freq_requirement' : 10,
                #-----Meta and Engineered Features Control Parameters
                'use_meta_features' : True,
                'use_eng_features_emotions' : True,
                'use_eng_features_subjectivity' : True,
                # Feature Selection Parameters
                'n_selected_features' : 0.1,
                'feature_selection_metric' : 'chi^2',
                'term_value' : 'frequency',
                'use_tfidf_scaling' : True,
                'use_L2_row_normalization' : True,
                'remove_zero_length_vectors' : True, # not fully implemented (filters by entire vector length only)
                # Model Selection Parameters
                'model_selection_data_limit' : 1500,
                'data_splitting' : 'stratified',
                'training_set_size' : 0.8,
                'testing_set_size' : 0.2,
                'model_selection_param_tuning' : False, # not implemented.
                'model_selection_cv_folds_count' : 5,
                'n_promising_models_to_select' : 3,
                'model_selection_performance_metric' : 'f1_macro',
                'model_selection_min_performance' : 0.1,
                # Model Tuning Parameters
                'n_hyperparam_combos' : 10,
                'model_tuning_cv_folds_count' : 5,
                'include_ensemble_voters' : True,
                'model_tuning_scoring' : 'f1_macro',
                # Model Evaluation Parameters
                'final_min_performance_metric' : 'f1_macro',
                'final_min_performance' : 0.0,

                # Statistics Parameters
                        },
            'options' : {
                'model_type' : {'classifier', 'regressor', 'recommender', 'visualizer', 'chatbot', 'reinforcement_learner'},
                'feature_col_name' : [lambda s: type(s)==str],
                'label_col_name' : [lambda s: type(s)==str],
                'verbosity' : {0,1,2,3},
                'number_of_records_to_include' : {lambda sz: type(sz)==float and 0.0<sz<=1.0, lambda sz: type(sz)==int and sz > 0},
                'min_records_per_class' : {lambda d: type(d)==int},
                'random_state': {None, lambda sz: type(sz)==int},
                # Preprocessing Parameters
                'use_spell_checking' : {True, False},
                'title_proper_nouns' : {True, False},
                'uppercase_acronyms' : {True, False},
                'remove_sub_terms' : {True, False},
                #-----Lemma Control Parameters
                'lemmatize_unigrams' : {True, False},
                'lemmas_are_alpha' : {True, False},
                'lemmas_are_dictionary' :  {True, False},
                'lemmas_allow_stop' : {True, False},
                'lemmas_allow_number' :  {True, False},
                'lemmas_allow_alphanum' :  {True, False},
                'lemmas_allow_acronyms' :  {True, False},
                'lemmas_allow_proper_nouns':  {True, False},
                #-----Bigram Collocation Control Parameters
                'bi_cols_term_count' : {None, lambda sz: type(sz)==int},
                'bi_cols_are_alpha' : {True, False},
                'bi_cols_are_dictionary' : {True, False},
                'bi_cols_allow_stop' : {True, False},
                'bi_cols_allow_number' : {True, False},
                'bi_cols_allow_alphanum' : {True, False},
                'bi_cols_allow_acronyms' : {True, False},
                'bi_cols_allow_proper_nouns': {True, False},
                'bi_cols_freq_requirement' : {lambda sz: type(sz)==int and sz >= 0},
                #-----Trigram Collocation Control Parameters
                'tri_cols_term_count' : {None, lambda sz: type(sz)==int},
                'tri_cols_are_alpha' : {True, False},
                'tri_cols_are_dictionary' : {True, False},
                'tri_cols_allow_stop' : {True, False},
                'tri_cols_allow_number' : {True, False},
                'tri_cols_allow_alphanum' : {True, False},
                'tri_cols_allow_acronyms' : {True, False},
                'tri_cols_allow_proper_nouns': {True, False},
                'tri_cols_freq_requirement' : {lambda sz: type(sz)==int and sz >= 0},
                #-----Meta and Engineered Features Control Parameters
                'use_meta_features' : {True, False},
                'use_eng_features_emotions' : {True, False},
                'use_eng_features_obj_subj' : {True, False},
                # Feature Selection Parameters
                'n_selected_features' : {lambda n: type(n)==int and n>0},
                'feature_selection_metric' : {'chi^2', 'F', 'mutual_info'},
                'term_value' : {'frequency', 'presence'},
                'use_tfidf_scaling' : {True, False},
                'use_L2_row_normalization' : {True, False},
                # Model Selection Parameters
                'model_selection_data_limit' : {lambda k: type(k)==int and k>0},
                'data_splitting' : {'stratified', 'random'},
                'training_set_size' : {lambda sz: type(sz)==float and 0<sz<1, lambda sz: type(sz)==int},
                'testing_set_size' : {lambda sz: type(sz)==float and 0<sz<1, lambda sz: type(sz)==int},
                'model_selection_param_tuning' : {True, False},
                'model_selection_cv_folds' : {lambda k: type(k)==int and k>0},
                'n_promising_models_to_select' :  {lambda n: type(n)==float and 0<n<=1.0, lambda n: type(n)==int},
                'model_selection_performance_metric' : {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'},
                'model_selection_min_performance' : {lambda n: type(n)==float and 0.0<=n<=1.0},
                # Model Tuning Parameters
                'n_hyperparam_combos' :  {lambda sz: type(sz)==int and sz >= 0},
                'model_tuning_cv_folds' : 5,
                'include_ensemble_voters' : {True, False}, 
                'model_tuning_scoring' : {'f1_macro', 'f1_micro'},
                # Model Evaluation Parameters
                'final_min_performance_metric' : {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'},
                'final_min_performance' : {lambda n: type(n)==float and 0.0<=n<=1.0},
            }
        }
        return parameter

    @staticmethod
    def _validate_parameters(self, inputted_pipeline_parameters):
        """
        Checks to make sure that the pipeline_parameters are appropriate.
        """
        from copy import deepcopy
        from utility import has_callables
        from functools import reduce
        
        # The default value and set of value options of every parameter.
        parameter = PipelineAutomator.get_parameter_defaults_and_options()

        pipeline_parameters = deepcopy(parameter['default'])

        for param in inputted_pipeline_parameters:
            value = inputted_pipeline_parameters[param]
            if param in parameter['default']:
                print(type(parameter['options'][param][0]) == callable)
                if value in parameter['options'][param]:
                    pipeline_parameters[param] = value

                elif has_callables(parameter['options'][param]):
                    if len(parameter['options'][param]) == 1:
                        if parameter['options'][param][0](value):
                            pipeline_parameters[param] = value
                    elif reduce(lambda callable1, callable2: callable1(value) or callable2(value)):
                        pipeline_parameters[param] = value
                    else:
                        raise ValueError('You passed an invalid value for parameter', param,' which takes the following types:', parameter['options'][param])
                        exit()
                else:
                        raise ValueError('You passed an invalid value for parameter', param,' which takes the following types:', parameter['options'][param])
                        exit()
            else:
                raise AttributeError('PipelineAutomator object has no attribute called ', param + '.')
                exit() 

        return pipeline_parameters

    def evaluate_decision(self, model_data):
        """
        Using the results from the model_decision, prompt the user and decide what to do next.
        """
        from preprocessing import Preprocessor
        from feature_selection import FeatureSelector
        from sklearn.pipeline import Pipeline
        
        preprocess = Preprocessor(self)
        feature_select = FeatureSelector(self)
        model_name, model, model_performance = model_data
        steps = [
            ('preprocess', preprocess),
            ('feature_select', feature_select),
            ('clf', model)

        ]
        pipeline = Pipeline(steps) # this is our classifier pipeline that transforms data and makes predictions.

        metric = self.parameters['final_min_performance_metric']
        model_performance = model_performance[metric]
        min_performance = self.parameters['final_min_performance']
        if model_performance > min_performance:
            print('Minimum performance required:', min_performance, metric)
            print('Model performance', model_performance, metric)
            print('The model meets minimum requirements!')
            deploy = input('Type "C" to cancel, or type anything else to save the model: ')
            if deploy.strip().lower() != 'c':
                file_name = input('Enter file name:')
                # save the model so it can be easily loaded next time and used to make predictions.
                self.file_handler.save_model(file_name, pipeline)

    def display_parameters(self):
        """ Prints the values of the parameters."""
        display_string = """Pipeline Automator (v1)"""
        col1_width = max([len(parameter) for parameter in self.parameters ])
        col2_width = max([len(str(self.parameters[parameter])) for parameter in self.parameters ])
        display_string += ("\n"+"~"*(col1_width+col2_width+1)+"\n")
        display_string += ("Pipeline Parameters:")

        for parameter in self.parameters:
            value = str(self.parameters[parameter])
            substring = f"{parameter:<{col1_width}s}:{value:>{col2_width}s}"
            display_string += "\n" + substring
        
        display_string += "\n"
        print(display_string)

    def display_metadata(self, shortened=True, col2_width=40):
        """ Prints the the values of the metadata."""
        display_string = """ """
        col1_width = max([len(category) for category in self.metadata])
        display_string += ("\n"+"~"*(col1_width+col2_width+1)+"\n")
        display_string += ("Pipeline Metadata:")
        
        for category in self.metadata:
            value = str(self.metadata[category])
            
            if shortened:
                if len(value) > col2_width:
                    value = value[:col2_width-3] + '...'
            
            substring = f"{category:<{col1_width}s}:{value:>{col2_width}s}"
            display_string += "\n" + substring
        
        display_string += "\n"
        print(display_string)