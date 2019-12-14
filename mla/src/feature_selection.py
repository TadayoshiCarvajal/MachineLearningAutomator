from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
from numpy import ndarray
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from functools import reduce
from sklearn.preprocessing import normalize
from time import time
from utility import get_time_string
from sklearn.base import BaseEstimator, TransformerMixin

stop_words=set(stopwords.words('english'))

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_automator):
        self.pipeline_automator = pipeline_automator

    def transform(self, X):
        tfidf_matrix, tfidf_terms = get_tfidf_matrix(X, self.pipeline_automator, only_transform=True)
        meta_features_matrix = get_meta_features_matrix(X)[0]
        X, y = tfidf_matrix[:,:-1], tfidf_matrix[:,-1]
        meta_features_tfidf_matrix = np.column_stack( [X, meta_features_matrix, y] )
        return univariate_feature_selection(meta_features_tfidf_matrix,
                None,
                None,
                None,
                self.pipeline_automator,
                only_transform=True)

    def fit(self, X, y=None):
        return self

def feature_select(data, pipeline_automator):
    print('\n\nFeature Selection... ')
    start = time()

    # seed the random number generator for reproducibility.
    random_state = pipeline_automator.parameters['random_state']
    np.random.seed(random_state)

    # Take our cleaned data and terms and transform into a TF-IDF matrix with L2-normalization applied to the row vectors
    terms = pipeline_automator.metadata['terms']
    print('Getting TF-IDF Matrix...')
    tfidf_matrix, tfidf_terms = get_tfidf_matrix(data, pipeline_automator, print_matrix=True)

    # Add the Meta-Features to the tf-idf matrix
    print('Adding Meta-Features...')
    meta_features_matrix, meta_features_col_names = get_meta_features_matrix(data)
    if meta_features_matrix is not None:
        X, y = tfidf_matrix[:,:-1], tfidf_matrix[:,-1]
        X = X.astype(np.float64)
        y = y.reshape( (y.shape[0], 1) )
        y = np.array(y.T)[0]
        meta_features_tfidf_matrix = np.column_stack( [X, meta_features_matrix, y] )
        features = tfidf_terms + meta_features_col_names
        print('tfidf + meta features shape:', meta_features_tfidf_matrix.shape )
    else:
        meta_features_tfidf_matrix = tfidf_matrix

    features_count = int(meta_features_tfidf_matrix.shape[1] - 1) #exclude the label column

    # Selected Features Matrix
    # Use the specified feature selection metric and the number of features to keep to determine which terms aka features to select
    print('Performing Univariate Feature Selection...')
    feature_selection_metric = pipeline_automator.parameters['feature_selection_metric']
    n_features_to_keep = int(features_count * pipeline_automator.parameters['n_selected_features']) # number / ratio of features to select

    top_features, top_features_matrix = univariate_feature_selection(meta_features_tfidf_matrix, 
                                        n_features_to_keep, 
                                        feature_selection_metric, 
                                        features,
                                        pipeline_automator)
    print('reduced tfidf shape:',top_features_matrix.shape)


    if pipeline_automator.parameters['remove_zero_length_vectors']:
        # Some records may not contain any of the selected features and should thus be ignored.
        top_features_matrix = remove_zero_length_vectors(top_features_matrix)
    
    print('Selected Features Matrix shape:', top_features_matrix.shape)
    print(top_features_matrix)

    for feature in top_features:
        print(feature)

    if pipeline_automator.parameters['use_L2_row_normalization']:
        # the remaining vectors are then normalized one last time to take the meta features into account.
        top_features_matrix = normalize_matrix(top_features_matrix)

    # Cache the metadata
    zero_row_vector_count = len(data) - len(top_features_matrix)
    feature_selection_matrix_shape = top_features_matrix.shape
    pipeline_automator.metadata['features'] = features
    pipeline_automator.metadata['features_count'] = len(features)
    pipeline_automator.metadata['selected_features'] = top_features
    pipeline_automator.metadata['selected_features_count'] = n_features_to_keep
    pipeline_automator.metadata['zero_row_vector_count'] = zero_row_vector_count
    pipeline_automator.metadata['feature_selected_matrix_shape'] = feature_selection_matrix_shape
    pipeline_automator.metadata['feature_selected_matrix'] = top_features_matrix
    stop = time()
    time_elapsed = get_time_string(stop - start)
    pipeline_automator.metadata['feature_selection_time'] = time_elapsed
    # Return the selected terms tf-idf-L2 scaled matrix representation of the data
    return top_features_matrix

def get_tfidf_matrix(data, pipeline_automator, print_matrix = False, only_transform=False):
    #combine the original description w/ lemma phrase for the count vectorizer to work. This only works if the description text is lowercased.
    records_set_lemmas = [ record[1] for record in data ]
    records_set = [ record[0] for record in data ]
    labels_set = [ record[3] for record in data ]

    lemmas = pipeline_automator.metadata['lemmas']
    colloc2= pipeline_automator.metadata['bigram_collocations']
    colloc3= pipeline_automator.metadata['trigram_collocations']

    binary_values_boolean = True if pipeline_automator.parameters['term_value'] == 'presence' else False
    norm_value = 'l2' if pipeline_automator.parameters['use_L2_row_normalization'] else None
    use_tfidf_scaling = not binary_values_boolean and pipeline_automator.parameters['use_tfidf_scaling']
    
    y = np.matrix(labels_set).T

    terms_matrices = []
    
    if lemmas:
        vectorizer_lemma = CountVectorizer(vocabulary=lemmas, ngram_range=(1,1), lowercase=False, binary=binary_values_boolean)
        X_lemmas = vectorizer_lemma.fit_transform(records_set_lemmas)
        if not only_transform: print('Vectorizing lemmas...')
        terms_idx_mapping_lemmas = {v:k for k,v in vectorizer_lemma.vocabulary_.items()}
        vocabulary = [terms_idx_mapping_lemmas[i] for i in range(len(terms_idx_mapping_lemmas))] #list index == column index in the matrix
        term_frequency_matrix_lemmas = X_lemmas.todense()
        terms_matrices.append(term_frequency_matrix_lemmas)    

    if len(colloc2) > 0:
        vectorizer_bigrams = CountVectorizer(vocabulary=colloc2, ngram_range=(2,2), lowercase=False, binary=binary_values_boolean)
        X_bigrams = vectorizer_bigrams.fit_transform(records_set)
        if not only_transform: print('Vectorizing bigram collocations...')
        terms_idx_mapping_bigrams = {v:k for k,v in vectorizer_bigrams.vocabulary_.items()}
        vocabulary += [terms_idx_mapping_bigrams[i] for i in range(len(terms_idx_mapping_bigrams))] #list index == column index in the matrix
        term_frequency_matrix_bigrams = X_bigrams.todense()
        terms_matrices.append(term_frequency_matrix_bigrams)

    if len(colloc3) > 0:
        vectorizer_trigrams = CountVectorizer(vocabulary=colloc3, ngram_range=(3,3), lowercase=False, binary=binary_values_boolean)
        X_trigrams = vectorizer_trigrams.fit_transform(records_set)
        if not only_transform: print('Vectorizing trigrams collocations...')
        terms_idx_mapping_trigrams = {v:k for k,v in vectorizer_trigrams.vocabulary_.items()}
        vocabulary += [terms_idx_mapping_trigrams[i] for i in range(len(terms_idx_mapping_trigrams))] #list index == column index in the matrix
        term_frequency_matrix_trigrams = X_trigrams.todense()
        terms_matrices.append(term_frequency_matrix_trigrams)
    
    term_frequency_matrix = np.column_stack(terms_matrices)

    if use_tfidf_scaling and not only_transform:
        tfidf = TfidfTransformer(norm=norm_value, use_idf=True)
        tfidf.fit(term_frequency_matrix)
        tfidf_matrix = tfidf.transform(term_frequency_matrix).todense()
        pipeline_automator.metadata["tfidf_transformer"] = tfidf
        pipeline_automator.metadata['idf_matrix'] = tfidf.idf_ # necessary for converting new examples into the correct format.
        final_matrix = np.concatenate( [tfidf_matrix, y], 1)
    elif use_tfidf_scaling and only_transform:
        tfidf = pipeline_automator.metadata['tfidf_transformer']
        tfidf_matrix = tfidf.transform(term_frequency_matrix).todense()
        final_matrix = np.concatenate( [tfidf_matrix, y], 1)
    else:
        final_matrix = np.concatenate( [term_frequency_matrix, y], 1)

    if print_matrix:
        print(final_matrix)
        print(final_matrix.shape)

    return final_matrix, vocabulary

def univariate_feature_selection(tfidf_matrix, k, metric, terms, pipeline_automator, only_transform=False):
    """
    Uses the f-statistic and chi-2 statistic to filter the k-best features for classification.
    """
    X, y = tfidf_matrix[:,:-1], tfidf_matrix[:,-1]
    X = X.astype(np.float64)
    y = y.reshape( (y.shape[0], 1) )
    y = np.array(y.T)[0]

    if not only_transform:
        statistic_metrics = {'chi^2':chi2 , 'F':f_classif, 'mutual_info':mutual_info_classif}
        statistic_names = {f_classif: 'F Score', chi2: 'Chi-Squared Score', mutual_info_classif: 'Mutual Information Score' } 
        statistic = statistic_metrics[metric]
        print('Calculating best features using:', statistic_names[statistic])
        # fit the SelectKBest object to the data:
        bestfeatures = SelectKBest(score_func=statistic, k=k)
        pipeline_automator.metadata["best_features"] = bestfeatures
        fit = bestfeatures.fit(X,y)
        tf_X = bestfeatures.transform(X)
        top_features_matrix = np.column_stack([tf_X, y])

        top_features = [] # the list of the K best features
        mask = bestfeatures.get_support() # list of booleans
        for boolean, feature in zip(mask, terms):
            if boolean:
                top_features.append(feature)
    else:
        # fit the SelectKBest object to the data:
        bestfeatures = pipeline_automator.metadata["best_features"]
        tf_X = bestfeatures.transform(X)
        return tf_X


    return top_features, top_features_matrix

def get_meta_features_matrix(data):
    from sklearn.preprocessing import MinMaxScaler

    metafeatures_column = [row[2] for row in data] #list of metafeatures dictionaries

    meta_features = list(sorted(metafeatures_column[0].keys()))

    if len(meta_features) > 0:
        meta_features_2d_list = [ [ dictionary[feature] for feature in meta_features] for dictionary in metafeatures_column ]
        unscaled_meta_features_matrix = np.array(meta_features_2d_list)
        
        scaler = MinMaxScaler()
        scaler.fit(unscaled_meta_features_matrix)
        scaled_meta_features_matrix = scaler.transform(unscaled_meta_features_matrix)

    else:
        return None, None
    return scaled_meta_features_matrix, meta_features

def normalize_matrix(matrix):
    """
    normalizes a matrix where the first n_cols-1 columns are features and the last column is a label.
    """
    X, y = matrix[:,:-1], matrix[:,-1]
    X = X.astype(np.float64)
    y = y.reshape( (y.shape[0], 1) )
    y = np.array(y.T)[0]
    X_norm = normalize(X, axis=1, norm='l2')
    normalized_matrix = np.column_stack([X_norm, y])
    return normalized_matrix

def remove_zero_length_vectors(matrix):
    """
    Some rows will not contain any of our selected terms, we remove these from consideration.
    """
    def vector_length(x):
        """
        Used to calculate the length of the vectors to check for any of zero length.
        """
        from math import sqrt
        return sqrt(sum([xi**2 for xi in x]))

    features = []
    labels = []
    for row in matrix:  
        total_row_vector = np.squeeze(np.asarray(row))
        terms_row_vector = total_row_vector[:-1]
        label = [total_row_vector[-1]]
        term_vals = list( float(val) for val in terms_row_vector )
        if vector_length(term_vals) > 0.0:
            features.append(term_vals)
            labels.append(label)
    features_and_label = np.column_stack([np.array(features), np.array(labels)])

    return features_and_label