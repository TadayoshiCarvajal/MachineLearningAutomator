import nltk
from nltk import wordpunct_tokenize as tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from tqdm import tqdm
from time import time
from nltk.corpus import names
from utility import get_time_string
import enchant
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

stop_words = set(stopwords.words('english'))
names = set( names.words('male.txt') + names.words('female.txt') )
enchant_dict = enchant.Dict("en_US")
excluded_words_list = set()
min_n_words_in_record = float('inf')
max_n_words_in_record = float('-inf')

##### PREPROCESSING #####

class Preprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, pipeline_automator):
        self.pipeline_automator = pipeline_automator

    def transform(self, X):
        return clean_data(X, self.pipeline_automator, progress=False)
    
    def fit(self, X, y=None, ):
        return self

def preprocess(data, pipeline_automator):
    print('\n\nPreprocessing...')
    start = time()

    # Get the cleaned data
    cleaned_data = clean_data(data, pipeline_automator)

    # Get the corpus text
    corpus_text = get_corpus_text(data)

    # Get the list of words
    words = get_words(corpus_text)

    # Get the corpus tokens
    corpus_tokens = get_tokens(corpus_text)

    # Get the bigrams, trigrams, collocations and lemmas in the data
    bigrams = get_bigrams(corpus_tokens)
    trigrams = get_trigrams(corpus_tokens)
    collocations2 = get_bigram_collocations(corpus_tokens, pipeline_automator)
    collocations3 = get_trigram_collocations(corpus_tokens, pipeline_automator)
    lemmas = get_lemmas(cleaned_data, pipeline_automator)

    if pipeline_automator.parameters['remove_sub_terms']:
        lemmas, collocations2, collocations3 = remove_redundant_terms(lemmas, collocations2, collocations3)

    # Get the terms that will be selected from in the feature selection step (lemmas and collocations)
    terms = lemmas + collocations2 + collocations3

    # Store all of the meta-data generated during preprocessing.
    pipeline_automator.metadata['ngrams']['words'] = words
    pipeline_automator.metadata['words_count'] = len(words)
    pipeline_automator.metadata['lemmas'] = lemmas
    pipeline_automator.metadata['lemma_count'] = len(lemmas)
    pipeline_automator.metadata['text'] = corpus_text
    pipeline_automator.metadata['ngrams']['bigrams'] = list(bigrams)
    pipeline_automator.metadata['ngrams']['trigrams'] = list(trigrams)
    pipeline_automator.metadata['bigram_collocations'] = collocations2
    pipeline_automator.metadata['bigram_collocation_count'] = len(collocations2)
    pipeline_automator.metadata['trigram_collocations'] = collocations3
    pipeline_automator.metadata['trigram_collocation_count'] = len(collocations3)
    pipeline_automator.metadata['terms'] = terms
    pipeline_automator.metadata['term_count'] = len(terms)
    stop = time()
    time_elapsed = get_time_string(stop - start)
    pipeline_automator.metadata['preprocessing_time'] = time_elapsed

    return cleaned_data

##### COMPONENTS OF PREPROCESSING #####
def clean_data(data, pipeline_automator, progress=True):
    """
    get the data ready for feature selection, we need cleaned records and terms.
    """
    global min_n_words_in_record, max_n_words_in_record
    
    if progress:
        # Tokenization... Now our data has 3 parts: the original description, the semi_cleaned_description, and the label
        semi_clean_data = [ [str(desc), get_tokens(str(desc)), label] for desc, label in tqdm(data, 'Tokenizing Records...') ]

        # Tag parts of speech and lowercase everything unless it's a proper noun.
        semi_clean_data = [[desc, part_of_speech_tag(semi_clean_desc, pipeline_automator), label] for desc, semi_clean_desc, label in tqdm(semi_clean_data, 'Tagging Parts of Speech...')]

        # Get the meta-features. Now our data has 4 parts since we've added a dictionary containing metafeatures of each record.
        semi_clean_data = [ [desc, semi_clean_desc, get_meta_and_engineered_features(semi_clean_desc, desc, pipeline_automator), label] for desc, semi_clean_desc, label in tqdm(semi_clean_data, 'Getting Meta and Engineered features...') ]

        # Remove junk
        semi_clean_data = [ [desc, remove_junk(semi_clean_desc, pipeline_automator), metafeatures, label] for desc, semi_clean_desc, metafeatures, label in tqdm(semi_clean_data, 'Removing Junk Words...') ]

        # Lemmatize the records
        cleaned_data = [ [desc, lemmatize(semi_clean_desc, pipeline_automator), metafeatures, label] for desc, semi_clean_desc, metafeatures, label in tqdm(semi_clean_data, 'Lemmatizing Records...')  ]
        
    else:
        # Tokenization... Now our data has 3 parts: the original description, the semi_cleaned_description, and the label
        semi_clean_data = [ [str(desc), get_tokens(str(desc)), label] for desc, label in data ]

        # Tag parts of speech and lowercase everything unless it's a proper noun.
        semi_clean_data = [[desc, part_of_speech_tag(semi_clean_desc, pipeline_automator), label] for desc, semi_clean_desc, label in semi_clean_data]

        # Get the meta-features. Now our data has 4 parts since we've added a dictionary containing metafeatures of each record.
        semi_clean_data = [ [desc, semi_clean_desc, get_meta_and_engineered_features(semi_clean_desc, desc, pipeline_automator), label] for desc, semi_clean_desc, label in semi_clean_data ]

        # Remove junk
        semi_clean_data = [ [desc, remove_junk(semi_clean_desc, pipeline_automator), metafeatures, label] for desc, semi_clean_desc, metafeatures, label in semi_clean_data ]

        # Lemmatize the records
        cleaned_data = [ [desc, lemmatize(semi_clean_desc, pipeline_automator), metafeatures, label] for desc, semi_clean_desc, metafeatures, label in semi_clean_data ]
        
    return cleaned_data

def get_corpus_text(data):
    # the period below prevents cross record bigram and trigram recognition.
    return '. '.join([str(description) for description, label in tqdm(data,'Generating corpus...')]).lower()

def get_bigrams(corpus_tokens):
    return sorted( set([ ' '.join(bigram) for bigram in tqdm(nltk.bigrams(corpus_tokens), 'Getting bigrams...')  ]))

def get_trigrams(corpus_tokens):
    return sorted( set([ ' '.join(trigram) for trigram in tqdm(nltk.trigrams(corpus_tokens), 'Getting trigrams...')  ]))

def get_bigram_collocations(corpus_tokens, pipeline_automator):
    print('Getting bigram collocations...')
    bi_cols_term_count = pipeline_automator.parameters['bi_cols_term_count']
    bi_cols_freq_requirement = pipeline_automator.parameters['bi_cols_freq_requirement']
    word_is_junk = lambda w: not word_is_not_junk(w, pipeline_automator, 'bi_cols') 

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = nltk.BigramCollocationFinder.from_words(corpus_tokens, window_size=2)
    bigram_finder.apply_word_filter(lambda w: word_is_junk(w))
    bigram_finder.apply_freq_filter(bi_cols_freq_requirement)
    colls2 = bigram_finder.nbest(bigram_measures.likelihood_ratio, bi_cols_term_count)
    return sorted([ ' '.join(bigram) for bigram in colls2 ])

def get_trigram_collocations(corpus_tokens, pipeline_automator):
    print('Getting trigram collocations...')
    tri_cols_term_count = pipeline_automator.parameters['tri_cols_term_count']
    tri_cols_freq_requirement = pipeline_automator.parameters['tri_cols_freq_requirement']
    word_is_junk = lambda w: not word_is_not_junk(w, pipeline_automator, 'tri_cols')

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigram_finder = nltk.TrigramCollocationFinder.from_words(corpus_tokens, window_size=3)
    trigram_finder.apply_word_filter( lambda w: word_is_junk(w) )
    trigram_finder.apply_freq_filter(tri_cols_freq_requirement)
    colls3 = trigram_finder.nbest(trigram_measures.likelihood_ratio, tri_cols_term_count)
    return sorted([ ' '.join(trigram) for trigram in colls3  ])

def get_lemmas(data, pipeline_automator):
    lemma_freq_requirement = pipeline_automator.parameters['lemmas_freq_requirement']
    lemma_len_requirement = pipeline_automator.parameters['lemmas_len_requirement']
    lemmas = {}
    for description, lemma_phrase, metafeatures, label in data:
        record_lemmas = set(lemma_phrase.split())
        for lemma in record_lemmas:
            if len(lemma) >= lemma_len_requirement:
                if lemma in lemmas:
                    lemmas[lemma] += 1
                else:
                    lemmas[lemma] = 1
    rtn = [ key for key in lemmas if lemmas[key] >= lemma_freq_requirement ]
    return sorted(rtn)

def get_words(corpus_text):
    return set(corpus_text.split())

def remove_redundant_terms(lemmas, bigrams, trigrams):
    '''
    Given a list of strings where each string is a term, check if lemma is a component of a bigram.
    if it is, remove the lemma from the terms since the collocation is probably more important.
    Repeat the same for bigrams and trigrams, i.e. make sure none of the bigram collocations are
    components of any of the trigram collocations.
    '''
    lemmas_to_keep = []
    bigrams_to_keep = []

    for lemma in lemmas:
        in_a_bigram = False # start off assuming each lemma is not a bigram component.
        for bigram in bigrams:
            bigram = bigram.split()
            if lemma in bigram:
                in_a_bigram = True
                break # don't need to check any more bigrams.

        # if not in any bigrams then keep
        if not in_a_bigram:
            lemmas_to_keep.append(lemma)

    for bigram in bigrams:
        bigram_list = bigram.split()
        in_a_trigram = False
        for trigram in trigrams:
            trigram_list = trigram.split()
            if bigram_list == trigram_list[:2] or bigram_list == trigram_list[1:]:
                in_a_trigram = True
                break # don't need to keep checking.
        
        # if not in any trigrams then keep
        if not in_a_trigram:
            bigrams_to_keep.append(bigram)

    return lemmas_to_keep, bigrams_to_keep, trigrams

##### COMPONENTS OF DATA CLEANING #####
def get_tokens(string):
    global stop_words
    rtn = []
    tokens = tokenize(string)

    for token in tokens:
        if token.isalnum() and token not in stop_words:
            rtn.append(token)
    return rtn
    
def part_of_speech_tag(tokens, pipeline_automator):    
    word_tags = pos_tag(tokens)
    # lower case every token unless we specify not to in pipeline automator..
    for i in range(len(word_tags)):
        word, pos = word_tags[i]
        if pos.startswith('J'):
            pos = 'a'
        elif pos.startswith('V'):
            pos =  'v'
        elif pos.startswith('R'):
            pos =  'r'
        elif pos.startswith('N'):
            pos =  'n'
        else:
            pos =  'o'
        word_tags[i] = (word.lower(), pos)
    return word_tags

def get_meta_and_engineered_features(word_pos_tokens, original_description, pipeline_automator):
    use_meta_features = pipeline_automator.parameters['use_meta_features']
    use_eng_features_subjectivity = pipeline_automator.parameters['use_eng_features_subjectivity']

    meta_and_eng_features = {}

    if use_meta_features:
        meta_and_eng_features.update(get_meta_features(word_pos_tokens, original_description))
    if use_eng_features_subjectivity:
        meta_and_eng_features.update(get_subjectivity_sentiment_analysis(original_description))
        
    return meta_and_eng_features

def get_meta_features(word_pos_tokens, string):
    from nltk import sent_tokenize

    # word and sentence data
    sentence_list = list(sent_tokenize(string))
    words_list = []
    # parts of speech data
    number_nouns = 0
    number_verbs = 0
    number_adj = 0
    number_adv = 0
    number_other = 0
    for word, pos in word_pos_tokens:
        if pos == 'n': number_nouns += 1
        if pos == 'v': number_verbs += 1
        if pos == 'a': number_adj += 1
        if pos == 'r': number_adv += 1
        if pos == 'o': number_other += 1
        words_list.append(word)

    number_words = len(words_list)
    if words_list:
        average_word_length = sum([len(word) for word in words_list]) / number_words
        average_sentence_length = number_words / len(sentence_list) #words/sentence
        lexical_diversity = len(set(words_list)) / number_words
        noun_ratio = number_nouns / number_words
        verb_ratio = number_verbs / number_words
        adj_ratio = number_adj / number_words
        adv_ratio = number_adv / number_words
        other_pos_ratio = number_other / number_words
    else:
        average_word_length = 0
        average_sentence_length = 0
        lexical_diversity = 0
        noun_ratio = 0
        verb_ratio = 0
        adj_ratio = 0
        adv_ratio = 0
        other_pos_ratio = 0

    meta_features = {
        'meta__words_count' : number_words,
        'meta__sentence_count' : len(sentence_list),
        'meta__avg_word_length' : average_word_length, 
        'meta__avg_sentence_length' : average_sentence_length,
        'meta__lexical_diversity' : lexical_diversity,
        'meta__noun_ratio' : noun_ratio,
        'meta__verb_ratio' : verb_ratio,
        'meta__adjective_ratio' : adj_ratio,
        'meta__adverb_ratio' : adv_ratio,
        'meta__other_pos_ratio' : other_pos_ratio, # other parts of speech(not n,v,a, or r)
        }
    return meta_features

def get_subjectivity_sentiment_analysis(string):
    from textblob import TextBlob

    agg_tb_scores =   {
                'eng__subjectivity_avg': 0.0,
                'eng__polarity_avg': 0.0, 
    } # the sentiments collected from TextBlob

    # get the polarity and subjectivity.
    description_sentiment = TextBlob(string).sentiment
    agg_tb_scores['eng__polarity_avg'] = description_sentiment.polarity
    agg_tb_scores['eng__subjectivity_avg'] = description_sentiment.subjectivity
    return agg_tb_scores

def word_is_not_junk(w, pipeline_automator, junk_filter_type='lemmas'):
    valid_filter_types = {'bi_cols', 'tri_cols', 'lemmas'}
    if junk_filter_type not in valid_filter_types:
        raise ValueError(junk_filter_type,'not in',valid_filter_types)

    # Functions used to build higher level boolean functions.
    # Think of these as individual requirements. Returns True is the req type is met otherwise False.
    lemmas_are_alpha = pipeline_automator.parameters[junk_filter_type+'_are_alpha']
    word_is_alpha = (lambda word: word.isalpha() if lemmas_are_alpha else lambda x: True)

    lemmas_are_dictionary = pipeline_automator.parameters[junk_filter_type+'_are_dictionary']
    word_is_dict= (lambda word: enchant_dict.check(word) if lemmas_are_dictionary else lambda x: True)

    lemmas_are_dictionary = pipeline_automator.parameters[junk_filter_type+'_are_dictionary']
    word_is_dict= (lambda word: enchant_dict.check(word) if lemmas_are_dictionary else lambda x: True)

    lemmas_are_not_stop = pipeline_automator.parameters[junk_filter_type+'_are_not_stop']
    word_is_not_stop = (lambda word: word.lower() not in stop_words if lemmas_are_not_stop else lambda x: True )

    lemmas_allow_number = pipeline_automator.parameters[junk_filter_type+'_allow_number']
    word_is_num = (lambda word: word.isnumeric() if lemmas_allow_number else lambda x: False)

    lemmas_allow_alphanum = pipeline_automator.parameters[junk_filter_type+'_allow_alphanum']
    word_is_alphanum = (lambda word: word.isalnum() if lemmas_allow_alphanum else lambda x: False)

    lemmas_allow_acronyms = pipeline_automator.parameters[junk_filter_type+'_allow_acronyms']
    word_is_acronym = (lambda word: word.isalnum() if lemmas_allow_acronyms else lambda x: False)

    lemmas_allow_proper_nouns = pipeline_automator.parameters[junk_filter_type+'_allow_proper_nouns']
    word_is_proper_noun = (lambda word: word in proper_nouns if lemmas_allow_proper_nouns else lambda x: False)

    # These are the higher level boolean functions:
    meets_the_are_requirements = (lambda word: word_is_alpha(word) and word_is_dict(word) and word_is_not_stop(word))

    meets_the_allow_requirements = lambda word: word_is_num(word) or word_is_alphanum(word) or word_is_acronym(word) or word_is_proper_noun(word)
    
    word_is_not_junk = lambda word: meets_the_are_requirements(word) and meets_the_allow_requirements(word)

    return word_is_not_junk(w)

def remove_junk(pos_tokens, pipeline_automator):
    # Iterate through the word,pos tuples and decide if it meets the conditions of not being junk.
    words_to_keep = []
    for word, pos in pos_tokens:
        if word_is_not_junk(word, pipeline_automator, junk_filter_type='lemmas'): # include these words
            words_to_keep.append((word,pos))
        else:
            excluded_words_list.add(word)
    return words_to_keep

def lemmatize(pos_tokens, pipeline_automator):
    wnl = WordNetLemmatizer()

    lemmatize_unigrams = pipeline_automator.parameters['lemmatize_unigrams']

    if lemmatize_unigrams:
        rtn = [wnl.lemmatize(word, pos) if pos != 'o' else wnl.lemmatize(word, 'n') for word, pos in pos_tokens]
    else:
        rtn = [word for word, pos in pos_tokens]
    
    return ' '.join(rtn)