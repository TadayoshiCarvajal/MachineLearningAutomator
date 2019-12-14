def has_callables(iterable):
    """
    A function used to determine if an iterable contains at least one callable.
    """
    for item in iterable:
        if callable(item):
            return True
    return False

def write_data_to_csv(file_name, data):
    """
    Writes a csv file from a list of lists.
    """
    import csv

    with open(file_name, 'w') as csvfile:
        for row in data:
            row = [str(item) for item in row]
            csvfile.write(",".join(row) + "\n")

def get_time_string(seconds):
    remaining_seconds = str(round(seconds % 60,1))+'s'
    minutes = int(seconds // 60)
    remaining_minutes = str(minutes % 60)+'m '
    hours = str(minutes // 60) + 'h '
    if hours != '0h ':
        rtn = hours + remaining_minutes + remaining_seconds
    else:
        if remaining_minutes != '0m ':
            rtn = remaining_minutes + remaining_seconds
        else:
            rtn = remaining_seconds

    return rtn

def split_data_ignore_include(data, test_size = 0.2, random_state = None, type_='stratified'):
    """
    A function used to split the data up in to a test set and a train set.
    There are two ways of doing this which is controlled by the type_ paramater:
    
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
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit

    #np.random.seed(random_state)

    if type_ == 'random':
        train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_set, test_set

    elif type_ == 'stratified':
        data = np.array(data)
        n_features = data.shape[1] - 1 # n_features = n_cols - 1, (last column is the target column)
        X, y = data[:,:n_features], data[:,n_features]

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
        return train_sets[0], test_sets[0]