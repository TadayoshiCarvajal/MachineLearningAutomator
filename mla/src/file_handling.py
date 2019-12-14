class FileHandler:
    def __init__(self, 
            input_files_directory='input_files', 
            output_files_directory='models'):

        self.input_files_directory = input_files_directory
        self.output_files_directory = output_files_directory

    def get_data(self, file_name, feature_col, target_col, min_records=100):
        import pandas as pd
        from collections import Counter
        file_ext = file_name.split(".")[1]
        if file_ext == 'csv':
            ans = self.get_data_from_csv(file_name, feature_col, target_col)
        elif file_ext == 'xlsx':
            ans = self.get_data_from_xlsx(file_name, feature_col, target_col)
        else:
            raise ValueError('The file extension for', file_name, 'is not recognized. Only csv and xlsx are permitted.')
        
        labels = ans[target_col].values.tolist()
        label_counts = Counter(labels)
        labels_to_keep = set([label for label in label_counts if label_counts[label] >= min_records])
        if len(labels_to_keep) == 1:
            raise ValueError(f"After filter only labels with >= {min:d} records, only one class remained. Try decreasing min_records_per_class.")
        ans = ans.loc[ans[target_col].isin(labels_to_keep)]
        rtn = ans.values.tolist()
        return rtn

    def get_data_from_csv(self, file_name, feature_col, target_col):
        import pandas as pd
        data = pd.read_csv(self.input_files_directory+'/'+file_name)
        data = data[[feature_col, target_col]]
        return data
         
    def get_data_from_xlsx(self, file_name, feature_col, target_col):
        import pandas as pd
        data = pd.read_excel(self.input_files_directory+'/'+file_name)
        data = data[[feature_col, target_col]]
        return data
    
    def save_model(self, file_name, model):
        import pickle
        file = open(self.output_files_directory + '/' + file_name+'.p', 'wb')
        pickle.dump(model, file)
        print('Successfully Saved.')
        file.close()

    def load_model(self, file_name):
        import pickle
        file = open(file_name, 'rb')
        rtn = pickle.load(file)
        file.close()
        return rtn