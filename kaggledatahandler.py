import itertools
import os
import csv

class KaggleDataHandler():
    def __init__(self):
        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')
            self.__create_file_to_label_csv__()
        pass

    # Creating the different combinations based on the training vs testing split ratio
    def __create_split__(self, target_test_folds):
        folds = [i for i in range(10)]
    
        target_test_fold_names = []
        for fold_number in target_test_folds:
            target_test_fold_names.append("fold"+str(fold_number+1))

        target_training_fold_names = []
        for fold_number in folds:
            if (fold_number in target_test_folds):
                continue
            target_training_fold_names.append("fold"+str(fold_number+1))

        #print("Test folds:", target_test_fold_names)
        #print("Training folds:", target_training_fold_names)
        return (target_test_fold_names, target_training_fold_names)
    
    def __create_file_to_label_csv__(self):
        folds_name = []
        folds = [i for i in range(10)]
        for fold_number in folds:
            folds_name.append("fold"+str(fold_number+1))

        for i, fold_name in enumerate(folds_name):
            folder_path = f'dataset/{fold_name}'
            files = os.listdir(folder_path)
            file_paths = []
            with open(f'processed_data/{fold_name}.csv', 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['file_name', 'classID'])
                for file in files:
                    y_value = self.get_class_id(file)
                    writer.writerow([file, y_value])

    def create_splits(self, number_of_test_folds):
        combinations = []
        if number_of_test_folds > 1:
            combinations = list(itertools.combinations(range(10), number_of_test_folds))
        else:
            combinations = [i for i in range(10)]

        dataset_folds = []
        for combination in combinations:
            output = self.__create_split__(combination)
            dataset_folds.append(output)
        return dataset_folds

    # Creating datastructure that contains the filepath to the different .wav based on the combinations found
    def create_set(self):
        folds = [i for i in range(10)]
        dataset_folds = []
        for fold_number in folds:
            dataset_folds.append("fold"+str(fold_number+1))
        
        datasets_filepath_organized = {}
        y_datasets_filepath_organized = {}
        for fold in dataset_folds:
            folder_path = f'dataset/{fold}'
            files = os.listdir(folder_path)
            file_paths = []
            y_values = []
            for file in files:
                #print("Wait.....")
                file_path = folder_path+'/'+file
                file_paths.append(file_path)
                y_value = self.get_class_id_pre_data(fold,file)
                y_values.append(y_value)                                                                                    
                #print("Wait..")
            datasets_filepath_organized[fold] = file_paths
            y_datasets_filepath_organized[fold] = y_values
        return datasets_filepath_organized, y_datasets_filepath_organized
    

    def get_class_id(self, slice_file_name):
        csv_path = "dataset/UrbanSound8K.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["slice_file_name"] == slice_file_name:
                    return int(row["classID"])
        return None

    def get_class_id_pre_data(self, fold_name,slice_file_name):
        csv_path = f'processed_data/{fold_name}.csv'
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["file_name"] == slice_file_name:
                    return int(row["classID"])
        return None

# EXAMPLE USE CASE

"""
KDHandler = KaggleDataHandler()
number_of_test_folds = 2
datasets_filepath_organized, y_datasets_filepath_organized = KDHandler.create_set()
#KDHandler.create_file_to_label_csv()

print(datasets_filepath_organized.keys())
print(datasets_filepath_organized["fold1"])
"""