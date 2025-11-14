import itertools

class KaggleDataHandler():
    def __init__(self):
        pass

    def create_split(self, target_test_folds):
        folds = [i for i in range(10)]
    
        target_test_fold_names = []
        for fold_number in target_test_folds:
            target_test_fold_names.append("fold"+str(fold_number))

        target_training_fold_names = []
        for fold_number in folds:
            if (fold_number in target_test_folds):
                continue
            target_training_fold_names.append("fold"+str(fold_number))

        #print("Test folds:", target_test_fold_names)
        #print("Training folds:", target_training_fold_names)
        return (target_test_fold_names, target_training_fold_names)

    def create_set(self, number_of_test_folds):
        combinations= []
        if number_of_test_folds > 1:
            combinations = list(itertools.combinations(range(10), number_of_test_folds))
        else:
            combinations = [i for i in range(10)]

        validations_set = []
        for combination in combinations:
            validations_set.append(self.create_split(combination))
        return validations_set

# EXAMPLE USE CASE
#KDHandler = KaggleDataHandler()
#number_of_test_folds = 2
#validation_sets = KDHandler.create_set(number_of_test_folds)
#for i, validation_set in enumerate(validation_sets):
#    print(f'#{i}: ',validation_set)

    

        
            

        


