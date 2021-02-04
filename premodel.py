from __future__ import division
#import sys
import os
import csv
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
# Import datasets, classifiers and performance metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from math import ceil
import util
from threading import Thread
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NearMiss
#-------------------------------
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


IMG_SIZE = 228
Percentage_TESTED = 10
#DATA_FILE = 'all_image_features.csv'
#DATA_FILE = 'all_image_features_norm.csv'
##DATA_FILE = 'top-1-few-features.csv'

class Premodel():
    def __init__(self, top_n):
        if top_n == 'Top-1':
            self.DATA_FILE = 'all_new_features_hier_norm_top_1.csv'
        elif top_n == 'Top-5':
            self.DATA_FILE = 'all_new_features_hier_norm_top_5.csv'
        else:
            print("String unrecognized, preference set to top-1")
            self.DATA_FILE = 'all_new_features_hier_norm_top_1.csv'

        PATH_TO_FILES = '/images/val/images'

        list_machines = (
            'nn',
            'dt16',
            'vc'
        )


    def machines_avialable():
        list_machines_available = []
        for first_level_machine in list_machines:
            for second_level_machine in list_machines:
                for third_level_machine in list_machines:
                    list_machines_available.append(str(first_level_machine)+" - " +str(second_level_machine)+" - "+str(third_level_machine))

        return list_machines_available


    def compare_array(self, list1, list2):
        """
        It compares to arrays and returns how many times they are equal
        """
        success = 0
        for position, number in enumerate(list1):
            if number == list2[position]:
                success = success + 1

        return success, len(list1)-success


    def cv_training_data (self, amount_images):
        """
        This functions returns the data that will be used for training and test different machine learning models.
        This information is collected from the file DATA_FILE.
        Return:
            data: Data for training the models
            data_result: data for validation
        """

        data = []
        first_level = []
        second_level = []
        third_level = []

        # Getting the images for training and testing
        row_count = 0
        with open(self.DATA_FILE, 'rb') as csvfile:
            lines = [line.decode('utf-8-sig') for line in csvfile]

            for row in csv.reader(lines):
                # Remove the headers of csv file
                if row_count is 0:
                    row_count = row_count + 1
                    continue

                data.append(row[-7:])                   # changes what features will be used in the premodel. In this array features start at 4th index, and end at the 10th
                first_level.append((row[0],row[1]))     # performance of the first level machine
                second_level.append((row[0],row[2]))    # performance of the second level machine
                third_level.append((row[0],row[3]))     # performance of the third level machine
                row_count = row_count + 1
                if row_count > amount_images:
                    break

        return  data, first_level, second_level, third_level

    # Nearest Neighbours model - TRAINING and PREDICTION
    def nearest_neighbour(self, X_train, X_test, Y_test):
        """
        Nearest neighbour function that returns the prediction of a list of images. With K = 5
        Args:
            X_train: List of images features used for training
            X_test: List of images results used for validate the trained images.
            Y_train: List of images features predicted
        """
        # Create and fit a nearest-neighbor classifier
        knn = KNeighborsClassifier()
        knn.fit(X_train, X_test)
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

        # Prediction
        predicted = knn.predict(Y_test)



        #(predicted.shape)
        return predicted

    def logistic_regression(self, X_train, X_test, Y_test):
        """
        Logistic regression function that returns the prediction of a list of images.
        Args:
            X_train: List of images features used for training
            X_test: List of images results used for validate the trained images.
            Y_train: List of images features predicted
        """
        # Create and fit a nearest-neighbor classifier
        log_reg = LogisticRegression()
        #log_reg = LogisticRegression(class_weight = 'balanced')
        log_reg.fit(X_train, X_test)
        # Prediction
        predicted = log_reg.predict(Y_test)
        predicted = predicted.astype(int)
        importance = model.coef_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        return predicted

    # Decision tree of level 2, 5,8,12 and 16 - TRAINING and PREDICTION
    def decision_tree(self, X_train, X_test, Y_train):
        """
        Decision Tree function that returns the prediction of a list of images. This function allows different deepth levels: 2,5,8,12 and 16
        Args:
            X_train: List of images features used for training
            X_test: List of images results used for validate the trained images.
            Y_train: List of images features predicted
        """

        # Create tree
        regr_2 = DecisionTreeRegressor(max_depth=2)
        regr_5 = DecisionTreeRegressor(max_depth=5)
        regr_8 = DecisionTreeRegressor(max_depth=8)
        regr_12 = DecisionTreeRegressor(max_depth=12)
        regr_16 = DecisionTreeRegressor(max_depth=16)

        # Fit tree
        regr_2.fit(X_train, X_test)
        regr_5.fit(X_train, X_test)
        regr_8.fit(X_train, X_test)
        regr_12.fit(X_train, X_test)
        regr_16.fit(X_train, X_test)

        # Predict
        predicted_level_2 = regr_2.predict(Y_train)
        predicted_level_5 = regr_5.predict(Y_train)
        predicted_level_8 = regr_8.predict(Y_train)
        predicted_level_12 = regr_12.predict(Y_train)
        predicted_level_16 = regr_16.predict(Y_train)

        return predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16

    # A support vector classifier model - TRAINING and PREDICTION
    def vecto_classifier(self, X_train, X_test, Y_train):
        """
        Vector Classification function that returns the prediction of a list of images. With gamma 0.001
        Args:
            X_train: List of images features used for training
            X_test: List of images results used for validate the trained images.
            Y_train: List of images features predicted
        """

        #Create a classifier: a support vector classifier
        classifier = svm.SVC(gamma=0.001)

        # We learn the digits on the first half of the digits
        classifier.fit(X_train, X_test)

        # Now predict the value of the digit on the second half:
        predicted = classifier.predict(Y_train)

        return predicted

    def CV_fold_worker(self, test_idx, train_idx, img_data, first_level, second_level, third_level, first_level_machine, second_level_machine, third_level_machine, return_wrapper):
        """
        Worker function for each fold in CV. Trains a model with training data, tests with
        test_idx. Places the results as (image, prediction) tuples in return wrapper
        Args:
            test_idx: List if indexes where the test_data is
            train_idx: List if indexes where the train_data is
            img_data: all of the image data
            first_level: The names of the classes, respective to model return
            return_wrapper: The list to add all results
        """
        # Create a validation set which is 10% of the training_data
        X_train, _ = util.list_split(img_data, train_idx, [0])
        X_train_first_level = X_train
        X_train_second_level = X_train
        X_train_third_level = X_train


        Y_train, _ = util.list_split(img_data, test_idx, [0])
        Y_test_first_level, _ = util.list_split(first_level, test_idx, [0])
        Y_test_second_level, _ = util.list_split(second_level, test_idx, [0])
        Y_test_third_level, _ = util.list_split(third_level, test_idx, [0])

        X_test_first_level, _ = util.list_split(first_level, train_idx, [0])
        X_test_second_level, _ = util.list_split(second_level, train_idx, [0])
        X_test_third_level, _ = util.list_split(third_level, train_idx, [0])

        X_val_first_level = [X_test_first_level[i][1] for i in range(0,len(X_test_first_level))]
        Y_val_first_level = [Y_test_first_level[i][1] for i in range(0,len(Y_test_first_level))]

        X_val_second_level = [X_test_second_level[i][1] for i in range(0,len(X_test_second_level))]
        Y_val_second_level = [Y_test_second_level[i][1] for i in range(0,len(Y_test_second_level))]

        X_val_third_level = [X_test_third_level[i][1] for i in range(0,len(X_test_third_level))]
        Y_val_third_level = [Y_test_third_level[i][1] for i in range(0,len(Y_test_third_level))]

        for i in range(len(X_train_first_level)):
            for j in range(len(X_train_first_level[i])):
                X_train_first_level[i][j] = float(X_train_first_level[i][j])
        for i in range(len(X_val_first_level)):
            X_val_first_level[i] = int(X_val_first_level[i])


        for i in range(len(X_train_second_level)):
            for j in range(len(X_train_second_level[i])):
                X_train_second_level[i][j] = float(X_train_second_level[i][j])
        for i in range(len(X_val_second_level)):
            X_val_second_level[i] = int(X_val_second_level[i])



        for i in range(len(X_train_third_level)):
            for j in range(len(X_train_third_level[i])):
                X_train_third_level[i][j] = float(X_train_third_level[i][j])
        for i in range(len(X_val_third_level)):
            X_val_third_level[i] = int(X_val_third_level[i])


        X_train_first_level = np.array(X_train_first_level)
        X_train_second_level = np.array(X_train_second_level)
        X_train_third_level = np.array(X_train_third_level)
        X_train_first_level = X_train_first_level.astype('float64')
        X_train_second_level = X_train_second_level.astype('float64')
        X_train_third_level = X_train_third_level.astype('float64')
        Y_train = np.array(Y_train)


        X_val_first_level = np.array(X_val_first_level)
        X_val_second_level = np.array(X_val_second_level)
        X_val_third_level = np.array(X_val_third_level)
        X_val_first_level = X_val_first_level.astype('int')
        X_val_second_level = X_val_second_level.astype('int')
        X_val_third_level = X_val_third_level.astype('int')



        #oversample = SMOTE()
        #X_train_first_level, X_val_first_level = oversample.fit_resample(X_train_first_level, X_val_first_level)


        #undersample = CondensedNearestNeighbour(n_neighbors=1)
        #undersample = TomekLinks()
        #undersample = EditedNearestNeighbours(n_neighbors=3)
        #undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
        undersample = NearMiss(version=1, n_neighbors=3)
    # transform the dataset
        X_train_first_level, X_val_first_level = undersample.fit_resample(X_train_first_level, X_val_first_level)

        X_train_second_level, X_val_second_level = undersample.fit_resample(X_train_second_level, X_val_second_level)

        X_train_third_level, X_val_third_level = undersample.fit_resample(X_train_third_level, X_val_third_level)



        list_predictions = []
        Y_train_second_level = []
        Y_train_second_level_position = []
        Y_train_third_level = []
        Y_train_third_level_position = []

        ##################################################################################################################
        # First Level of hierarchy [Mobilnet_v1]
        ##################################################################################################################
        if first_level_machine == 'nn':
            predicted = self.nearest_neighbour(X_train_first_level, X_val_first_level, Y_train)
            predicted = predicted.tolist()
        elif first_level_machine == 'log_reg':
            predicted = self.logistic_regression(X_train_first_level, X_val_first_level, Y_train)
        elif first_level_machine == 'dt16':
            predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = self.decision_tree(X_train_first_level, X_val_first_level, Y_train)
            predicted = predicted_level_16
        elif first_level_machine == 'vc':
            predicted = self.vecto_classifier(X_train_first_level, X_val_first_level, Y_train)
        elif first_level_machine == 'nb':
            predicted = self.naive_bayes(X_train_first_level, X_val_first_level, Y_train)

        for position, prediction in enumerate(predicted):
            if first_level_machine == 'dt16':
                if prediction > 0.5:
                    if Y_test_first_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                    else:
                        list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    Y_train_second_level.append(Y_train[position])
                    Y_train_second_level_position.append(position)
            else:
                if prediction == 1:
                    if Y_test_first_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                    else:
                        list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    Y_train_second_level.append(Y_train[position])
                    Y_train_second_level_position.append(position)

        # Not necessary to go to the next level
        if len(Y_train_second_level) == 0:
            return_wrapper.append(list_predictions)
            return

        ##################################################################################################################
        # Second Level of hierarchy [Inception_v4]
        ##################################################################################################################
        #predicted = nearest_neighbour(X_train, X_val_second_level, Y_train_second_level)
        #predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val, Y_train)
        #predicted = predicted_level_16
        #predicted = vecto_classifier(X_train, X_val, Y_train)

        if second_level_machine == 'nn':
            predicted = self.nearest_neighbour(X_train_second_level, X_val_second_level, Y_train_second_level)
        elif second_level_machine == 'log_reg':
            predicted = self.logistic_regression(X_train_second_level, X_val_second_level, Y_train_second_level)
        elif second_level_machine == 'dt16':
            predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = self.decision_tree(X_train_second_level, X_val_second_level, Y_train_second_level)
            predicted = predicted_level_16
        elif second_level_machine == 'vc':
            predicted = self.vecto_classifier(X_train_second_level, X_val_second_level, Y_train_second_level)
        elif second_level_machine == 'nb':
            predicted = self.naive_bayes(X_train_second_level, X_val_second_level, Y_train_second_level)

        for position, prediction in enumerate(predicted):
            if second_level_machine == 'dt16':
                if prediction > 0.5:
                    if Y_test_second_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2, prediction, 2, 'tf-inception_v4'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0, prediction, 2, 'tf-inception_v4'))
                else:
                    Y_train_third_level.append(Y_train_second_level[position])
                    Y_train_third_level_position.append(Y_train_second_level_position[position])
            else:
                if prediction == 1:
                    if Y_test_second_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2, prediction, 2, 'tf-inception_v4'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0, prediction, 2, 'tf-inception_v4'))
                else:
                    Y_train_third_level.append(Y_train_second_level[position])
                    Y_train_third_level_position.append(Y_train_second_level_position[position])

        if len(Y_train_third_level) == 0:
            return_wrapper.append(list_predictions)
            return

        ##################################################################################################################
        # Third Level of hierarchy [Resnet_v1_152]
        ##################################################################################################################
        #predicted = nearest_neighbour(X_train, X_val_third_level, Y_train_third_level)
        #predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_third_level, Y_train_third_level)
        #predicted = predicted_level_16
        #predicted = vecto_classifier(X_train, X_val, Y_train)

        if third_level_machine == 'nn':
            predicted = self.nearest_neighbour(X_train_third_level, X_val_third_level, Y_train_third_level)
        elif third_level_machine == 'log_reg':
            predicted = self.logistic_regression(X_train_third_level, X_val_third_level, Y_train_third_level)
        elif third_level_machine == 'dt16':
            predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = self.decision_tree(X_train_third_level, X_val_third_level, Y_train_third_level)
            predicted = predicted_level_16
        elif third_level_machine == 'vc':
            predicted = self.vecto_classifier(X_train_third_level, X_val_third_level, Y_train_third_level)
        elif third_level_machine == 'nb':
            predicted = self.naive_bayes(X_train_third_level, X_val_third_level, Y_train_third_level)

        for position, prediction in enumerate(predicted):
            if third_level_machine == 'dt16':
                if prediction > 0.5:
                    if Y_test_third_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 3, 'tf-resnet_v1_152'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 3, 'tf-resnet_v1_152'))
                else:
                    if Y_test_third_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 0, 'failed'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 0, 'failed'))
            else:
                if prediction == 1:
                    if Y_test_third_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 3, 'tf-resnet_v1_152'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 3, 'tf-resnet_v1_152'))
                else:
                    if Y_test_third_level[position][1] == 1:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 0, 'failed'))
                    else:
                        list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 0, 'failed'))


        return_wrapper.append(list_predictions)

    def prototype(self, amount_images, list_premodels):
        """
        Produce a .csv file with the fields <Image_filename, Ground truth model, predicted model>
        for every image in the train information set. We use k-fold cross validation, where k=10.
        """
        percentage_results = []
        report_results = []

        if len(list_premodels) == 0:
            print("No premodels were selected!")
            return percentage_results
        if amount_images == 0:
            print("No images were selected!")
            return percentage_results

        #print("Creating training data...")
        data, first_level_data, second_level_data, third_level_data = self.cv_training_data(amount_images)

        for counter,(first_level_machine, second_level_machine, third_level_machine) in enumerate(list_premodels):
            # Split training data in k-fold chunks
            # Minimum needs to be 2
            k_fold = 10
            worker_threads = list()
            chunk_size = int(ceil(len(data) / float(k_fold)))
            # Create a new thread for each fold
            for i, (test_idx, train_idx) in enumerate(util.chunkise(range(len(data)), chunk_size)):
                return_wrapper = list()
                p = Thread(target=self.CV_fold_worker, args=(test_idx, train_idx, data, first_level_data, second_level_data, third_level_data, first_level_machine, second_level_machine, third_level_machine, return_wrapper))
                p.start()
                worker_threads.append((p, return_wrapper))


            # Wait for threads to finish, collect results
            all_predictions = list()
            for p, ret_val in worker_threads:
                p.join()
                all_predictions += ret_val

            predicted = []
            correct_result = []

            for p in all_predictions:
                for image, groundtruth_label, result_prediction, prediction, model_predicted in p:
                    correct_result.append(groundtruth_label)
                    predicted.append(prediction)

            percentage_results.append(accuracy_score(predicted, correct_result, [list_premodels[counter]]))
            report_results.append(precision_recall_fscore_support(correct_result, predicted, labels = [0, 1]))
        return_wrapper = list()
        preds = list()
        for counter,(first_level_machine, second_level_machine, third_level_machine) in enumerate(list_premodels):
            preds.append(self.CV_fold_worker(test_idx, train_idx, data, first_level_data, second_level_data, third_level_data, first_level_machine, second_level_machine, third_level_machine, return_wrapper))
        return percentage_results, report_results, predicted, correct_result, return_wrapper, worker_threads


    # def gaussian_process(self, X_train, X_test, Y_test):
    #     kernel = DotProduct() + WhiteKernel()
    #     gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_train, X_test)
    #     predicted = gpr.predict(Y_test)
    def naive_bayes(self, X_train, X_test, Y_train):
        # Create and fit a Naive Bayes classifier

        nb = GaussianNB()
        nb.fit(X_train, X_test)

        # Obtain predictions

        predicted = nb.predict(Y_train)
        #predicted = predicted.astype(int)

        return predicted

if __name__ == "__main__":
    cross_validation()
