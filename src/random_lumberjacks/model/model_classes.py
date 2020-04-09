import pandas as pd
import numpy as np
import math
from copy import copy, deepcopy
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

#Incomplete function to handle models/grid search.
class ModelSwitcher(object):
    def __init__(self, data, duplicate=False):
        self.duplicate = duplicate
        self._instantiate_data(data)

    def _instantiate_data(self, data):
        if not self.duplicate:
            self.data = data
        else:
            self.data = deepcopy(data)

#An object that manages the workflow for dummy variables, transforming features, polynomial features, class balancing,
# and scaling data.
class DataPreprocessor(object):
    def __init__(self, df, target, cat_features={}, cont_features={}, poly_features={}, create_dummies=False,
                 scale_dummies=False):
        self.df = df
        self.create_dummies = create_dummies
        self.scale_dummies = scale_dummies
        self._set_features(target, cat_features, cont_features, poly_features)
        self.X = pd.concat([df[self.cols.drop(labels=self.cols_generated_dummies)], self.dummies], axis=1)
        self.y = df[target]

    #Creates various attributes storing column names from specifically structured dictionaries for the
    # categorical and continuous variables.
    def _set_features(self, target, cat_features, cont_features, poly_features):
        self._parse_poly_dict(poly_features)
        self._get_cat_features(cat_features)
        self._get_cont_features(cont_features)
        self.cols_initial = self.cols_continuous.union(self.cols_categorical, sort=False)
        self.cols = self.cols_initial
        self.target = target

    #Gathers categorical column name information and creates corresponding attributes.
    def _get_cat_features(self, feature_dict):
        self.cols_nominal = self._get_indiv_feature(feature_dict, "nominal_features")
        self.cols_standard_dummies = self._get_indiv_feature(feature_dict, "standard_dummies")
        self.cols_impute_dummies = self._get_indiv_feature(feature_dict, "impute_dummies")
        if self.create_dummies:
            self._generate_dummies()
            self.cols_nominal = pd.Index([])
        else:
            self.dummies = pd.DataFrame()
            self.cols_generated_dummies = pd.Index([])
        self.cols_dummies = self.cols_standard_dummies.union(self.cols_impute_dummies, sort=False)
        self.cols_dummies = self.cols_dummies.union(self.cols_generated_dummies, sort=False)
        self.cols_categorical = self.cols_nominal.union(self.cols_dummies, sort=False)

    # Gathers continuous column name information and creates corresponding attributes, calling transformation functions if specified.
    def _get_cont_features(self, feature_dict):
        transformed_dict = self._get_feature_group(feature_dict, "transformed")
        self.cols_linear = self._get_indiv_feature(feature_dict, "untransformed")
        if transformed_dict:
            self._get_trans_features(transformed_dict)
            self.cols_continuous = self.cols_linear.union(self.cols_transformed, sort=False)
        else:
            self.cols_transformed = pd.Index([])
            self.cols_continuous = self.cols_linear

    #Gathers transformed features in the dictionary for continous features. New transformed columns are performed for whatever
    # transformations are specified.
    def _get_trans_features(self, transformed_dict):
        logged = self._get_feature_group(transformed_dict, "logged")
        pow = self._get_feature_group(transformed_dict, "exp")
        self.cols_logged = pd.Index([])
        if logged:
            self._log_features(logged)
        else:
            pass
        self.cols_transformed = self.cols_logged

    #Checks the for the existence of a key in a nested dictionary.
    def _get_feature_group(self, feature_dict, key):
        return feature_dict.get(key)

    #Checks a specific category of features being present in the passed dictionary returning an empty list if no results.
    def _get_indiv_feature(self, feature_dict, key, default=[]):
        return pd.Index(feature_dict.get(key, default))

    #Performs log transformations and gathers the new column names.
    def _log_features(self, logged_dict):
        for column, base in logged_dict.items():
            if base:
                new_col_name = f"{column}_log_b{base}"
            else:
                new_col_name = f"{column}_ln"
            self.df[new_col_name] = self.df[column].map(lambda x: math.log(x, base) if base else math.log(x))
            self._manage_poly_renames(column, pd.Index([new_col_name]))
            self.cols_logged = self.cols_logged.append(pd.Index([new_col_name]))

    # Generates dummy variables from a list of categorical columns.
    def _generate_dummies(self):
        self.dummies = pd.DataFrame()
        print("Creating Dummies")
        for column in self.cols_nominal:
            unique = self.df[column].unique().size
            if unique > 20:
                print(f"Warning: {column} has {unique} unique values")
            self.df[column] = self.df[column].astype('category')
            new_dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
            self._manage_poly_renames(column, new_dummies.columns)
            self.dummies = pd.concat([self.dummies, new_dummies], axis=1)
        self.cols_generated_dummies = self.dummies.columns

    def _manage_poly_renames(self, original, replacements):

        if original in self.polynomial["columns"]:
            self.polynomial["columns"] = self.polynomial["columns"].drop(labels=[original])
            self.polynomial["columns"] = self.polynomial["columns"].union(replacements, sort=False)

    def _parse_poly_dict(self, poly_dict):

        self.polynomial = {"method":poly_dict.get("method", "all")}
        self.polynomial["columns"] = pd.Index(poly_dict.get("columns", []))

    #Performs a train_test split.
    def _train_test_split(self):
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    #Creates a scaler attribute fit to the data loaded in the object.
    def _fit_scale(self):
        self._choose_scaled_columns()
        if self.scale_type == "standard":
            print("Using standard scaler")
            self.scaler = StandardScaler()
        elif self.scale_type == "minmax":
            print("Using Min/Max scaler")
            self.scaler = MinMaxScaler()
        else:
            print("No scaling specified")
            self.scale_type = False
            return
        self.scaler.fit(self.X_train[self.cols_scaled])

    #Rescales the data with the saved scalar attribute.
    def _rescale(self):
        if self.scale_type == False:
            print("Skipping scaling")
            return
        else:
            columns = self.cols_scaled

        #Overwrites original train/test dataframes to prevent linkage errors.
        self.X_test, self.X_train = self.X_test.copy(), self.X_train.copy()

        #Calls the transform function on both the train and test data targeting only the relevant columns.
        X_train = self._transform_scale(self.X_train)
        X_test = self._transform_scale(self.X_test)
        self.X_train[columns] = X_train[columns]
        self.X_test[columns] = X_test[columns]

    #Executes the scaling but preserves the index and returns it to a DataFrame.
    def _transform_scale(self, data):
        to_transform = data[self.cols_scaled]
        indices = to_transform.index
        scaled = self.scaler.transform(to_transform)
        return pd.DataFrame(scaled, columns=self.cols_scaled, index=indices)

    #Performs class balancing using the algorithm indicated in the arguments.
    def _class_imbalance(self):
        df = pd.concat([self.X_train, self.y_train], axis=1)
        if self.balance_class == "upsample":
            print("Performing upsample")
            self._simple_resample(df)
        elif self.balance_class == "downsample":
            print("Performing downsample")
            self._simple_resample(df, down=True)
        elif self.balance_class == "smote":
            print("Performing SMOTE")
            self._smote_data()
        elif self.balance_class == "tomek":
            print("Performing Tomek Links")
            self._tomek_data()
        else:
            print("Skipping class imbalance functions")

    #Performs a random choice to upsample/downsample all values to those with the maximum or minimum counts.
    def _simple_resample(self, df, down=False):
        target = self.target
        groups = [item for item in df[target].unique()]
        counts = {group: df[df[target] == group][target].count() for group in groups}
        most, least = max(counts, key=counts.get), min(counts, key=counts.get)
        if down == False:
            goal, samples = most, counts[most]
        else:
            goal, samples = least, counts[least]
        sample_queue = [remaining for remaining in groups if remaining != goal]
        new_df = df[df[target]==goal]
        for sample in sample_queue:
            current = df[df[target]==sample]
            resampled = resample(current, replace=True, n_samples=samples, random_state=self.random_state)
            new_df = pd.concat([new_df, resampled])
        self.X_train, self.y_train = new_df.drop(self.target, axis=1), new_df[self.target]

    #Performs a SMOTE upsampling of the data. If there are nominal columns detected, it will change SMOTE algorithms.
    def _smote_data(self):
        if self.cols_nominal.size > 0:
            cats = self.X_train.columns.isin(self.cols_nominal)
            sm = SMOTENC(categorical_features=cats, sampling_strategy='not majority', random_state=self.random_state)
        else:
            sm = SMOTE(sampling_strategy='not majority', random_state=self.random_state)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    #Performs tomek links. Can not handle nominal values.
    def _tomek_data(self):
        if self.cols_nominal.size > 0:
            print("Skipping Tomek Links. Cannot perform with raw categorical data. Create dummies to use.")
            return
        tl = TomekLinks()
        self.X_train, self.y_train = tl.fit_sample(self.X_train, self.y_train)

    #Creates polynomial features and creates a selection of those columns.
    def _poly_features(self):
        if type(self.poly_degree) == int:
            print(f"Getting polynomial features of degree {self.poly_degree}")
            orig_columns = self._choose_poly_columns()
            X_cont = self.X[orig_columns]
            X_cont_index = X_cont.index
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_cont)
            columns = pd.Index(poly.get_feature_names(X_cont.columns))
            poly_df = pd.DataFrame(X_poly, index=X_cont_index, columns=columns)
            self.cols_polynomial = columns.drop(labels=orig_columns)
            self.X = pd.concat([self.X[self.cols_initial], poly_df[self.cols_polynomial]], axis=1)
            self.cols = self.cols_initial.union(self.cols_polynomial, sort=False)
        else:
            print("Skipping polynomial features")
            self.poly_degree = False
            self.cols_polynomial = pd.Index([])
            self.X = self.X[self.cols_initial]

    #Creates a column list for polynomial features including or excluding dummy variables and transformed features depending
    # on arguments.
    def _choose_poly_columns(self):

        baseline = self.cols_initial.drop(labels=self.cols_nominal)
        sel = self.polynomial["columns"]
        method = self.polynomial["method"]
        if method == "choose":
            mask = baseline.isin(sel)
        elif method == "eliminate":
            mask = baseline.isin(sel) == False
        elif method == "linear":
            mask = baseline.isin(self.cols_linear)
        elif method == "no_dummies":
            mask = baseline.isin(self.cols_continuous)
        elif method == "no_transformed":
            mask = baseline.isin(self.cols_linear.union(self.cols_dummies, sort=False))
        else:
            mask = np.full(baseline.size, True)
        columns = baseline[mask]
        return columns

    #Determines whether or not dummy variables will be scaled based on an initialization argument.
    def _choose_scaled_columns(self):
        if self.scale_dummies:
            self.cols_scaled = self.cols.drop(labels=self.cols_nominal)
        else:
            self.cols_scaled = self.cols.drop(labels=self.cols_categorical)

    #Chains the commands together for polynomial features, class balancing, and scaling.
    def data_preprocessing(self, balance_class=False, scale_type=False, poly_degree=False):
        self.random_state = 1
        self.test_size = .2
        self.poly_degree = poly_degree
        self.balance_class = balance_class
        self.scale_type = scale_type
        self._poly_features()
        self._train_test_split()
        self._class_imbalance()
        self._fit_scale()
        self._rescale()

    def column_drop(self, columns):
        """Drops unwanted fetures from the column selection"""

        self.cols = self.cols.drop(labels=columns)

#Prints the models, accuracy/f1 score.
def evaluate_model(model, X_test, y_test):
    """Tests a model for accuracy, f1 score, precision, and recall"""
    y_pred = model.predict(X_test)
    if y_test.unique().size > 2:
        f1 = f1_score(y_test, y_pred, average='micro')
    else:
        f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
