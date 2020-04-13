import functools
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


class ModelSwitcher(object):
    """Incomplete function to handle models/grid search."""

    def __init__(self, data, duplicate=False):
        self.duplicate = duplicate
        self._instantiate_data(data)

    def _instantiate_data(self, data):
        if not self.duplicate:
            self.data = data
        else:
            self.data = deepcopy(data)


class DataPreprocessor(object):
    """An object that manages the workflow for dummy variables, transforming features, polynomial features, class balancing,
    and scaling data."""

    def __init__(self, df, target, cat_features={}, cont_features={}, poly_features={}, create_dummies=False,
                 scale_dummies=False, random_state=None, prediction_df=None):
        self.df = df.copy()
        self.make_predictions = self._check_predictions(prediction_df)
        if self.make_predictions:
            self.prediction_df = prediction_df.copy()
        else:
            self.prediction_df = prediction_df
        self.create_dummies = create_dummies
        self.scale_dummies = scale_dummies
        self.random_state = random_state
        self._set_features(target, cat_features, cont_features, poly_features)
        self.X = pd.concat([df[self.cols.drop(labels=self.cols_generated_dummies)], self.dummies], axis=1)
        if self.make_predictions:
            self.prediction_df = pd.concat([self.prediction_df[self.cols.drop(labels=self.cols_generated_dummies)], self.pred_dummies], axis=1)
        self.y = df[target]


    def _check_selection_mask(func):
        """Decorator to to ensure columns from the dropped list to be removed when the function is called."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.cols_dropped.size > 0:
                self.selection = self.cols[self.col_mask]
            return func(self, *args, **kwargs)
        return wrapper

    def _check_predictions(self, prediction_df):
        """Tests for the presence of an additional df for predictions"""

        return type(prediction_df) is not type(None)

    def _set_features(self, target, cat_features, cont_features, poly_features):
        """Creates various attributes storing column names from specifically structured dictionaries for the
        categorical and continuous variables."""

        self._parse_poly_dict(poly_features)
        self._get_cat_features(cat_features)
        self._get_cont_features(cont_features)
        self.cols_initial = self.cols_continuous.union(self.cols_categorical, sort=False)
        self.cols = self.cols_initial
        self.selection = self.cols
        self.cols_dropped = pd.Index([])
        self.target = target

    def _get_cat_features(self, feature_dict):
        """Gathers categorical column name information and creates corresponding attributes."""

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

    def _get_cont_features(self, feature_dict):
        """Gathers continuous column name information and creates corresponding attributes, calling
        transformation functions if specified."""

        transformed_dict = self._get_feature_group(feature_dict, "transformed")
        self.cols_linear = self._get_indiv_feature(feature_dict, "untransformed")
        if transformed_dict:
            self._get_trans_features(transformed_dict)
            self.cols_continuous = self.cols_linear.union(self.cols_transformed, sort=False)
        else:
            self.cols_transformed = pd.Index([])
            self.cols_continuous = self.cols_linear

    def _get_trans_features(self, transformed_dict):
        """Gathers transformed features in the dictionary for continous features. New transformed columns are performed for whatever
        ransformations are specified."""

        logged = self._get_feature_group(transformed_dict, "logged")
        pow = self._get_feature_group(transformed_dict, "exp")
        self.cols_logged = pd.Index([])
        if logged:
            self._log_features(logged)
        else:
            pass
        self.cols_transformed = self.cols_logged

    def _get_feature_group(self, feature_dict, key):
        """Checks the for the existence of a key in a nested dictionary."""

        return feature_dict.get(key)

    def _get_indiv_feature(self, feature_dict, key, default=[]):
        """Checks a specific category of features being present in the passed dictionary returning an empty list if no results."""

        return pd.Index(feature_dict.get(key, default))

    def _log_features(self, logged_dict):
        """Performs log transformations and gathers the new column names."""

        for column, base in logged_dict.items():
            if base:
                new_col_name = f"{column}_log_b{base}"
            else:
                new_col_name = f"{column}_ln"
            self.df[new_col_name] = self.df[column].map(lambda x: math.log(x, base) if base else math.log(x))
            if self.make_predictions:
                self.prediction_df[new_col_name] = self.prediction_df[column].map(lambda x: math.log(x, base) if base else math.log(x))
            self._manage_poly_renames(column, pd.Index([new_col_name]))
            self.cols_logged = self.cols_logged.append(pd.Index([new_col_name]))

    def _generate_dummies(self):
        """Generates dummy variables from a list of categorical columns."""

        self.dummies, self.pred_dummies = pd.DataFrame(), pd.DataFrame()
        print("Creating Dummies")
        for column in self.cols_nominal:
            unique = self.df[column].unique().size
            if unique > 20:
                print(f"Warning: {column} has {unique} unique values")
            self.df[column] = self.df[column].astype('category')
            new_dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
            self._manage_poly_renames(column, new_dummies.columns)
            self.dummies = pd.concat([self.dummies, new_dummies], axis=1)
            if self.make_predictions:
                pred_dummies = pd.get_dummies(self.prediction_df[column], prefix=column, drop_first=True)
                self.pred_dummies = pd.concat([self.pred_dummies, pred_dummies], axis=1)
        self.cols_generated_dummies = self.dummies.columns

    def _manage_poly_renames(self, original, replacements):
        """Ensures that new columns generated with the model classer function via transformations or dummy variables
        are represented by their new column names in the dictionary of columns to have polynomial features."""

        if original in self.polynomial["columns"]:
            self.polynomial["columns"] = self.polynomial["columns"].drop(labels=[original])
            self.polynomial["columns"] = self.polynomial["columns"].union(replacements, sort=False)

    def _parse_poly_dict(self, poly_dict):
        """Creates an attribute that tracks which columns will be queued for polynomial features."""

        acceptable_methods = ["all", "choose", "eliminate", "linear", "no_dummies", "no_transformed"]
        method = poly_dict.get("method", None)
        if not method:
            print(f'Missing "method" key in polynomial selection dictionary. Assuming all values will be transformed')
        elif method not in acceptable_methods:
            print(f"{method} is not an acceptable value for the polynomial dict. Choose from {acceptable_methods}")
        self.polynomial = {"method":poly_dict.get("method", "all")}
        self.polynomial["columns"] = pd.Index(poly_dict.get("columns", []))

    def _train_test_split(self):
        """Performs a train_test split."""

        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def _fit_scale(self):
        """Creates a scaler attribute fit to the data loaded in the object."""

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

    def _rescale(self):
        """Rescales the data with the saved scalar attribute."""

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
        if self.make_predictions:
            predictions = self._transform_scale(self.prediction_df)
            self.prediction_df[columns] = predictions[columns]

    def _transform_scale(self, data):
        """Executes the scaling but preserves the index and returns it to a DataFrame."""

        to_transform = data[self.cols_scaled]
        indices = to_transform.index
        scaled = self.scaler.transform(to_transform)
        return pd.DataFrame(scaled, columns=self.cols_scaled, index=indices)

    def _class_imbalance(self):
        """Performs class balancing using the algorithm indicated in the arguments."""

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

    def _simple_resample(self, df, down=False):
        """Performs a random choice to upsample/downsample all values to those with the maximum or minimum counts."""

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

    def _smote_data(self):
        """Performs a SMOTE upsampling of the data. If there are nominal columns detected, it will change SMOTE algorithms."""

        if self.cols_nominal.size > 0:
            cats = self.X_train.columns.isin(self.cols_nominal)
            sm = SMOTENC(categorical_features=cats, sampling_strategy='not majority', random_state=self.random_state)
        else:
            sm = SMOTE(sampling_strategy='not majority', random_state=self.random_state)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    def _tomek_data(self):
        """Performs tomek links. Can not handle nominal values."""

        if self.cols_nominal.size > 0:
            print("Skipping Tomek Links. Cannot perform with raw categorical data. Create dummies to use.")
            return
        tl = TomekLinks()
        self.X_train, self.y_train = tl.fit_sample(self.X_train, self.y_train)

    def _poly_features(self):
        """Creates polynomial features and creates a selection of those columns."""

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
            if self.make_predictions:
                pred_cont = self.prediction_df[orig_columns]
                pred_cont_index = pred_cont.index
                pred_poly = poly.fit_transform(pred_cont)
                pred_poly_df = pd.DataFrame(pred_poly, index=pred_cont_index, columns=columns)
                self.prediction_df = pd.concat([self.prediction_df[self.cols_initial], pred_poly_df[self.cols_polynomial]], axis=1)
        else:
            print("Skipping polynomial features")
            self.poly_degree = False
            self.cols_polynomial = pd.Index([])
            self.X = self.X[self.cols_initial]
            if self.make_predictions:
                self.prediction_df = self.prediction_df[self.cols_initial]

    def _choose_poly_columns(self):
        """Creates a column list for polynomial features including or excluding dummy variables and transformed features depending
        on arguments."""

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

    def _choose_scaled_columns(self):
        """Determines whether or not dummy variables will be scaled based on an initialization argument."""

        if self.scale_dummies:
            self.cols_scaled = self.cols.drop(labels=self.cols_nominal)
        else:
            self.cols_scaled = self.cols.drop(labels=self.cols_categorical)

    def column_drop(self, columns, reverse=False):
        """Drops unwanted fetures from the column selection"""

        # Checks the reversed argument and sets the appropriate variables on whether the selction should be
        # expanded or contracted.
        if reverse:
            function, description = self.cols_dropped.drop, "Removing"
        else:
            function, description = self.cols_dropped.union, "Adding"

        if type(columns) == str:
            columns = [columns]
        columns = pd.Index(columns)
        n_cols = columns.size

        mask = self.cols.isin(columns)
        dropped = self.cols[mask]

        #Does a quick test if columns are an exact match, printing anything missing.
        if mask.sum() != n_cols:
            missing = np.setdiff1d(columns, self.cols).tolist()
            print(f"The columns: {missing} are either missing or have not been entered in correctly.")
        else:
            print(f"{description} {dropped.to_numpy().tolist()}")

        self.cols_dropped = function(dropped)
        self.col_mask = self.cols.isin(self.cols_dropped) == False

    def data_preprocessing(self, balance_class=False, scale_type=False, poly_degree=False, test_size=.2):
        """Chains the commands together for polynomial features, class balancing, and scaling."""

        self.test_size = test_size
        self.poly_degree = poly_degree
        self.balance_class = balance_class
        self.scale_type = scale_type
        self._poly_features()
        self._train_test_split()
        self._class_imbalance()
        self._fit_scale()
        self._rescale()
        self.selection = self.cols

    @_check_selection_mask
    def get_X_train(self):
        """Gets the X train data with updated column selection"""

        return self.X_train[self.selection]

    @_check_selection_mask
    def get_X_test(self):
        """Gets the X test data with updated column selection"""

        return self.X_test[self.selection]

    @_check_selection_mask
    def get_df(self):
        """recombines the entire dataset but excludes features outside of the updated column selection"""

        X = pd.concat([self.X_train[self.selection], self.X_test[self.selection]], axis = 0)
        y = pd.concat([self.y_train, self.y_test], axis=0)
        return pd.concat([X, y], axis=1)

    @_check_selection_mask
    def get_predictions(self):
        """Gets the X test data with updated column selection"""

        return self.prediction_df[self.selection]


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
    print("Params:")
    print(model.get_params())