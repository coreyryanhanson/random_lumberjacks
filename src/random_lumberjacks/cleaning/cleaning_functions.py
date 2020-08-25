import numpy as np
import pandas as pd


def full_value_comparison(df, column1, column2):
    """Iterates through unique values in a column, and calls the full value counts function on a second column
    to show the full representation of values on each divided section of the dataframe."""
    
    for value in df[column1].unique():
        slice = df[df[column1] == value]
        print(value)
        full_value_counts(slice, column2)
        print("")


def full_value_counts(df, column):
    """Shows the full breadth of possilbe values and nans for a column of a dataframe."""
    
    unique, total = df[column].unique().size, df[column].size
    totalna = df[column].isna().sum()
    percent_na = totalna/total
    print(f"There are {unique} unique values with {totalna} nan values making up {percent_na*100:.1f}%")
    for value, count in df[column].value_counts().iteritems():
        print(f"{count}-{value} --{count/total*100:.2f}%")


def print_full(x):
    """Dynamically changes the amount of max visible rows for a pandas object."""

    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')

