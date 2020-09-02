import numpy as np
import pandas as pd

def parse_fixed_noaa_data(bstring, params):
    """Used to loop through and capture the appropriate substrings from a specifically formatted parameter list"""
    
    return [bstring[param[1]:param[2]] for param in params]

def parse_optional_noaa_data(bstring, params):
    """Loops thrrough a specially formatted ordered dict of optional parameters capturing their values or nans
    depending on whether or not they are encountered."""
    
    results = []
    for param in params:
        sliced = extract_noaa_optional_str(bstring, param, params[param][0])
        if sliced:
            values = parse_fixed_noaa_data(sliced, params[param][1])
        else:
            values = [np.nan] * len(params[param][1])
        results.extend(values)
    return results

def noa_df_convert_nans(df, column, nan_val):
    """Checks the dataframe of raw strings for specific values to replace with nans."""
    
    if nan_val:
        nan_val = bytes(nan_val, "utf-8")
        df[column] = np.where(df[column] == nan_val, np.nan, df[column])

def noa_df_convert_nums(df, column, dtype, scalar):
    """Converts the numerical coiumns in the dataframe of raw strings applying appropriate conversions to the units if
    needed."""
    
    df[column] = df[column].astype(dtype)
    if scalar:
        df[column] = df[column]/scalar

def noa_df_convert_strings(df, column, dtype):
    """Removes the byte encoding on the string and datetime columns of the dataframe of raw strings ."""
    
    if dtype == "str" or dtype == "datetime":
        df[column] = df[column].str.decode("utf-8")

def noa_df_convert_datetime(df, column, dtype):
    """Converts the date values to datetime in the dataframe of raw strings."""
    
    if dtype == "datetime":
        df[column] = pd.to_datetime(df[column], format="%Y%m%d%H%M")

def fix_noaa_df_dtypes(df, fixed_params, optional_params=None):
    """Goes through a raw dataframe from parsed NOAA strings, converts the datatypes, and adds nans based on a
    parameter list."""
    
    params = fixed_params
    
    # Adds conversions to specified optional parameters if they exist.
    if optional_params:
        for subset in optional_params:
            params.extend([item for item in optional_params[subset][1]])
    
    for param in params:
        col, dtype, nan_val, scalar = param[0], param[3], param[4], param[5]
        
        noa_df_convert_nans(df, col, nan_val)
        
        if dtype != "str" and dtype != "datetime":
            noa_df_convert_nums(df, col, dtype, scalar)
        else:
            noa_df_convert_strings(df, col, dtype)
            noa_df_convert_datetime(df, col, dtype)
            
def extract_noaa_optional_str(bstring, code, char_count):
    """Abstract template to check a line in the NOAA dataset for optional data and return a substring of that line
    if it exists."""
    
    #Ensures that the search string is also in bytes.
    if type(code) is str:
        code = code.encode('utf-8')
        
    idx = bstring.find(code)
    if idx >= 0:
        substr = bstring[idx:idx+char_count]
    else:
        substr = None
    return substr