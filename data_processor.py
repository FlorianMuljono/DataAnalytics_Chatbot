import pandas as pd
import numpy as np
import os

def process_data(file_path, file_extension):
    """
    Process the uploaded file based on its extension.
    Returns a pandas DataFrame.
    """
    if file_extension in ['csv']:
        # Try different encodings and delimiters for CSV
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            # Try with a different encoding
            df = pd.read_csv(file_path, encoding='latin1')
        except pd.errors.ParserError:
            # Try with a different delimiter
            df = pd.read_csv(file_path, sep=';')
    
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(file_path)
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Basic data cleaning
    # Handle column names: strip whitespace and remove special characters
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Convert column names to strings
    df.columns = df.columns.astype(str)
    
    return df

def clean_column_name(col_name):
    """Clean column names by removing special characters and extra whitespace"""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    
    # Replace spaces with underscores and remove special characters
    import re
    col_name = re.sub(r'\s+', '_', col_name.strip())
    col_name = re.sub(r'[^\w_]', '', col_name)
    
    return col_name

def get_data_info(df):
    """
    Extract useful information about the DataFrame.
    Returns a dictionary with summary statistics and metadata.
    """
    # Basic info
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        # Get basic statistics for numeric columns
        desc = df[numeric_cols].describe().to_dict()
        info["numeric_stats"] = desc
    else:
        info["numeric_stats"] = {}
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        cat_info = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 categories
            unique_count = df[col].nunique()
            cat_info[col] = {
                "unique_count": unique_count,
                "top_values": value_counts
            }
        info["categorical_stats"] = cat_info
    else:
        info["categorical_stats"] = {}
    
    # Date columns
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        date_info = {}
        for col in date_cols:
            date_info[col] = {
                "min_date": df[col].min().strftime('%Y-%m-%d') if not pd.isna(df[col].min()) else None,
                "max_date": df[col].max().strftime('%Y-%m-%d') if not pd.isna(df[col].max()) else None
            }
        info["date_stats"] = date_info
    else:
        info["date_stats"] = {}
    
    return info

def detect_data_types(df):
    """
    Detect and categorize data types in the DataFrame.
    Returns a dictionary mapping columns to their inferred types.
    """
    data_types = {}
    
    for column in df.columns:
        col_data = df[column]
        
        # Check if the column has a datetime dtype
        if pd.api.types.is_datetime64_any_dtype(col_data):
            data_types[column] = "datetime"
            continue
        
        # Check numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            # Check if it looks like a categorical/ordinal column despite being numeric
            if col_data.nunique() < 10 and col_data.nunique() / len(col_data) < 0.05:
                data_types[column] = "categorical"
            # Check if it might be a binary column
            elif col_data.nunique() == 2:
                data_types[column] = "binary"
            # Check if it looks like an ID column
            elif col_data.nunique() == len(col_data) and col_data.nunique() > 0.9 * len(col_data):
                data_types[column] = "id"
            # Otherwise, it's continuous
            else:
                data_types[column] = "continuous"
            continue
        
        # For non-numeric columns
        if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            # Try to convert to datetime
            try:
                pd.to_datetime(col_data, errors='raise')
                data_types[column] = "datetime"
                continue
            except (ValueError, TypeError):
                pass
            
            # Check if it might be a binary column
            if col_data.nunique() == 2:
                data_types[column] = "binary"
            # Check if it looks like an ID or unique identifier
            elif col_data.nunique() == len(col_data) and col_data.nunique() > 0.9 * len(col_data):
                data_types[column] = "id"
            # Check if it could be a name column
            elif any(name_identifier in column.lower() for name_identifier in ["name", "first", "last", "full"]):
                data_types[column] = "name"
            # Otherwise, consider it categorical
            else:
                data_types[column] = "categorical"
            continue
        
        # If we can't determine the type, mark as unknown
        data_types[column] = "unknown"
    
    return data_types
