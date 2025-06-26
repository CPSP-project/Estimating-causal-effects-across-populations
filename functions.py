import pandas as pd

def compute_joint_probability(df, column):
    """
    Compute joint probability over a single column.
    Returns a DataFrame with unique values and their associated probabilities.
    """
    counts = df.groupby(column).size().reset_index(name='count')
    total = counts['count'].sum()
    counts['P_joint'] = counts['count'] / total
    return counts

def compute_marginal_probability(df, column):
    """
    Compute marginal probability over a single column.
    Returns a DataFrame with unique values and their associated probabilities.
    """
    counts = df.groupby(column).size().reset_index(name='count')
    total = counts['count'].sum()
    counts['P'] = counts['count'] / total
    return counts

def compute_conditional_probability(df, group_col, cond_col):
    """
    Compute conditional probability P(cond_col | group_col)
    for two columns: one as grouping variable and one as conditioned.

    Returns a DataFrame with columns:
        group_col, cond_col, count, total, P_cond
    """
    df_grouped = (
        df.groupby([group_col, cond_col])
          .size()
          .reset_index(name='count')
    )
    df_grouped['total'] = (
        df_grouped.groupby(group_col)['count'].transform('sum')
    )
    df_grouped['P_cond'] = df_grouped['count'] / df_grouped['total']
    return df_grouped

def create_key(df, columns, key_name="Z_key"):
    """
    Create a key column based on a single existing column.
    Accepts either:
        - a string representing a single column name
        - a list/tuple with exactly one element
    Raises an error if more than one column is provided.
    """
    if isinstance(columns, (list, tuple)):
        if len(columns) != 1:
            raise ValueError("create_key currently supports only a single column.")
        columns = columns[0]

    df[key_name] = df[columns].astype(str)  # Copy and convert column to string
    return df

def extract_columns(df):
    """
    Extract column names for X, Z, and Y assuming a fixed structure:
        - Column 0 is assumed to be ID
        - Column 1 is X
        - Column 2 is Z
        - Column 3 is Y

    Returns:
        X_columns: list containing column name for X
        Z_columns: list containing column name for Z
        Y_columns: list containing column name for Y
    """
    if df.shape[1] < 4:
        raise ValueError("At least 4 columns are required: ID, X, Z, Y")

    X_columns = [df.columns[1]]
    Z_columns = [df.columns[2]]
    Y_columns = [df.columns[3]]
    return X_columns, Z_columns, Y_columns
