import pandas as pd

def compute_joint_probability(df, columns):
    """
    Group by the specified columns and compute the joint probability.
    """
    counts = df.groupby(columns).size().reset_index(name='count')
    total = counts['count'].sum()  # Total number of rows
    counts['P_joint'] = counts['count'] / total  # Normalize to get the joint probability
    return counts

def compute_marginal_probability(df, columns):
    """
    Compute the marginal probability for the specified column (or group of columns).
    """
    if not isinstance(columns, list):
        columns = [columns]  # Convert 'columns' to a list if it is a single column (string), 
                             # so that the function works uniformly with both single and multiple columns
    counts = df.groupby(columns).size().reset_index(name='count')
    total = counts['count'].sum()
    counts['P'] = counts['count'] / total
    return counts

def compute_conditional_probability(df, group_cols, cond_col):
    """
    Compute the conditional probability P(cond_col | group_cols) by grouping occurrences.
    """
    # If cond_col is a list, concatenate it directly; otherwise, convert it to a list
    if isinstance(cond_col, list):
        all_cols = group_cols + cond_col
    else:
        all_cols = group_cols + [cond_col]
    
    df_grouped = df.groupby(all_cols).size().reset_index(name='count')
    df_grouped['total'] = df_grouped.groupby(group_cols)['count'].transform('sum')
    df_grouped['P_cond'] = df_grouped['count'] / df_grouped['total']
    return df_grouped

def create_key(df, columns, key_name="Z_key"):
    """
    Combine the specified columns (in 'columns') into a single string column called key_name.
    """
    df[key_name] = df[columns].astype(str).agg('_'.join, axis=1)
    return df

def extract_columns(df, num_x, num_z, num_y):
    """
    Estrae le colonne X, Z e Y dai primi N campi della tabella.
    Assumiamo che dopo una colonna ID, le X vengano prima, poi Z, poi Y.
    """
    X_columns = list(df.columns[1 : 1 + num_x])
    Z_columns = list(df.columns[1 + num_x : 1 + num_x + num_z])
    Y_columns = list(df.columns[1 + num_x + num_z : 1 + num_x + num_z + num_y])
    return X_columns, Z_columns, Y_columns

