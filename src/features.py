from data import load_data, clean_data, split_data
import pandas as pd
import numpy as np

def add_highCharges(df: pd.DataFrame) -> pd.DataFrame:
    # add high charges classificaiton
    df['highCharges'] = df["charges"] > 20000
    df.drop('charges', axis=1)
    return df

def encode(data: pd.DataFrame) -> pd.DataFrame:
    """Returns the input data frame with categorical variables converted to encodings"""

    encoded = data.select_dtypes(include=['int64', 'float64', 'bool'])
    encoded.insert(3, "Sex-Female", data["sex"]=="female")
    encoded.insert(3, "Smoker", data["smoker"]=="yes")
    for region in ["SW", "NW", "NE"]:
        encoded.insert(3, "region-"+region, data['region']==region)
    return encoded

def upsample(X_train, y_train):
    """Balances by upsampling the highCharges cases
    :return X_train, y_train"""
    imbalance = (len(y_train) - sum(y_train)) - sum(y_train)
    highCharges_indices = y_train[y_train == True].index

    np.random.seed(42)
    upsampled_indices = np.random.choice(highCharges_indices, size=imbalance, replace=True)

    new_indices = np.concatenate([y_train.index, upsampled_indices])
    # Create upsampled datasets
    X_train_upsampled = X_train.loc[new_indices].reset_index(drop=True)
    y_train_upsampled = y_train.loc[new_indices].reset_index(drop=True)
    return X_train_upsampled, y_train_upsampled


