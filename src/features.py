from sklearn.compose import ColumnTransformer

from data import load_data, clean_data, split_data
import pandas as pd
import numpy as np
from data import load_clean_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

def load_scaled_encoded_data(isClassification: bool):
    X_train, X_test, y_train, y_test = load_clean_split(isClassification)

    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['smoker', 'sex', 'region']
    # Create a transformer to standardize
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(), categorical_features),
        ]
    )
    transformer.fit(X_train)

    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test, y_train, y_test


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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import get_filepath
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv(get_filepath("data", "insurance.csv"))

threshold = 20000
df['highCharges'] = (df['charges'] > threshold).astype(int)

X = df.drop(['charges', 'highCharges'], axis=1)
y = df['highCharges']

X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)