import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(filename: str = "insurance.csv") -> pd.DataFrame:
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / filename
    return pd.read_csv(data_path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Shorten region names
    region_map = {'southwest': "SW", 'southeast': 'SE',
                  'northwest': "NW", 'northeast': 'NE',
                  }
    data['region'] = data['region'].replace(region_map)

    # add high charges classificaiton
    data['highCharges'] = data["charges"] > 20000
    return data


def split_data(data: pd.DataFrame, isClassification: bool):
    if isClassification:
        targetCol = "highCharges"
    else:
        targetCol = "charges"
    X = data.drop(["charges", "highCharges"], axis=1)
    y = data[targetCol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def load_clean_split(isClassification):
    """
    Loads, cleans and splits the data

    args:
        isClassificaiton (bool): true for classification tasks

    returns:
        pd.DataFrame: X_train, X_test, y_train, y_test
    """
    data = load_data()
    data = clean_data(data)
    return split_data(data, isClassification)


def _test():
    # Verifying data has been split properly
    X_train, X_test, y_train, y_test = load_clean_split(True)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.size)
    print(X_train.head)


if __name__ == "__main__":
    _test()


