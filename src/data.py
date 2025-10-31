import pandas
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(filename: str = "insurance.csv") -> pandas.DataFrame:
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / filename
    return pandas.read_csv(data_path)

def clean_data(data: pandas.DataFrame) -> pandas.DataFrame:
    # add high charges classificaiton
    data['highCharges'] = data["charges"] > 20000
    print(data.head())
    return data
def split_data(data: pandas.DataFrame, isClassification):
    if isClassification:
        targetCol = "highCharges"
    else:
        targetCol = "charges"
    X = data.drop(columns=targetCol)
    y = data[targetCol]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


