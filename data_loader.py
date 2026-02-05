import pandas as pd

def load_dataset(path):
    """
    Load student performance dataset
    """
    df = pd.read_csv(path, sep=';')
    return df

if __name__ == "__main__":
    data = load_dataset("data/student-mat.csv")
    print(data.head())
