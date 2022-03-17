def data_import(file=None):
    import pandas as pd
    data = pd.read_csv(file)

    return data