def rename_filter(data=None):
    import pandas as pd
    from module.ohecolnames import ohe_colnames
    data = pd.DataFrame(data)
    names = ohe_colnames()

    sel_features = ['marital-status_ Married-civ-spouse', 'capital_loss', 'education_num',
                    'occupation_ Exec-managerial',
                    'capital_gain', 'age', 'hours_per_week', 'sex_ Female', 'relationship_ Own-child',
                    'occupation_ Other-service', 'occupation_ Prof-specialty', 'fnlwgt']

    data.columns = names
    data = data[sel_features].copy()

    return data