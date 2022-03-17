def data_clean(data=None):
    data.loc[data['workclass'] == ' ?', 'workclass'] = data['workclass'].mode()
    data.loc[data['occupation'] == ' ?', 'occupation'] = data['occupation'].mode()
    data.loc[data['native_country'] == ' ?', 'native_country'] = data['native_country'].mode()

    return data.copy()