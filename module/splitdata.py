def split_data(data=None):
    from sklearn.model_selection import train_test_split
    X = data.drop('inc_>50K', axis=1)
    y = data['inc_>50K']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()