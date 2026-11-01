from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train):
        self.scaler.fit(X_train)
    
    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(X_train)
    

