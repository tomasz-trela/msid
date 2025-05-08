from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder ,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fetch_data import fetch_obesity_data
from linear_regression import LinearRegressionClosedForm
import numpy as np

from logistic_regression import LogisticRegressionGD

if __name__ == "__main__":

    df = fetch_obesity_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['NObeyesdad'])
    X = df.drop(columns=['NObeyesdad'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ordinal_columns = ["CAEC", "CALC"]
    binary_columns = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]

    ordinal_labels = [
        ["no", "Sometimes", "Frequently", "Always"],  # CAEC
        ["no", "Sometimes", "Frequently", "Always"]   # CALC
    ]

    transformer = ColumnTransformer(transformers=[
        ('binary', OrdinalEncoder(), binary_columns),
        ('ordinal', OrdinalEncoder(categories=ordinal_labels), ordinal_columns),
        ('cat', OneHotEncoder(), ['MTRANS']),
    ])

    clf = Pipeline([
        ('preprocessing', transformer),
        ('scaler', RobustScaler()),
        # ('classifier', DecisionTreeClassifier())
        # ('classifier', LogisticRegression())
        # ('classifier', LinearRegressionClosedForm())
        # ('classifier', SVC(kernel='rbf', C=1.0, probability=True))
        ('classifier', LogisticRegressionGD())
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
