from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from fetch_data import fetch_obesity_data
from logistic_regression import MyLogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def cross_validate_model(X, y, model, transformer, n_splits=3, sampling=None, param_grid=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train_raw, X_val_raw = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_transformed = transformer.fit_transform(X_train_raw)
        X_val_transformed = transformer.transform(X_val_raw)

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_val_scaled = scaler.transform(X_val_transformed)

        if sampling == 'smote':
            sampler = SMOTE(random_state=42)
            X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
        elif sampling == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
            X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)

        if param_grid:
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            print(f"[Fold {fold}] Best Params: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)

        y_pred = best_model.predict(X_val_scaled)

        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"Fold {fold}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        scores.append((acc, precision, recall, f1))
        fold += 1

    scores = np.array(scores)
    print(f"\nAverage Accuracy: {np.mean(scores[:,0]):.4f} ± {np.std(scores[:,0]):.4f}")
    print(f"Average Precision: {np.mean(scores[:,1]):.4f}")
    print(f"Average Recall:    {np.mean(scores[:,2]):.4f}")
    print(f"Average F1-score:  {np.mean(scores[:,3]):.4f}")
    return scores

def train_and_plot(X, y, model_class, transformer):
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_transformed = transformer.fit_transform(X_train_raw)
    X_val_transformed = transformer.transform(X_val_raw)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_val_scaled = scaler.transform(X_val_transformed)

    model = model_class(learning_rate=0.1, n_iters=300)
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)


    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(model.train_losses, label="Train Loss")
    plt.plot(model.test_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of Cost Function")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df = fetch_obesity_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['NObeyesdad'])
    X = df.drop(columns=['NObeyesdad'])

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ordinal_columns = ["CAEC", "CALC"]
    binary_columns = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]

    ordinal_labels = [
        ["no", "Sometimes", "Frequently", "Always"],  # CAEC
        ["no", "Sometimes", "Frequently", "Always"]   # CALC
    ]

    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ])

    transformer = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_columns),
        ('binary', OrdinalEncoder(), binary_columns),
        ('ordinal', OrdinalEncoder(categories=ordinal_labels), ordinal_columns),
        ('cat', OneHotEncoder(), ['MTRANS']),
    ])
    
    tree_param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    cross_validate_model(X, y, DecisionTreeClassifier(random_state=42), transformer,
                         sampling='smote', param_grid=tree_param_grid)
    # train_and_plot(X, y, MyLogisticRegression, transformer)