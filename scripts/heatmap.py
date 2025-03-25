from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
import seaborn as sns


def my_heatmap(df):
    label_encoder = LabelEncoder()

    df['NObeyesdad_numeric'] = label_encoder.fit_transform(df['NObeyesdad'])

    df['NObeyesdad_numeric']

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_columns]

    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
    plt.show()