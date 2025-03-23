from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import seaborn as sns


def reduce_dimensions_pca(df, pallete):
    encoded_df = df.copy(deep=True)
    categorical = df.select_dtypes(include=['object']).columns

    encoder = LabelEncoder()

    for name in categorical:
        encoded_df[name] = encoder.fit_transform(df[name])



    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(encoded_df)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)

    df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df_pca['Gender'] = encoded_df['Gender']


    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=df_pca['Gender'].map({0: 'Female', 1: 'Male'}),
                    palette=pallete, alpha=0.3)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of 17-Dimensional Data')
    plt.legend(title='Gender')
    plt.grid(True)
    plt.show()