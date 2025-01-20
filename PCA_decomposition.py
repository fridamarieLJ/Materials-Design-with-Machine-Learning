import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def PCA_decomposition(X, number_of_PCs):
    # Standardize attributes
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    number_of_PCs = 10

    pca = PCA(n_components = number_of_PCs).fit(X_standardized)
    X_train_pca = pca.transform(X_standardized)
    print("With {} PCA components {var:0.4f}% of the variance is explained".format(number_of_PCs, var = 100*np.sum(pca.explained_variance_ratio_)))
    
    return X_train_pca

