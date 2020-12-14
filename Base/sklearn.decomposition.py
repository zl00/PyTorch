import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 1], [2, 2]])
pca = PCA(n_components=1)
X_transformed = pca.fit_transform(X)
print(X_transformed)