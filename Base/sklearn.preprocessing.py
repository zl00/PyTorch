from sklearn.preprocessing import StandardScaler

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print("scaler.fit----")
print(scaler.fit(data))
print("scaler.mean_----")
print(scaler.mean_)
print("scaler.transform----")
print(scaler.transform(data))
print("scaler.fit_transform----")
print(scaler.fit_transform(data))
print("----")