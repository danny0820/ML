import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 設置標記器和顏色映射
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 繪出決策邊界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 畫出樣本的散點圖
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

# 讀取資料
data = pd.read_csv('hw1_train.dat.txt', header=None, sep='\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 創建 AdalineGD 實例
adaline = AdalineGD(eta=0.001, n_iter=100)

# 訓練模型
adaline.fit(X, y)

# 顯示訓練後的權重
print("訓練後的權重:", adaline.w_)

# 顯示每次迭代的成本
print("每次迭代的成本:", adaline.cost_)

# 繪製成本下降圖
plt.plot(range(1, len(adaline.cost_) + 1), adaline.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.001')
plt.show()

# # 選擇所有可能的特徵對並繪製決策邊界
# feature_combinations = list(itertools.combinations(range(X.shape[1]), 2))
# for (f1, f2) in feature_combinations:
#     X_subset = X[:, [f1, f2]]
#     adaline.fit(X_subset, y)

#     plt.figure(figsize=(8, 6))
#     plot_decision_regions(X_subset, y, classifier=adaline)
#     plt.title(f'Decision regions for features {f1+1} and {f2+1}')
#     plt.xlabel(f'Feature {f1+1}')
#     plt.ylabel(f'Feature {f2+1}')
#     plt.legend(loc='upper left')
#     plt.show()
