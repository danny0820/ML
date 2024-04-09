import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    感知機分類器。
    參數:
    eta : float
        學習速率（介於 0.0 和 1.0 之間）
    n_iter : int
        資料集的訓練次數。
    random_state : int
        用於隨機權重初始化的亂數生成器種子。
    屬性:
    w_ : 1d-array
        訓練後的權重。
    errors_ : list
        每一次迭代中的錯誤分類數量。
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        適應訓練數據。
        參數:
        X : {array-like}, shape = [n_examples, n_features]
            訓練向量，n_examples 是例子的數量，
            n_features 是特徵的數量。
        y : array-like, shape = [n_examples]
            目標值。
        返回:
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """計算淨輸入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """回傳單位階躍後的類別標籤"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
# 讀取資料
data = pd.read_csv('hw1_train.dat.txt', header=None, sep='\s+')
X = data.iloc[:, :-1].values  # 取前面的列為特徵
y = data.iloc[:, -1].values  # 取最後一列為標籤

# 創建感知機實例
perceptron = Perceptron(eta=0.01, n_iter=50)

# 訓練模型
perceptron.fit(X, y)


# 顯示訓練後的權重
print("訓練後的權重:", perceptron.w_)

# 顯示每次迭代的錯誤數量
print("每次迭代的錯誤數量:", perceptron.errors_)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)


# 據已有的感知機模型訓練情況繪製成本下降圖
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Cost (Misclassifications) over epochs')
plt.show()


# # 讀取資料
data = pd.read_csv('hw1_train.dat.txt', header=None, sep='\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 選擇所有可能的特徵對並繪製決策邊界
feature_combinations = list(itertools.combinations(range(X.shape[1]), 2))
for (f1, f2) in feature_combinations:
    perceptron = Perceptron(eta=0.01, n_iter=50)
    X_subset = X[:, [f1, f2]]
    perceptron.fit(X_subset, y)

    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_subset, y, classifier=perceptron)
    plt.title(f'Decision regions for features {f1+1} and {f2+1}')
    plt.xlabel(f'Feature {f1+1}')
    plt.ylabel(f'Feature {f2+1}')
    plt.legend(loc='upper left')
    plt.show()
