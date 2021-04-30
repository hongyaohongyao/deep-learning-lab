import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_img(X, label, title):
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(X[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}:{}".format(title, label[i].item()), fontdict=dict(fontsize=25))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def draw_confusion_matrix(y_real, y_pred):
    plt.figure(figsize=(12, 12))
    cfm = confusion_matrix(y_real, y_pred)
    sns.set(font_scale=1.5)
    sns.heatmap(cfm, linewidths=0.05, annot=True, linecolor='red',
                cmap="rainbow")
    plt.show()
