#-*-coding:utf-8-*-

"""
梯度下降法实现对率回归(Logistic regression)
来源: '机器学习，周志华'
模型: P69 problem 3.3
数据集 P89 watermelon_3.0a 
"""
import matplotlib.pyplot as plt
import numpy as np
import xlrd

def sigmoid(x):
    """
    Sigmoid function.
    Input:
        x: np.array
    Return:
        y: the same shape with x
    """
    y = 1.0 / (1 + np.exp(-x))
    return y

def Gradient_Descent(X, y):
    """
    Input:
        X: np.array with shape [N, 3]. Input.
        y: np.array with shape [N, 1]. Label.
    Return:
        beta: np.array with shape [1, 3]. Last one is b.
                Optimal params with Gradient_Descent method
    """
    # beta = beta - alpha * first_order
    # Give the learning rate alpha
    alpha = 0.1

    N = X.shape[0]
    beta = np.ones((1, 3))
    z = X.dot(beta.T)

    # log-likehood
    old_l = 0
    new_l = np.sum(-y*z + np.log(1+np.exp(z))) 
    iters = 0
    while (np.abs(old_l - new_l) > 1e-5):
        # shape [N, 1]
        p1 = np.exp(z) / (1 + np.exp(z))

        # shape [1, 3]
        first_order = -np.sum(X * (y - p1), 0, keepdims=True)
        
        # update
        beta -= alpha * first_order
        z = X.dot(beta.T)
        old_l = new_l
        new_l = np.sum(-y*z + np.log(1+np.exp(z))) 

        iters += 1
    print("梯度下降法法收敛的迭代次数iters: ", iters)
    print("梯度下降法的beta: ", beta)
    print('梯度下降法收敛后对应的代价函数值: ', new_l)
    return beta

if __name__ == "__main__":
    
    # Read data from csv file 
    workbook = xlrd.open_workbook("3.0alpha.xlsx")
    sheet = workbook.sheet_by_name("Sheet1")
    X1 = np.array(sheet.row_values(0))
    X2 = np.array(sheet.row_values(1))
    X3 = np.ones(X1.shape[0])           # for parameter b

    # X, y
    X = np.vstack((X1, X2, X3)).T
    y = np.array(sheet.row_values(3))
    y = y.reshape(-1, 1)

    # Plot training data
    for i in range(X1.shape[0]):
        if y[i, 0] == 0:
            plt.plot(X1[i], X2[i], 'r+')

        else:
            plt.plot(X1[i], X2[i], 'bo')

    # Get optimal parameters beta with Gradient Descent method
    beta = Gradient_Descent(X, y)
    GD_left = -( beta[0, 0]*0.1 + beta[0, 2] ) / beta[0, 1]
    GD_right = -( beta[0, 0]*0.9 + beta[0, 2] ) / beta[0, 1]
    
    # Out put the figure
    plt.plot([0.1, 0.9], [GD_left, GD_right], 'g-', label='Gradient_Descent method')
    plt.legend()

    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title("Logistic Regression")
    plt.show()
