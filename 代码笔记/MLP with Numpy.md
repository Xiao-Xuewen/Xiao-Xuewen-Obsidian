## 0. 简述
使用Numpy写神经网络无疑是麻烦的 写前向传播和反向传播需要手写矩阵计算（对应着[[神经网络入门(多层感知机，MLP)#核心结论(单样本)]]中的核心结论公式），不过这一过程将数据通过前向传播从输入到输出，参数通过反向传播优化器更新，再次前向传播....这一套神经网络的基本流程以及实现方法进行了深刻的理解与记忆！
## 1. 超参数
```
import numpy as np

import matplotlib.pyplot as plt

num_per_class = 100

num_classes = 3

noise = 0.2

seed = 0

input_dim = 2

hidden_dim = 100

out_dim = 3

num_steps = 5000

num_samples = num_per_class * num_classes

learning_rate = 1.0
```

## 2. 二维螺旋生成函数
```

def make_2D_spiral(num_per_class,num_classes,noise,seed):

    num_samples = num_per_class * num_classes

  

    X = np.zeros((2,num_samples)) 

    y = np.zeros(num_samples,dtype=int)

  

    rg = np.random.default_rng(seed)

    for j in range(num_classes):

        idx = slice(j*num_per_class,(j+1)*num_per_class)

  

        r = np.linspace(0,1,num_per_class)

        theta = np.linspace(j*4,(j+1)*4,num_per_class)

        theta += rg.normal(0,noise,num_per_class)

  

        X[0,idx] = r * np.sin(theta)

        X[1,idx] = r * np.cos(theta)

  

        y[idx] = j

  

    return X,y

```
## 3. 训练数据与GT生成

```

X,y = make_2D_spiral(num_per_class,num_classes,noise,seed)

  

Y = np.zeros((num_classes,num_samples))

Y[y,np.arange(num_samples)] = 1

  
```

## 4. 激活函数与损失函数
```
def relu(z):

    return np.maximum(z,0)

  

def softmax(z):

    z = z - np.max(z,axis = 0 ,keepdims = 1)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z,axis = 0,keepdims = 1)

```
## 5. 参数初始化
```

rg = np.random.default_rng(seed)

w1 = rg.standard_normal((hidden_dim,input_dim))

b1 = np.zeros((hidden_dim,1))

  

w2 = rg.standard_normal((out_dim,hidden_dim))

b2 = np.zeros((out_dim,1))

  

# ChatGPT 绘图代码

w1_init = w1.copy()

b1_init = b1.copy()

w2_init = w2.copy()

b2_init = b2.copy()

# ChatGPT 绘图代码

  
```
## 6. 循环训练
```
for step in range(num_steps):
```
### 6.1 前向传播
```
	
    z1 = w1 @ X + b1

    a1 = relu(z1)

    z2 = w2 @ a1 +b2

  

    probs = softmax(z2)

    correct_probs = probs[y,np.arange(num_samples)]

  

    loss = -np.mean(np.log(correct_probs+1e-12))

```
### 6.2 反向传播
```

    delta2 = (probs - Y)/num_samples

  

    dw2 = delta2 @ a1.T

    db2 = np.sum(delta2,axis = 1,keepdims=1)

  

    delta1 = w2.T @ delta2 * (z1 > 0)

  

    dw1 = delta1 @ X.T

    db1 = np.sum(delta1,axis = 1,keepdims=1)

  

    w1 -= dw1 *learning_rate

    w2 -= dw2 *learning_rate

    b1 -= db1 *learning_rate

    b2 -= db2 *learning_rate

  
```
### 6.3 打印结果
```
    if step % 500 == 0:

        prediction = np.argmax(probs,axis=0)

        accuracy = np.mean(y == prediction)

        print(f"step:{step:4d}")

        print(f"loss:{loss:.4f}")

        print(f"accuracy:{accuracy:.4f}")

  
```
## 7. 重新前向传播计算最终结果
```
z1 = w1 @ X + b1

a1 = relu(z1)

z2 = w2 @ a1 +b2

probs = softmax(z2)

prediction = np.argmax(probs,axis=0)

accuracy = np.mean(y == prediction)

print(f"Final accuracy:{accuracy:.4f}")

```
## 8. 绘图
```

# ChatGPT 绘图代码

def predict_grid(X, w1, b1, w2, b2):

    # 生成网格范围

    x_min, x_max = X[0].min() - 0.2, X[0].max() + 0.2

    y_min, y_max = X[1].min() - 0.2, X[1].max() + 0.2

  

    xx, yy = np.meshgrid(

        np.linspace(x_min, x_max, 300),

        np.linspace(y_min, y_max, 300)

    )

  

    # 每个网格点都当成一个输入样本

    grid = np.vstack([xx.ravel(), yy.ravel()])   # shape = (2, 300*300)

  

    # 前向传播

    z1 = w1 @ grid + b1

    a1 = relu(z1)

    z2 = w2 @ a1 + b2

    probs = softmax(z2)

  

    # 取预测类别

    pred = np.argmax(probs, axis=0)

    pred = pred.reshape(xx.shape)

  

    return xx, yy, pred

  
  

# 训练前（随机初始化时）的分类区域

xx1, yy1, pred_before = predict_grid(X, w1_init, b1_init, w2_init, b2_init)

  

# 训练后（最终参数）的分类区域

xx2, yy2, pred_after = predict_grid(X, w1, b1, w2, b2)

  
  

# 画两个子图

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  

# -----------------------------

# 左图：训练前

# -----------------------------

axes[0].contourf(xx1, yy1, pred_before, alpha=0.3)

axes[0].scatter(X[0], X[1], c=y)

axes[0].set_title("Before Training")

axes[0].set_xlabel("x1")

axes[0].set_ylabel("x2")

  

# -----------------------------

# 右图：训练后

# -----------------------------

axes[1].contourf(xx2, yy2, pred_after, alpha=0.3)

axes[1].scatter(X[0], X[1], c=y)

axes[1].set_title(f"After Training (acc={accuracy:.4f})")

axes[1].set_xlabel("x1")

axes[1].set_ylabel("x2")

  

plt.tight_layout()

plt.show()

# ChatGPT 绘图代码
```
