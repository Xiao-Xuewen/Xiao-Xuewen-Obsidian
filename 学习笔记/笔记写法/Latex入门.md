# SN2N核心公式

```
常用快捷公式输入：
mk 行内公式
dm 独立公式
sr 平方
cb 立方
rd 上角标
_  下角标
sq 根号
// 空分数
x/y+tan 快速分数
text 插入文字
sum 求和
\mathbb{E} 期望
\| 范数
hat 头顶符号
tilde 估计 扰动
bmatrix 矩阵 &分割 \\换行
\mathcal{F} 傅里叶F
```


$$
y_{1} = x + n_{1}
$$

$$
\hat{x} = f_{\theta}(x)
$$
$$
\tilde{x}
$$

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y_{i}})^{2}
$$

$$
L = \|x - y\|_{1}
$$

$$
L = \|x - y\|_{2}^{2}
$$

$$
\mathbb{E}[n] = 0
$$

$$
x_{1} = \frac{{a + b}}{2}
$$

$$
x_{2} = \frac{{b+c}}{2}
$$

$$
\hat{x}_{1} = f_{\theta}(\tilde{x}_{1})
$$

$$
\hat{x}_{2}=f_{\theta}(\tilde{x}_{2})
$$

$$
L_{c} = \lambda\|\hat{x}_{1}-\hat{x}_{2}\|_{1}
$$

$$
L = \frac{{\|\hat{x}_{1}-\tilde{x}_{2}\|_{1}+\|\hat{x}_{2}-\tilde{x}_{1}\|_{1}+\lambda\|\hat{x}_{1}-\hat{x}_{2}\|_{1}}}{2+\lambda}
$$

$$
y = Hx+n
$$

$$
y =h*x+n
$$

$$
PSNR = 10\log_{10}{\frac{{MAX^{2}}}{MSE}}
$$

$$
SSIM(x,y)
$$

$$
f^{*}(y)=\mathbb{E}[x|y]
$$

$$
\theta^{*}=arg\,\min_{\theta}L(theta)
$$

$$
\theta_{t+1}=\theta_{t}-\eta{\frac{m_{t}}{\sqrt{ v_{t} }+\xi}}
$$

$$
y_{i}=\sum w_{j}x_{x+j}+b
$$

$$
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

$$
F(u,v)=\mathcal{F}\{f(x,y)\}
$$

$$
F(x,y)=\mathcal{F}^{-1}\{F(u,v)\}
$$
