
神经元：存放激活值的容器--->一个输入是上一层的神经元的输出 本身的输出是一个0-1的数代表本神经元的激活值的函数
比如检测某一个像素区域是否是“横线”（两边是黑色的 中间是白色的）我们把这一区域的上一层激活值赋予的权重就可以是**两边的激活值权重是负的中间的激活值权重是正的**（实际训练时，初始权重往往是随机的，通过损失函数--反向传播--修改权重） 
为什么呢 这样我们在进入激活函数之前的加权和的表现就是在它是横线的时候较大不是横线的时候较小 因为假如现在这个区域就是一条横线 白色区域对应的是接近1的激活值 黑色部分是接近0的激活值 白色的地方加分 黑色的地方基本上不扣分 总和就较大 如果不是横线 是全白的 那么扣分的区域它扣的就多 加权和就小 如果全是黑色 那虽然边缘基本不扣分 但是中间部分也基本不加分 所以加权和也不大 在计算完加权和之后我们可以加一个常数偏置 来控制下一个神经元有“多容易激活”，加权和+偏置再进入激活函数 可以选择sigmoid 来把输出控制在0-1之间（当然不是说神经元输出的激活值只能是0-1，其实sigmoid已经有些old，还有很多激活函数比如ReLU(z)=max(0,z)输出的值就不一定了）
另外上面两段说的激活值是由输入层对应的是图片像素（归一化之后），但是后续的层的神经元就不一定代表某个像素的“亮度”了，可能代表的就是我们通过输入层得到的一些信息，比如某一个区域的像素是”横线“的概率有多大
线性运算
在神经网络的两层运算当中 我们可以使用矩阵运算的方式极大的减少我们写代码的工作量
第0层的第$i$个神经元(共有n个)：
$$
a^{(0)}_{i}
$$
$$
a^{(0)} = \begin{bmatrix}
a^{(0)}_{1} \\
a^{(0)}_{2} \\
. \\
. \\
a^{(0)}_{n}
\end{bmatrix}_{n\times 1}
$$
$$

$$

第0层的第$i$个神经元对应第1层第$m$个神经元（共有k个）的权重：
$$
w_{m,i}
$$
第0层到第1层所有的权重：
$$
W = \begin{bmatrix}
w_{1,1}&w_{1,2}&w_{1,3}&\dots&w_{1,n} \\
w_{2,1}&w_{2,2}&w_{2,3}&\dots&w_{2,n} \\
\dots&\dots&\dots&\dots&\dots \\
w_{k,1}&w_{k,2}&w_{k,3}&\dots&w_{k,n}
\end{bmatrix}_{{k}\times n}
$$
第1层的第m个神经元对应的第0层的偏置(共有k个)：
$$
b_{m}
$$
$$
b = \begin{bmatrix}
b_{1} \\
b_{2} \\
. \\
. \\
b_{k}
\end{bmatrix}_{k\times 1}
$$

第1层的第m个神经元(共有k个)：
$$
a^{(1)}_{m}
$$
$$
a^{(1)} = \begin{bmatrix}
a^{(1)}_{1} \\
a^{(1)}_{2} \\
. \\
. \\
a^{(1)}_{k}
\end{bmatrix}_{k\times 1}
$$

那么：
$$
\begin{bmatrix}
a^{(1)}_{1} \\
a^{(1)}_{2} \\
. \\
. \\
a^{(1)}_{k}
\end{bmatrix}_{k\times 1}=\sigma(\begin{bmatrix}
w_{1,1}&w_{1,2}&w_{1,3}&\dots&w_{1,n} \\
w_{2,1}&w_{2,2}&w_{2,3}&\dots&w_{2,n} \\
\dots&\dots&\dots&\dots&\dots \\
w_{k,1}&w_{k,2}&w_{k,3}&\dots&w_{k,n}
\end{bmatrix}_{{k}\times n}\times \begin{bmatrix}
a^{(0)}_{1} \\
a^{(0)}_{2} \\
. \\
. \\
a^{(0)}_{n}
\end{bmatrix}_{n\times 1} + \begin{bmatrix}
b_{1} \\
b_{2} \\
. \\
. \\
b_{k}
\end{bmatrix}_{k\times 1})
$$

即：
$$
a^{(1)} = \sigma(W\times a^{(0)}+b)
$$
单个神经元的计算：
$$
a^{(1)}_{m} = \sigma(\sum_{i=1}^{n}w_{m,i}\times a^{(0)}_{i}+b_{m})
$$

那么这些权重以及偏置的值是如何确定的呢？
首先引入**代价函数**（cost）：它是衡量网络对于分类结果的偏差的表现 这里的数字分类的cost计算就是一次的输出比如10个数字的10个分数，正确答案应该是一个数字为1 ，其他为0，那么最简单的就是输出的分数-对应的正确答案分数的平方和，所有样本的平均值就是cost
cost越大，分类效果就越差，cost越小，分类效果越好 本质上cost就是以这个网络的所有参数为自变量的一个多元函数，有一些是直接通过输出层来参与的 大部分是通过全连接层来参与的 但总之他们最终都作用到了cost中
**损失函数**（loss）是评估单一样本的分类结果的，代价函数一般是很多训练样本损失函数值的平均值，但二者代表的意思本质相同 后续使用loss来代指 
我们的目的应该是让loss尽可能的小，这样就代表着我们分类的就越好(并不是绝对的 因为模型最终需要泛化 loss小不一定代表泛化的好)，但是我们如何的去控制这些参数的变化呢？
这里引入**梯度**：一个多元函数在某一个点上对所有的自变量的偏导数值组成的向量就是该点的梯度值 在这个点的附近走一小步时 沿着梯度指向的方向变化 函数值增大的最多 相反 沿着梯度指向的反向变化 函数值减少的最多
那么这里我们记loss为:
$$
L
$$
记网络参数（这里就是神经元权重和偏置）为一个列向量：
$$
\theta=\begin{bmatrix}
\theta_{1} \\
\theta_{2} \\
. \\
. \\
\theta_{n}
\end{bmatrix}_{n\times 1}
$$
我们有：
$$
L = f(\theta_{1},\theta_{2},\dots \theta_{n})
$$
这时我们开始基于随机的$[\theta_{n}]$,然后计算该点的梯度：
$$
\nabla_{\theta} L = \begin{bmatrix}
\frac{{\partial L}}{\partial \theta_{1}} \\
\frac{{\partial L}}{\partial \theta_{2}} \\
. \\
. \\
\frac{{\partial L}}{\partial \theta_{n}} 
\end{bmatrix}_{n\times 1}
$$
也许我们可以得到一个这样的列向量（这其实就是反向传播做的事情，具体的计算过程涉及到链式法则，这里先不做详细说明）：
$$
\begin{bmatrix}
1.4 \\
-0.3 \\
. \\
. \\
2.1
\end{bmatrix}
$$
那么我们知道偏导为正 对应的$\theta_{1}$如果增大 那么$L$也会增大，偏导为负 对应的$\theta_{2}$增大$L$会变小，我们需要让$L$变小也就是说让对应的$\theta_{1}$变小，$\theta_{2}$变大....
所以说：
$$
\theta_{i} \leftarrow \theta_{i} - \eta \frac{\partial L}{\partial \theta_{i}}
$$
这里的$\eta$是学习率，后面细说 
这里的公式只是用来直观的体现通过反向传播来计算梯度（链式法则）来告诉网络下一步参数应该往哪里改 然后梯度下降来真正的更新参数(这一步往往涉及到优化器，优化器是真正的更新参数的模块，它所用到的可能不止有当前的梯度，最简单的是梯度下降),另外，梯度的绝对值大小代表着它的变化对于loss带来的影响的大小，也就是loss对于对应的参数变化的敏感程度
整个网络训练的流程大致可以表述为：
$$
初始化参数\to 前向传播计算L \to 反向传播计算\nabla_{\theta} L \to 优化器梯度下降更新参数
$$
重复的前向传播，反向传播，梯度下降更新参数，一步通常称为一个step，直到满足停止条件，网络训练完毕，一般来说可能是满足多少的epoch就停止，对于图像处理来说，如果是full batch训练那么一次的参数更新step就是一个epoch，如果不是那么一个epoch中的参数更新次数取决于batch的大小以及训练样本总数，一个epoch实际上就是跑完一次全部的数据集的过程,对于大量的样本 我们如果每次都计算所有样本的梯度然后取平均值实在是太慢了 所以就使用batch做划分

> [!question] 神经网络学习的本质

？？？给予一个随机混乱的图像 他依旧可以给出一个自信且确定的答案！？
给出答案我不例外 因为网络最终一定要有一个输出的，但是这个输出的激活值为什么会远大于其他？这背后的道理？
因为网络只会做分类 不会去思考这是不是一个正常的数字...另外学习的时候也并不是像是每个神经元精确的去识别一个”短边“，前面的解释只是方便理解，它识别的可能只是一种“模式”，一种”组合“，因为约束参数更新的目前只有loss，只要loss下降了对于网络来说就行了...识别到啥程度是不是“短边”网络是不在乎的 也许只是一种模糊的“模式”，但是足够使loss降低，就可以了
拿到一个混乱的图像，如果我们不加以其他的限制例如OOD detection,这个随机的噪声也许某一些模式刚好对应到网络学习的对应的数字区域，那么就会有偶然性的发生，将对应的神经元强烈的激活使得网络确定的自信的输出一个结果（该结果的激活值远大于其他）

*ChatGPT: 随机输入虽然在人类看来毫无意义，但它仍然会**经过网络学到的那些模式和决策边界**。有些随机输入可能让某个类别的内部得分远高于其他类别，因此最终表现出很高的“置信度”。*

也许回头再来看这个问题会有新的理解

接下来详细的说一说反向传播，也就是参数到底要怎么去修改更新
其实反向传播的结果在梯度那里已经写出来了
这里主要讲一讲这个$\nabla_{\theta} L$到底怎么计算的:
**链式法则+从输出层到输入层计算**
反向传播的优点就是对于靠前的参数 我们没有必要每次都从$L$重新计算起来，从他的前一层（从输出到输入来看）就能得到一部分的计算值
> [!warning] 符号问题
> 在每一层只有一个神经元的前两种情况下 并没有对层数进行角标的划分 只使用了一个角标！！！不要和后面的情况混淆符号！！！
### 1. 一个输入 每层一个神经元 
首先我们来看一个很简单的两层（一层隐藏 一层输出） 每层就一个神经元：
$$
z_{1} = w_{1}x+b_{1}
$$
$$
a_{1}=\sigma(z_{1})
$$
$$
z_{2}=w_{2}a_{1}+b_{2}
$$
$$
a_{2} = \sigma(z_{2})
$$
$$
L = \frac{1}{2}(a_{2}-y)^{2}
$$
我们如果需要计算：
$$
\frac{\partial L}{\partial w_{1}}
$$
根据链式法则他就等于是：
$$
\frac{\partial L}{\partial w_{1}} = \frac{\partial L}{\partial a_{2}}\frac{\partial a_{2}}{\partial z_{2}}\frac{\partial z_{2}}{\partial a_{1}}\frac{\partial a_{1}}{\partial z_{1}}\frac{\partial z_{1}}{\partial w_{1}}
$$
给出既定的x,y我们就可以通过求导法则计算出来$\frac{\partial L}{\partial w_{1}}$的值
### 2. 两个输入 每层一个神经元
接下来我们添加一个输入其余不变，也就是说多了一个权重：
$$
z_{1}= w_{1,1}x_{1}+w_{1,2}x_{2}+b_{1}
$$
那么计算:
$$
\frac{\partial L}{\partial w_{1,1}} ,\frac{\partial L}{\partial w_{1,2}}
$$
就是:
$$
\frac{\partial L}{\partial w_{1,1}} = \frac{\partial L}{\partial a_{2}}\frac{\partial a_{2}}{\partial z_{2}}\frac{\partial z_{2}}{\partial a_{1}}\frac{\partial a_{1}}{\partial z_{1}}\frac{\partial z_{1}}{\partial w_{1,1}}
$$
$$
\frac{\partial L}{\partial w_{1,2}} = \frac{\partial L}{\partial a_{2}}\frac{\partial a_{2}}{\partial z_{2}}\frac{\partial z_{2}}{\partial a_{1}}\frac{\partial a_{1}}{\partial z_{1}}\frac{\partial z_{1}}{\partial w_{1,2}}
$$
我们如果记：
$$
\frac{\partial L}{\partial z_{1}} = \delta_{1}
$$

$$
\frac{\partial L}{\partial z_{2}} = \delta_{2}
$$
那么我们可以发现：
$$
\frac{\partial L}{\partial w_{1,1}} = \delta_{1}\frac{\partial z_{1}}{\partial w_{1,1}}
$$
$$
\frac{\partial L}{\partial w_{1,2}} = \delta_{1}\frac{\partial z_{1}}{\partial w_{1,2}}
$$
也就是说我们在计算第一层的一个神经元的两个权重关于损失函数的导数的时候我们不需要两次都从$L$到$w_{i}$都计算一遍 我们计算了$w_{1,1}$时在中间保存$\delta_{1}$即可，计算$w_{1,2}$的时候直接使用$\delta_{1}$就可以了
另外有：
$$
\frac{\partial L}{\partial b_{1}} = \delta_{1}\frac{\partial z_{1}}{\partial b_{1}}
$$
$$
\frac{\partial L}{\partial b_{2}} = \delta_{2}\frac{\partial z_{2}}{\partial b_{2}}
$$
$$
\frac{\partial L}{\partial w_{2}} = \delta_{2}\frac{\partial z_{2}}{\partial w_{2}} 
$$
### 3. 两个输入 第一层两个神经元 第二层一个神经元
那么表达式应该是：
隐藏层：
$$
z_{1}^{(1)} = w_{1,1}^{(1)}x_{1}+w_{1,2}^{(1)}x_{2}+b_{1}^{(1)}
$$
$$
z_{2}^{(1)} = w_{2,1}^{(1)}x_{1}+w_{2,2}^{(1)}x_{2}+b_{2}^{(1)}
$$
$$
a_{1}^{(1)} = \sigma (z_{1}^{(1)})
$$
$$
a_{2}^{(1)} = \sigma (z_{2}^{(1)})
$$
输出层：
$$
z_{1}^{(2)} = w_{1,1}^{(2)}a_{1}^{(1)}+w_{1,2}^{(2)}a_{2}^{(1)}+b_{1}^{(2)}
$$
$$
a_{1}^{(2)} = \sigma(z_{1}^{(2)})
$$
损失函数：
$$
L = \frac{1}{2}(a_{1}^{(2)}-y)^{2}
$$
接下来我们推一下梯度：
输出层：
$$
\frac{\partial L}{\partial w_{1,1}^{(2)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}} \frac{\partial z_{1}^{(2)}}{\partial w_{1,1}^{(2)}} 
$$
$$
\frac{\partial L}{\partial w_{1,2}^{(2)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}} \frac{\partial z_{1}^{(2)}}{\partial w_{1,2}^{(2)}} 
$$
我们记：
$$
\delta_{1}^{(2)} = \frac{\partial L}{\partial z_{1}^{(2)}}=\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}
$$
我们就有：
$$
\frac{\partial L}{\partial w_{1,1}^{(2)}} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial w_{1,1}^{(2)}}
$$
$$
\frac{\partial L}{\partial w_{1,2}^{(2)}} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial w_{1,2}^{(2)}}
$$
这里我们看不出什么来 这其实就是第二种情况 一个神经元 两个输入 只不过这里的输入不是x，而是上一层的两个神经元
接下来我们看隐藏层：
$$
\frac{\partial L}{\partial w^{(1)}_{1,1}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{1,2}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,1}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,2}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}}
$$
我们记：
$$
\delta_{1}^{(1)} = \frac{\partial L}{\partial z_{1}^{(1)}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
$$
$$
\delta_{2}^{(1)} = \frac{\partial L}{\partial z_{2}^{(1)}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
$$
就有:
$$
\frac{\partial L}{\partial w^{(1)}_{1,1}} = \delta_{1}^{(1)}\frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{1,2}} = \delta_{1}^{(1)}\frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,1}} = \delta_{2}^{(1)}\frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,2}} = \delta_{2}^{(1)}\frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}}
$$
 我们有：
 $$
\delta_{1}^{(2)} = \frac{\partial L}{\partial z_{1}^{(2)}}=\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}} = (a_{1}^{(2)}-y)\sigma^{'}(z_{1}^{(2)})
$$
$$
\delta_{1}^{(1)} = \frac{\partial L}{\partial z_{1}^{(1)}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
$$
$$
\delta_{2}^{(1)} = \frac{\partial L}{\partial z_{2}^{(1)}} = \frac{\partial L}{\partial a^{(2)}_{1}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
$$

即：
$$
\delta_{1}^{(1)} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}} = \delta_{1}^{(2)}w_{1,1}^{(2)}\sigma^{'}(z_{1}^{(1)})
$$
$$
\delta_{2}^{(1)} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}} = \delta_{1}^{(2)}w_{1,2}^{(2)}\sigma^{'}(z_{2}^{(1)})
$$
可以发现：
**某一层的${}\delta{}$ = 上一层（输出${}\to{}$输入）的${}\delta{}$${}\times{}$对应的神经元权重${}\times{}$当前激活函数的导数（第四种情况时需要升级结论 因为某一层的神经元要受上一层的多个神经元的影响）**
**每一个${}\delta{}$都对应一个神经元**
计算权重参数:
$$
\frac{\partial L}{\partial w^{(1)}_{1,1}} = \delta_{1}^{(1)}\frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}} = \delta_{1}^{(1)}x_{1}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{1,2}} = \delta_{1}^{(1)}\frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}} = \delta_{1}^{(1)}x_{2}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,1}} = \delta_{2}^{(1)}\frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}} = \delta_{2}^{(1)}x_{1}
$$
$$
\frac{\partial L}{\partial w^{(1)}_{2,2}} = \delta_{2}^{(1)}\frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}} = \delta_{2}^{(1)}x_{2}
$$
$$
\frac{\partial L}{\partial w_{1,1}^{(2)}} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial w_{1,1}^{(2)}} = \delta_{1}^{(2)}a_{1}^{(1)}
$$
$$
\frac{\partial L}{\partial w_{1,2}^{(2)}} = \delta_{1}^{(2)}\frac{\partial z_{1}^{(2)}}{\partial w_{1,2}^{(2)}}= \delta_{1}^{(2)}a_{2}^{(1)}
$$
可以发现：
**权重参数的梯度 = 权重”指向“的神经元${}\delta{}$${}\times{}$权重“来源”的神经元激活值**
可以写成：
$$
\frac{\partial L}{\partial w^{(l)}_{j,i}} = \delta^{(l)}_{j}a^{(l-1)}_{i}
$$
### 4.两个输入 两层都为两个神经元 
我们先写前向传播：
隐藏层：
$$
z_{1}^{(1)} = w_{1,1}^{(1)}x_{1}+w_{1,2}^{(1)}x_{2}+b_{1}^{(1)}
$$
$$
z_{2}^{(1)} = w_{2,1}^{(1)}x_{1}+w_{2,2}^{(1)}x_{2}+b_{2}^{(1)}
$$

$$
a_{1}^{(1)} = \sigma(z_{1}^{(1)})
$$
$$
a_{2}^{(1)} = \sigma(z_{2}^{(1)})
$$
输出层：
$$
z_{1}^{(2)} = w_{1,1}^{(2)}a_{1}^{(1)}+w_{1,2}^{(2)}a_{2}^{(1)}+b_{1}^{(2)}
$$
$$
z_{2}^{(2)} = w_{2,1}^{(2)}a_{1}^{(1)}+w_{2,2}^{(2)}a_{2}^{(1)}+b_{2}^{(2)}
$$

$$
a_{1}^{(2)} = \sigma(z_{1}^{(2)})
$$
$$
a_{2}^{(2)} = \sigma(z_{2}^{(2)})
$$
损失函数,我们给定目标的标签${}y_{1},y_{2}{}$：
$$
L = \frac{1}{2}(a_{1}^{(2)}-y_{1})^{2}+\frac{1}{2}(a_{2}^{(2)}-y_{2})^{2}
$$
我们现在来计算梯度:
输出层：
$$
\frac{\partial L }{\partial w_{1,1}^{(2)}} = \frac{\partial L }{\partial a_{1}^{(2)}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial w_{1,1}^{(2)}}
$$
$$
\frac{\partial L }{\partial w_{1,2}^{(2)}} = \frac{\partial L }{\partial a_{1}^{(2)}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial w_{1,2}^{(2)}}
$$
$$
\frac{\partial L }{\partial w_{2,1}^{(2)}} = \frac{\partial L }{\partial a_{2}^{(2)}}\frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial w_{2,1}^{(2)}}
$$
$$
\frac{\partial L }{\partial w_{2,2}^{(2)}} = \frac{\partial L }{\partial a_{2}^{(2)}}\frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial w_{2,2}^{(2)}}
$$
我们记输出层的两个神经元的${}\delta{}$为：
$$
\delta_{1}^{(2)} = \frac{\partial L}{\partial z_{1}^{(2)}}=\frac{\partial L }{\partial a_{1}^{(2)}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}
$$
$$
\delta_{2}^{(2)} = \frac{\partial L}{\partial z_{2}^{(2)}} = \frac{\partial L }{\partial a_{2}^{(2)}}\frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}
$$
输出层的权重参数梯度可以写为：
$$
\frac{\partial L }{\partial w_{1,1}^{(2)}} = \delta_{1}^{(2)} \frac{\partial z_{1}^{(2)}}{\partial w_{1,1}^{(2)}}
=
\delta_{1}^{(2)}a_{1}^{(1)}
$$
$$
\frac{\partial L }{\partial w_{1,2}^{(2)}} = \delta_{1}^{(2)} \frac{\partial z_{1}^{(2)}}{\partial w_{1,2}^{(2)}}
=
\delta_{1}^{(2)}a_{2}^{(1)}
$$
$$
\frac{\partial L }{\partial w_{2,1}^{(2)}} = \delta_{2}^{(2)} \frac{\partial z_{2}^{(2)}}{\partial w_{2,1}^{(2)}}
=
\delta_{2}^{(2)}a_{1}^{(1)}
$$
$$
\frac{\partial L }{\partial w_{2,2}^{(2)}} =  \delta_{2}^{(2)} \frac{\partial z_{2}^{(2)}}{\partial w_{2,2}^{(2)}}
=
\delta_{2}^{(2)}a_{2}^{(1)}
$$
隐藏层（${}\sum{}$即将出场）：
$$
\frac{\partial L}{\partial w_{1,1}^{(1)}} = 
\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w_{1,2}^{(1)}} = 
\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}\frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}}
$$
$$
\frac{\partial L}{\partial w_{2,1}^{(1)}} = 
\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}}
$$
$$
\frac{\partial L}{\partial w_{2,2}^{(1)}} = 
\frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}\frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}}
$$
我们记隐藏层的两个神经元的${}\delta{}$为：
$$
\delta_{1}^{(1)} = \frac{\partial L}{\partial z_{1}^{(1)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
$$
$$
\delta_{2}^{(1)} = \frac{\partial L}{\partial z_{2}^{(1)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
$$
隐藏层的权重参数梯度可以写成：
$$
\frac{\partial L}{\partial w_{1,1}^{(1)}} = \delta_{1}^{(1)} \frac{\partial z_{1}^{(1)}}{\partial w_{1,1}^{(1)}}
=
\delta_{1}^{(1)}x_{1}
$$
$$
\frac{\partial L}{\partial w_{1,2}^{(1)}} = \delta_{1}^{(1)} \frac{\partial z_{1}^{(1)}}{\partial w_{1,2}^{(1)}}
=
\delta_{1}^{(1)}x_{2}
$$
$$
\frac{\partial L}{\partial w_{2,1}^{(1)}} = \delta_{2}^{(1)} \frac{\partial z_{2}^{(1)}}{\partial w_{2,1}^{(1)}}
=
\delta_{2}^{(1)}x_{1}
$$
$$
\frac{\partial L}{\partial w_{2,2}^{(1)}} = \delta_{2}^{(1)} \frac{\partial z_{2}^{(1)}}{\partial w_{2,2}^{(1)}}
=
\delta_{2}^{(1)}x_{2}
$$
我们反向传播的优点前面提到过就是一个${}\delta{}$可以复用 那我们从输出往输入算 我们看一下输出层和隐藏层的${}\delta{}$有什么数学关系：
$$
\delta_{1}^{(2)} = \frac{\partial L}{\partial z_{1}^{(2)}}=\frac{\partial L }{\partial a_{1}^{(2)}}\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}
$$
$$
\delta_{2}^{(2)} = \frac{\partial L}{\partial z_{2}^{(2)}} = \frac{\partial L }{\partial a_{2}^{(2)}}\frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}
$$
$$
\delta_{1}^{(1)} = \frac{\partial L}{\partial z_{1}^{(1)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}} 
$$
$$
\delta_{2}^{(1)} = \frac{\partial L}{\partial z_{2}^{(1)}} = \frac{\partial L}{\partial a_{1}^{(2)}} \frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
+
\frac{\partial L}{\partial a_{2}^{(2)}} \frac{\partial a_{2}^{(2)}}{\partial z_{2}^{(2)}}\frac{\partial z_{2}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
$$

$$
\delta_{1}^{(1)} 
= 
\delta_{1}^{(2)} \frac{\partial z_{1}^{(2)}}{\partial a_{1}^{(1)}}\frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
+
\delta_{2}^{(2)} \frac{\partial z_{2}^{(2)}}{\partial a_{1}^{(1)}} \frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}}
=
\delta_{1}^{(2)} w_{1,1}^{(2)} \sigma^{'}(z_{1}^{(1)})
+
\delta_{2}^{(2)} w_{2,1}^{(2)} \sigma^{'}(z_{1}^{(1)})
$$

$$
\delta_{1}^{(1)} 
= 
(\delta_{1}^{(2)} w_{1,1}^{(2)}
+
\delta_{2}^{(2)} w_{2,1}^{(2)})
\sigma^{'}(z_{1}^{(1)})
$$

$$
\delta_{2}^{(1)} 
= 
\delta_{1}^{(2)} \frac{\partial z_{1}^{(2)}}{\partial a_{2}^{(1)}}\frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
+
\delta_{2}^{(2)} \frac{\partial z_{2}^{(2)}}{\partial a_{2}^{(1)}} \frac{\partial a_{2}^{(1)}}{\partial z_{2}^{(1)}}
=
\delta_{1}^{(2)} w_{1,2}^{(2)} \sigma^{'}(z_{2}^{(1)})
+
\delta_{2}^{(2)} w_{2,2}^{(2)} \sigma^{'}(z_{2}^{(1)})
$$

$$
\delta_{2}^{(1)} 
= 
(\delta_{1}^{(2)} w_{1,2}^{(2)}
+
\delta_{2}^{(2)} w_{2,2}^{(2)})
\sigma^{'}(z_{2}^{(1)})
$$

我们就能看出来：
第${}l{}$层的第${}i{}$个神经元的${}\delta_{i}^{(l)}{}$等于这个神经元的加权和的激活函数导数${}\times{}$ 
（第${}l+1{}$层的${}k{}$个神经元，每一个神经元的${}\delta_{j}^{(l+1)}{}$${}\times{}$第${}l{}$层第${}i{}$个神经元到${}l+1{}$层第${}j{}$个神经元的权重再求和）
$$
\delta^{(l)}_{i} = \sigma^{'}(z_{i}^{(l)})\sum_{j=1}^{k}(\delta^{(l+1)}_{j} w_{j,i}^{(l+1)})
$$
写成矩阵形式：
第${}l{}$层有${}n{}$个神经元
$$
\delta^{(l)} = 
\begin{bmatrix}
\delta^{(l)}_{1} \\
\delta^{(l)}_{2} \\
. \\
. \\
\delta^{(l)}_{n}
\end{bmatrix}_{n \times 1}
$$
$$
\sigma^{'}(z^{(l)})=
\begin{bmatrix}
\sigma^{'}(z^{(l)}_{1}) \\
\sigma^{'}(z^{(l)}_{2}) \\
. \\
. \\
\sigma^{'}(z^{(l)}_{t})
\end{bmatrix}_{n \times 1}
$$
第${}l+1{}$层有${}k{}$个神经元
$$
\delta^{(l+1)} = 
\begin{bmatrix}
\delta^{(l+1)}_{1} \\
\delta^{(l+1)}_{2} \\
. \\
. \\
\delta^{(l+1)}_{k}
\end{bmatrix}_{k \times 1}
$$
我们之前有：
$$
\begin{bmatrix}
a^{(1)}_{1} \\
a^{(1)}_{2} \\
. \\
. \\
a^{(1)}_{k}
\end{bmatrix}_{k\times 1}=\sigma(\begin{bmatrix}
w_{1,1}&w_{1,2}&w_{1,3}&\dots&w_{1,n} \\
w_{2,1}&w_{2,2}&w_{2,3}&\dots&w_{2,n} \\
\dots&\dots&\dots&\dots&\dots \\
w_{k,1}&w_{k,2}&w_{k,3}&\dots&w_{k,n}
\end{bmatrix}_{{k}\times n}\times \begin{bmatrix}
a^{(0)}_{1} \\
a^{(0)}_{2} \\
. \\
. \\
a^{(0)}_{n}
\end{bmatrix}_{n\times 1} + \begin{bmatrix}
b_{1} \\
b_{2} \\
. \\
. \\
b_{k}
\end{bmatrix}_{k\times 1})
$$
$$
a^{(1)} = \sigma(W\times a^{(0)}+b)
$$
但这是前向传播的公式
反向传播时我们将${}W{}$转置：
$$
W^{T} = 
\begin{bmatrix}
w_{1,1}&w_{2,1}&\dots &w_{k,1} \\
w_{1,2}&w_{2,2}&\dots &w_{k,2} \\
\dots&\dots&\dots&\dots \\
w_{1,n}&w_{2,n}&\dots&w_{k,n}
\end{bmatrix}_{n \times k}
$$
就有：
$$
\delta^{(l)} = (W^{(l+1)})^{T}\delta^{(l+1)}\odot\sigma^{'}(z^{(l)})
$$

## 核心结论(单样本)
$$
a^{(l+1)} = \sigma(W^{(l+1)}\times a^{(l)}+b)
$$
输出层的误差信号需要根据loss单独计算
$$
\delta^{(l)}_{i} = \sigma^{'}(z_{i}^{(l)})\sum_{j=1}^{k}(\delta^{(l+1)}_{j} w_{j,i}^{(l+1)})
$$
$$
\delta^{(l)} = (W^{(l+1)})^{T}\delta^{(l+1)}\odot\sigma^{'}(z^{(l)})
$$
$$
\frac{\partial L}{\partial w^{(l)}_{j,i}} = \delta^{(l)}_{j}a^{(l-1)}_{i}
$$
$$
dw^{(l)} = \delta^{(l)}\times (a^{(l-1)})^{T}
$$
$$
\frac{\partial L}{\partial b^{(l)}_{j}} = \delta^{(l)}_{j}
$$
$$
db^{(l)} = \delta^{(l)}
$$
$$
\theta_{i} \leftarrow \theta_{i} - \eta \frac{\partial L}{\partial \theta_{i}}
$$
