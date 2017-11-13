#### **mini-batch梯度下降法**

- 及其学习一般伴随着巨大的计算量和训练许多模型才可以找到合理的模型，在大数据时代，如果可以提高算法的计算速度可以很好的提高工作效率，因此，介绍了mini-batch
- 要求快速训练，得到验证结果
  - 如果样本太大，如果全样本计算一次，则太耗时间
  - 因此，我们就需要寻找合适的方法去提高速度
- 但是如果每次处理训练数据的一部分，即用其子集进行梯度下降，则我们的算法速度会执行的更快。而处理的这些一小部分训练子集即称为Mini-batch。

**Batch vs Stochastic vs Mini-batch gradient descent** 

- mini-batch的核心算法如图![Image text](https://github.com/yinjw1995/neural_network/raw/master/note_pictures\1.jpg)

- 如果考虑极端的情况，每一个batch的尺寸为m，则只有一个，我们可以称之为batch梯度下降，如果是尺寸为1，则有m个，这就是随机梯度下降

  - Batch gradient descent 是对全部样本做backward prop，比如，5000000样本，vectorization能高效训练 (forward, backward) 得到最终损失值，但是计算一次损失函数值所用的时间太长
  - stochastic gradient descent 针对单一样本做backward prop，很快看见损失值，因为只算一个样本，所以导致每计算出来的梯度并不一定是指向全局最低的，只是大致朝着减小的方向，因此包含很多噪声，并且最终不会收敛，只是在一个较小值附近波动。同样降低了效率
  - 因此，合理的选择mini-batch的尺寸可以使得效率最高

  ![2](https://github.com/yinjw1995/neural_network/raw/master/note_pictures\2.png)

  ![3](https://github.com/yinjw1995/neural_network/raw/master/note_pictures\3.png)

- **Mini-batch 大小的选择**

  - 如果训练样本的大小比较小时，如 $m<2000$ 时 ------ 选择batch梯度下降法；
  - 如果训练样本的大小比较大时，典型的大小为： $2^6、2^7、，，，、2^{10}$ ；
  - Mini-batch的大小要符合CPU/GPU内存。

  ![4](https://github.com/yinjw1995/neural_network/raw/master/note_pictures\4.png)

  ​
