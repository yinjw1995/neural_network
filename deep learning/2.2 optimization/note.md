#### **mini-batch梯度下降法**

- 及其学习一般伴随着巨大的计算量和训练许多模型才可以找到合理的模型，在大数据时代，如果可以提高算法的计算速度可以很好的提高工作效率，因此，介绍了mini-batch
- 要求快速训练，得到验证结果
  - 如果样本太大，如果全样本计算一次，则太耗时间
  - 因此，我们就需要寻找合适的方法去提高速度
- 但是如果每次处理训练数据的一部分，即用其子集进行梯度下降，则我们的算法速度会执行的更快。而处理的这些一小部分训练子集即称为Mini-batch。

**Batch vs Stochastic vs Mini-batch gradient descent** 

- mini-batch的核心算法如图![1](https://raw.githubusercontent.com/yinjw1995/neural_network/master/note_pictures/1.jpg)

- 如果考虑极端的情况，每一个batch的尺寸为m，则只有一个，我们可以称之为batch梯度下降，如果是尺寸为1，则有m个，这就是随机梯度下降

  - Batch gradient descent 是对全部样本做backward prop，比如，5000000样本，vectorization能高效训练 (forward, backward) 得到最终损失值，但是计算一次损失函数值所用的时间太长
  - stochastic gradient descent 针对单一样本做backward prop，很快看见损失值，因为只算一个样本，所以导致每计算出来的梯度并不一定是指向全局最低的，只是大致朝着减小的方向，因此包含很多噪声，并且最终不会收敛，只是在一个较小值附近波动。同样降低了效率
  - 因此，合理的选择mini-batch的尺寸可以使得效率最高

  ![2](https://raw.githubusercontent.com/yinjw1995/neural_network/master/note_pictures/2.png)

  ![3](https://raw.githubusercontent.com/yinjw1995/neural_network/master/note_pictures/3.png)

- **Mini-batch 大小的选择**

  - 如果训练样本的大小比较小时，如 $m<2000$ 时 ------ 选择batch梯度下降法；
  - 如果训练样本的大小比较大时，典型的大小为： $2^6、2^7、，，，、2^{10}$ ；
  - Mini-batch的大小要符合CPU/GPU内存。

  ![4](https://raw.githubusercontent.com/yinjw1995/neural_network/master/note_pictures/4.png)


#### 指数加权平均

- 指数加权平均的核心算法为

  $v_t=\beta v_{t-1}+(1-\beta)\theta_t$ 

- 以伦敦的天气为例，要计算出十日平均温度的大致曲线走势，则$\beta=0.9$ 
  $$
  v_{100}=0.9v_{99}+0.1\theta_{100}\\
  v_{99}=0.9v_{98}+0.1\theta_{99}\\
  v_{98}=0.9v_{97}+0.1\theta_{98}\\
  \vdots
  $$
  展开有：
  $$
  v_{100}=0.1\theta_{100}+0.9(0.1\theta{99}+0.9(0.1\theta_{98}+0.9v_{97}))\\
  =0.1\theta_{100}+0.1*0.8\theta_{99}+0.1*(0.9)^2\theta_{98}+\dots
  $$
  上式中所有 $\theta$ 前面的系数相加起来为1或者接近于1，称之为偏差修正。 

  总的来说，$(1-\epsilon)^{1/\epsilon}=\frac1e$ ，在我们的例子中$1-\epsilon =\beta=0.9$ ，即$0.9^{10}\approx0.35\approx\frac1e$ 。相当于十天前的数据，系数的峰值（这里是$0.1 $）下降到原来的$\frac1e$ ，因此，可以说只关注了十天的平均值。

  ![5](F:\program\neural_network\note_pictures\5.png)

  因此，在计算当前时刻的平均值，只需要前一天的平均值和当前时刻的值，所以在数据量非常大的情况下，指数加权平均在节约计算成本的方面是一种非常有效的方式，可以很大程度上减少计算机资源存储和内存的占用。

  当$\beta$ 分别为0.9、0.8、0.5时，指数加权平均数结果如图：

  ![6](F:\program\neural_network\note_pictures\6.jpg)

- 指数加权平均的偏差修正

  - 当我们执行上述公式是，假设$\beta =0.98$ ,实际得到的并不是图中的绿色曲线，而是下图中的紫色曲线，其起点比较低。

  ![7](F:\program\neural_network\note_pictures\7.jpg)

  原因：

  ![8](F:\program\neural_network\note_pictures\8.png)

  - 如果第一天的值为40，则$v_1=0.02*40=8$ ,得到的值要远小于实际值，后面几天的情况也会由于初值引起的影响，均低于实际均值。
  - 偏差修正：

  使用$v_1=0.02*40=8$

  当$t=2$ 时：

  $1-\beta^t=1-(0.98)^2=0.0396$

  $v_2=\frac{0.0196\theta_1+0.02\theta_2}{0.0396}$ 

  偏差修正得到了绿色的曲线，在开始的时候，能够得到比紫色曲线更好的计算平均的效果。随着 $t$ 逐渐增大， $\theta^t$ 接近于0，所以后面绿色的曲线和紫色的曲线逐渐重合了。

  虽然存在这种问题，但是在实际过程中，一般会忽略前期均值偏差的影响。

#### 动量（momentum）梯度下降法

- 动量梯度下降的基本思想就是计算梯度的指数加权平均数，并利用该梯度来更新权重。在我们优化 Cost function 的时候，以下图所示的函数图为例：

  ![9](F:\program\neural_network\note_pictures\9.png)

  在利用梯度下降法来最小化该函数的时候，每一次迭代所更新的代价函数值如图中蓝色线所示在上下波动，而这种幅度比较大波动，减缓了梯度下降的速度，而且我们只能使用一个较小的学习率来进行迭代。

  如果用较大的学习率，结果可能会如紫色线一样偏离函数的范围，所以为了避免这种情况，只能用较小的学习率。

  但是我们又希望在如图的纵轴方向梯度下降的缓慢一些，不要有如此大的上下波动，在横轴方向梯度下降的快速一些，使得能够更快的到达最小值点，而这里用动量梯度下降法既可以实现，如红色线所示。

- 算法实现：

  ![9](F:\program\neural_network\note_pictures\9.jpg)

  $\beta$ 最常用的值为0.9

  在我们进行动量梯度下降算法的时候，由于使用了指数加权平均的方法。原来在纵轴方向上的上下波动，经过平均以后，接近于0，纵轴上的波动变得非常的小；但在横轴方向上，所有的微分都指向横轴方向，因此其平均值仍然很大。最终实现红色线所示的梯度下降曲线。

  **算法本质解释**

  在对应上面的计算公式中，将Cost function想象为一个碗状，想象从顶部往下滚球，其中：

  - 微分项 $dw,db$ 想象为球提供的加速度；
  - 动量项 $v_{dw},v_{db}$ 相当于速度；

  小球在向下滚动的过程中，因为加速度的存在使得速度会变快，但是由于 $\beta$ 的存在，其值小于1，可以认为是摩擦力，所以球不会无限加速下去。

#### RMSprob

除了上面所说的**Momentum**梯度下降法，**RMSprop**（root mean square prop）也是一种可以加快梯度下降的算法。

算法如图所示：

![10](F:\program\neural_network\note_pictures\10.jpg)

这里假设参数b的梯度处于纵轴方向，参数w的梯度处于横轴方向（当然实际中是处于高维度的情况），利用RMSprop算法，可以减小某些维度梯度更新波动较大的情况，如图中蓝色线所示，使其梯度下降的速度变得更快，如图绿色线所示

可以说，RMSprop与上面的momentum类似，都是通过使用指数平均来减小振动幅值，如果$db^2$ 特别大，则会使得$\frac{db}{ \sqrt{s_{db}+\varepsilon}}$ 变小，也就是说，可以适当的增加$\alpha$,由于 $\frac{dw}{ \sqrt{s_{dw}+\varepsilon}}$ 变大了，乘以更大的$\alpha$使得$w$ 进一步的变大，从而提高了优化速度.