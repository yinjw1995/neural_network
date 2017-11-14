#### 神经网络概述

- 我们为什么要使用神经网络呢，我们需要了解下面六个问题：

  - 什么时候使用神经网络
  - 分类还是回归（classification or regression）
  - deterministic or stochastic
  - supervised or unsupervised
  - online or offline
  - pc还是其他硬件设备（DSP芯片）

- 什么时候使用神经网络

  - 没有数据模型，并且不需要100%的精确、
  - 有数学模型，但是太复杂了，很难应用并且不需要100%精确

- 分类还是回归？

  - 回归可以说是分类的一种特殊形式

- deterministic or stochastic

  输入参数的比重是确定的还是随机的

  ![1](F:\program\neural_network\MATLAB neural network of math\image\1.png)

- 监督还是非监督学习

  监督：确定输入输出之间的关系

  非监督：对于没有输出的数据进行分类

- 在线和离线

  一般都是offline

  ![2](F:\program\neural_network\MATLAB neural network of math\image\2.png)

- 自己的电脑可以设计比较复杂的程序，但是如果实在DSP芯片上需要简化神经网络

#### BP神经网络

- MATLAB中的BP神经网络时根据widrow-hoff的学习法则得到的多层非线性传递函数

- MATLAB中的函数：

  ```matlab
  net=newff(minmax(p),[10,3],'trainlm','learindm','mse')
  %第一项为限值，
  %第二项为网络结构，如[2,3,4,1]
  %第三项为训练函数，默认线性trainlm
  %权值学习函数，train，traingd（梯度下降）
  %性能函数，‘mes均方差函数’，‘msereg均方差规范函数’
  net.trainParam.epochs=200
  net.trainParam.goal=0.2
  net=train(net,p,t)
  %设置训练参数
  ```

  ​