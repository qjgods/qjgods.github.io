---
typora-copy-images-to: .\
layout: post
title: Computational Learning Theory and Model Selection
description: "本文介绍了计算学习理论笔记"
tags: 机器学习
---


## True vs. Empirical Risk 

当你有一个函数f的时候，你自然就想知道它到底是好还是坏，那么怎么评价呢？我们可以用一个叫risk的东西来衡量这个函数的风险，这里的risk可以直观地看做你函数的**错误率**。然而，根据no free lunch定理，显然没有一个函数永远是最优的，那这个risk到底是什么？它其实是在你数据分布$\mathcal{D}$上的风险。而我们通常用empirical risk来对true risk进行估计，也就是对错误率估计,更一般empirical risk可以称为**经验误差**，当你的样本量越多的时候，你对函数f的风险的估计就越准确。

True Risk:

- Classification - 误分类概率 $P(f(X)\ne Y)$
- Regression - Mean Squared Error $\mathbb{E}[(f(X)-Y)^2]$


Empirical Risk:

- Classification - $\frac{1}{n}\sum_{i=1}^n1_{f(X_i\ne Y_i)}$
- Regression -  $\frac{1}{n}\sum_{i=1}^n{(f(X_i)- Y_i)^2}$



%可是我们仅仅用经验误差是不够的，因为经验误差只考虑了“训练集”的误差，我们可以很容易构造出一个函数使得他完美拟合所有数据。所以我们还要考虑泛化误差才行。



我们设存在一个完美的risk,$R^\*$ ,还有一个从n个数据中估计出来的f以及它对应的risk，$E[R(\hat{f_n})]$，我们定义一个Excess risk为$E[R(\hat{f_n})]-R^*$,于是这个excess risk可以分解为两部分，分别是estimation error 和 approximation error.

![1518535927188](1518535927188.png)

直观来看，estimation error，是因为缺少足够的样本，从而导致我们从函数族$\mathcal{F}$选择模型时没法取得最优的模型而产生的误差。而approximation error是由于函数族$\mathcal{F}$的限制而产生的误差，比如说线性回归，由于限制在了线性的空间中，而对非线性的数据存在误差。

简单地说，estimation error是样本的问题，approximation error是模型复杂度的问题。

然而模型越复杂，所需的样本也就越多，这就形成一个平衡，你需要一个合适的hypothese class $\mathcal{F}$大小。

![1518535937954](1518535937954.png)




## Learning Theory 

Empirical Risk Minimization 

![1518537111943](1518537111943.png)

三个重要的引理

![1518537123984](1518537123984.png)

![1518881523085](1518881523085.png)

使用这几个引理可以让我们证明在learning theory中非常重要的结论。第三个定理是，一般套路就是考虑只有一个样本改变后两个相减的差，小于某个d，然后就可以套这个不等式，然后令最右边的是$\delta$，就得出了一个界。

现在我们来定义一下Hypothesis Class，数据集不同的划分方式其实就对应了不同的假设类，而所有的划分方法就组成了假设类$\mathcal{H}$. 这个是跟，“概念类”对应的一个概念，所谓概念类其实就是数据正确的标签$\mathcal{C}$.，所以我们要做的事情就是搜索出一个最优的假设h，使得逼近正确的标签。



### Finite Hypothesis Space

对于有限的假设空间，有以下定理

![1518881888335](1518881888335.png)

上面这个引理是由hoeffding 不等式的出来的，因为这里的$\hat{E}$对应的是经验误差，是对E的近似，所以直接用hoeffding不等式。

![1518877689847](1518877689847.png)

令$\delta=2|\mathcal{H}|\exp(-2m\epsilon^2)$即可得12.19

上面定理用了引理1的集合的不等式。通过最下面的这个界可以看到，当$|H|$越大的时候这个界是越大的。

### Infinite Hypothesis Space

当假设类是无限时，我们可以构造出一个叫增长函数的东西，用来表示假设空间H对m个样本所能赋予的最大可能结果数。

![1518882559587](1518882559587.png)

如果我们的假设空间能够对这所有m个样本赋予任意的标签，那么H在m个样本下是可以打散这个数据集D的，那么我们定义那个可以打散的m的最大值，称为H的VC维。也就是说，VC维是用于衡量假设空间$|H|$大小的一个东西。

![1518882573822](1518882573822.png)

这样我们就能用VC维来构造出经验误差的界,VC维有点像有限假设类下的类比。

![1518882637789](1518882637789.png)



###  Rademacher 复杂度

Rademacher 复杂度，他是对VC维的一个改进，因为VC维在刻画假设空间的大小时，并没有考虑数据的分布，这使得他的界非常松，实际意义比较小。

![1518882752502](1518882752502.png)

这里$y_i$取1，-1 ，然而这里yi其实是指现实中的值，而如果考虑更一般的情况，我们将y_i换成一个1，-1的随机变量，于是，问题转化为，我们希望找到一个h，使得这个最大（等价于经验误差最小）

![1518882879727](1518882879727.png)

这里我的理解是，因为sup是在$\sigma_i$外面的，所以当我们选择h的时候，$\sigma_i$相当于是固定的，也就是可以看做是一个随机标签的样本，于是如果我们总能找到一个h使得每个$h(x_i)=\sigma_i$，那么这个东西就等于1，那这个假设空间就很棒。

![1518883423948](1518883423948.png)

最后这里考虑的是在数据集D下m个样本的函数空间F的复杂度，它的做法就是对所有从D中采样，而且是m个样本的数据集Z进行积分，也就是考虑所有m个样本的组合，然后求均值，最后的就是要求的平均复杂度，我们可以通过这个函数空间的复杂度来给出这泛化误差的界。





# Reference



[Computational Learning Theory](http://www.cs.cmu.edu/~epxing/Class/10701-11f/recitation/recitation_4_Bin.pdf)

[What does the term “Estimation error” mean?](https://stats.stackexchange.com/questions/87750/what-does-the-term-estimation-error-mean)

[Excess Error, Approximation Error, and Estimation Error](http://drona.csa.iisc.ernet.in/~shivani/Teaching/E0370/Aug-2011/Lectures/10.pdf)

[如何通俗的理解机器学习中的VC维、shatter和break point](https://www.zhihu.com/question/38607822)

