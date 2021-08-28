[TOC]



# P3  1:Recognition-Case Study <!--比较重要，都会 85'-->

## <u>Regression: Output a scalar</u>

## 应用场景

1. Stock Market Forecast  股票预测
2. Self-driving Car 自动驾驶,根据观测到的结果判断方向盘的角度
3. Recommendation 商品推荐，使用者A购买商品B的可能性

## eg.Estimating the Combat Power (CP) of a pokemon after evolution 预测宝可梦的进化后的CP(战斗能力)值

- input:妙蛙种子*X*进化前的数值 目前战斗力  *X**c**p* ,   物种*Xs*,  HP值即生命值 *Xhp*,  体重*Xw* ,  身高*Xh*
- **output:进化后的 *CP* 值 y（数值型）**  ,想要培养cp值高的宝可梦

<img src="3-1.PNG" alt="3-1" style="zoom:60%;" />

### step 1 : Model

1. **A set of Function** ，里面有无穷多个 function

   假设  ***y*=*b*+*w*⋅*Xcp***（w and b are parameters (can be any value)）

2. **Linear model**  

$$
y=b+∑w_ix_i
$$

​			$𝑥_𝑖 $: an attribute of input :$𝑥_{𝑐𝑝}, 𝑥_h𝑝, 𝑥_𝑤, 𝑥_h$ ... feature

​			$𝑤_𝑖$  : weight

​			b : bias

### Step 2: Goodness of Function 衡量model的好坏

真的输入一些宝可梦进化前/后的数值，用实际的 y 和预测的 y_hat  计算  **Loss function 𝐿**

​	Input : a function

​	output : how bad it is
$$
L(f) = \sum_{n=1}^{10}(\hat y^n - f(x_{cp} ^n))^2 => L(w,b) = \sum_{n=1}^{10}(\hat y^n - (b+w\cdot x_{cp} ^n))^2
$$

### Step 3 : Pick the Best Function

**寻找一个 Best Function (也就是 f*),使损失函数L(f)最小**

<img src="3-2.PNG" alt="3-2" style="zoom:50%;" />

## Step 3: Gradient Descent 

简化问题，只用一个参数w,找出w,使L(w)最小
$$
W^*=arg\,\min_{w}L(w)
$$

### 梯度下降步骤

1. (Randomly) Pick an initial value   *w^0*, 随机找一个初始  *w^0*

2. 计算在  *w^0* 处，参数 *w* 对 *L* 的微分，找一下切线的斜率

   1. 斜率是负的==>增加 *w*
   2. 斜率是正的==>减小 *w*

   $$
   W^1 \leftarrow W^0 - \eta{dL\over dw}| _{w=w^0}
   $$

   

3. 新 w^0 ==> w^1  ( **η is called “learning rate”决定了学习的速度有多快**) 

   1. 计算在w^1处微分，计算得到 w^2 
   2. …
   3. **直到找到 w^T，此处参数w对L的微分为0，微分是0 就没办法再动**
      *（saddle point 驻点也会停下来，或者很平的地方plateau）*

   **我们很容易找到 Local minima ，不一定找到 global minima**

<img src="3-3.PNG" alt="3-3" style="zoom:60%;" />

#### *两个参数就对两个参数同时做偏导,随机选取两个初始值 w^0和 b^0*

<img src="3-4.PNG" alt="3-4" style="zoom:60%;" />

#### 微分为0的点的问题

理论上：每一次用Gradient Descent ，更新参数之后，𝜃  都会使 L(𝜃)   越来越小
$$
𝐿(𝜃^0) > 𝐿(𝜃^1) > 𝐿(𝜃^2) > ⋯
$$
直到我们没有办法使loss变得更小，就结束这个回合（这个假设不会永远正确）

<img src="3-5.PNG" alt="3-5" style="zoom:67%;" />

微分是 0 的点不是只有**global minima** (Local minima)

**saddle point** (驻点) 微分也是 0

很平的地方 **plateau**,微分很小的地方也会停下来

#### 计算 𝜕𝐿/𝜕𝑤   and   𝜕𝐿/𝜕𝑏  

<img src="3-6.PNG" alt="3-6" style="zoom:67%;" />

## Result

得出结果后,再抓10只宝可梦,预测其进化后的cp值,进行测试,计算误差，

### 一般测试集误差略大于训练集是正常的

<img src="\3-7.PNG" alt="3-7" style="zoom:60%;" />

### 可以尝试引入2次项,3次项... (复杂的式子会包含简单的式子)

当model越来越复杂，训练集上error会越来越小，**但过于复杂会过拟合，**error反而变大

<img src="3-8.PNG" alt="3-8" style="zoom:80%;" />



## 如何解决 Over fitting ==> 收集更多的数据

### Back to step 1:<u>重新设计model,</u>

- 搜集更多的宝可梦之后发现，宝可梦的种类影响了进化后的 c p值,根据不同种类宝可梦,计算不同的model的参数,结果会变好

<img src="3-10.PNG" alt="3-10" style="zoom:50%;" />



- 考虑是否有其他不同的 factors 影响结果，**把每个特征和y做一个x-y的图，查看相关性**

### **Back to step 2: Regularization 正则化**

*损失函数不仅让error最小，还要让**参数平方之和很小**（平滑的function 对噪音不敏感）*

<img src="3-12.PNG" alt="3-12" style="zoom:60%;" />

==> **意味着这是一个平滑的function ,平滑的function更可能正确**

**𝜆 是超参**，𝜆 越大越平滑，太平滑也不好，太平滑就是一条直线了，b 和平滑没有关系

## Conclusion

1. Pokémon : 宝可梦进化后的 c p 值和进化前的 c p 和种类有关
   - There are probably other hidden factors 
2. Gradient descent 梯度下降
3. We finally get average error = 11.1 on the testing data  
   -  How about new data? Larger error? Lower error?  
4. Next lecture: Where does the error come from?
   - More theory about overfitting and regularization
   - The concept of validation  