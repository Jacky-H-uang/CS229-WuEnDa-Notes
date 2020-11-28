# Logistic Regression
### 对于LR算法 用来解决 Classification 等问题
1. Logistic Regression 的假设函数和 cost-function
   
   假设函数：hΘ(x) = g(Θ^T X) = 1 / (1 + e(-Θ^T X))
   
   其中的 g(z) = 1 / (1 + e(-Θ^T X)) --- sigmoid 函数
    
   cost-function: J(θ)=1/m ∑i=1~m(-ylog(hθ(x))-(1-y)log(1-hθ(x)))
   
   
2. 解决分类问题 , 判别概率问题。
    
    if y == 1 : P(y = 1 | x;Θ) = hΘ(x)
    
    if y == 0 : P(y = 0 | x;Θ) = 1 - hΘ(x)
    
    所以：P(y | x;Θ) = h(x)^y * (1-h(x))^(1-y)
    y 属于 {0,1}
    
3. 使用 MLE 和 “log likehood”
    
   经过公式的推导得出：
   
   Θj := Θj + a ∑i=1~M (y(i) - hΘ(x(i)))xj(i)
   (其中的 a 为 learning rate)
   
   与梯度下降的公式如出一辙, 但是关于 Θ 的定义不同。
   
   所以可以采用类似梯度下降的梯度上升来求解最佳的Θ。

4. 得出逻辑回归的流程：
    
    初始化线性参数为1
    
    构造 sigmoid 函数
    
    循环计算：
        
        计算数据集梯度
        update 数据集梯度
    得出 sigmoid函数
    
    训练算法
    
    最后用 sigmoid 函数去解决classification 问题