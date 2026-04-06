---
title: Mathematics
mathjax: true
date: 2020-12-20 20:50:44
categories: Course Notes
description: 高数公式、笔记等等杂项
---

## 函数与极限

### 函数的极限

#### 定义

极限：$\lim\limits_{x\to x_0}f(x)=A\Leftrightarrow ∀ε>0，∃δ>0，当0<|x-x_0|<δ时，有|f(x)-A|<ε$

左极限：$\lim\limits_{x\to x_0^-}f(x)=A\Leftrightarrow ∀ε>0，∃δ>0，当x_0-δ<x<x_0时，有|f(x)-A|<ε$

右极限：$\lim\limits_{x\to x_0^+}f(x)=A\Leftrightarrow ∀ε>0，∃δ>0，当x_0<x<x_0+δ时，有|f(x)-A|<ε$

#### 性质

存在的充要条件：$\lim\limits_{x\to x_0}f(x)=A\Leftrightarrow \lim\limits_{x\to x_0^-}f(x)=\lim\limits_{x\to x_0^+}f(x)=A$

唯一性：$\lim\limits_{x\to x_0}f(x)=A\Rightarrow 该极限唯一$

局部有界性：$\lim\limits_{x\to x_0}f(x)=A\Rightarrow ∀M>0，δ>0，0<|x-x_0|<δ，|f(x)|≤M$

局部保号性：$\lim\limits_{x\to x_0}f(x)=A>0\Rightarrow ∀δ>0，0<|x-x_0|<δ，f(x)>0$，小于0同理

### 极限运算法则

- **有限个**无穷小的和是无穷小（任意个为Undefined）
- **有限个**无穷小（大）的积是无穷小（大）（任意个为Undefined）
- 有界函数乘无穷小的积是无穷小

#### 线性运算性质

$\lim f(x)=A，\lim g(x)=B\Rightarrow\lim (f(x)±g(x))=A±B$

$\lim f(x)=A，\lim g(x)=B\Rightarrow\lim (f(x)g(x))=AB$

$\lim f(x)=A，\lim g(x)=B\neq 0\Rightarrow\lim\frac{f(x)}{g(x)}=\frac{A}{B}$

$\lim (cf(x))=c\lim f(x)$（c为常数）

$\lim (f(x))^n=(\lim f(x))^n$（n为正整数）

#### 括号穿透

$\lim\limits_{x\to x_0}f(g(x))=f(\lim\limits_{x\to x_0}g(x))$

### 极限存在准则 两个重要极限

#### 夹逼准则

$x∈\mathring{U}(x_0,r)，g(x)≤f(x)≤h(x)，且\lim\limits_{x\to x_0}g(x)=\lim\limits_{x\to x_0}h(x)=A\Rightarrow \lim\limits_{x\to x_0}f(x)=A$

#### 重要极限

- $\lim\limits_{x\to x_0}\frac{sinx}{x}=1$
> 证明：$sinx<x<tanx\Rightarrow cosx<\frac{sinx}{x}<1，\lim\limits_{x\to 0}cosx=1\Rightarrow 夹逼准则证出$

- $\lim\limits_{x\to ∞}(1+\frac{1}{x})^x=\lim\limits_{x\to 0}(1+x)^\frac{1}{x}=e$
> 证明：$(1+\frac{1}{n})^n二项式展开后\leq \sum_{i=0}^{n}\frac{1}{i!}\leq 2+\sum_{i=1}^{n}\frac{1}{2^i}=3-\frac{1}{2^(n-1)}\le 3$，证明极限存在，**定义其为e**

### 无穷小的比较

#### 常用等价无穷小

- $x \sim sinx \sim arcsinx \sim tanx \sim arctanx \sim ln(x+1) \sim e^x-1$
- $1-cosx \sim \frac{x^2}{2} \sim \frac{1}{cosx}-1$
- $(1+x)^α-1 \sim αx$
- $tanx-sinx \sim \frac{x^3}{2}$

### 连续函数的性质

#### 介值定理

$f(x)∈c[a,b]，f(a)=A，f(b)=B\Rightarrow ∀C∈(A,B)，∃ξ∈(a,b)，st.f(ξ)=C$

## 导数与微分

连续不一定可导（可微）；可导（可微）一定连续

不是明确可导的函数，不能直接写$f'(x)$，而要先写$\lim\limits_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}$存在

### 求导法则

#### 反函数

$y=f(x)$单调、可导、且$f'(x)\neq 0\Rightarrow (f^{-1}(y))'=\frac{1}{f'(x)}$，**最后记得化作$g(x)=f^{-1}(y)$**

#### 复合函数

$y=f(g(x))，g(x)在x处可导，f(u)在u=g(x)处可导\Rightarrow y'=\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}=f'(g(x))g'(x)$

#### 常用公式表及推导方法

| 原函数 | 导数 | 推导 |
|-|-|-|
| $x^u$ | $ux^{u-1}$ | - |
| $sinx$ | $cosx$ | - |
| $cosx$ | $-sinx$ | - |
| $tanx$ | $sec^2x$ | $tanx=\frac{sinx}{cosx}$ |
| $cotx$ | $-csc^2x$ | $cotx=\frac{cosx}{sinx}$ |
| $secx$ | $secxtanx$ | $secx=\frac{1}{cosx}$ |
| $cscx$ | $-cscxtanx$ | $cscx=\frac{1}{sinx}$ |
| $arcsinx$ | $\frac{1}{\sqrt{1-x^2}}$ | $(arcsinx)'=\frac{1}{(siny)'}=\frac{1}{cosy}$ |
| $arccosx$ | $\frac{-1}{\sqrt{1-x^2}}$ | 同上 |
| $arctanx$ | $\frac{1}{1+x^2}$ | 同上 |
| $arccotx$ | $\frac{-1}{1+x^2}$ | 同上 |
| $a^x$ | $a^xlna$ | - |
| $log_ax$ | $\frac{1}{xlna}$ | $(log_ax)'=\frac{1}{(a^y)'}=\frac{1}{a^ylna}$ |

### 常用高阶导数

| 原函数 | n阶导 |
|-|-|
| $sinx$ | $sin(x+\frac{n}{2}π)$ |
| $cosx$ | $cos(x+\frac{n}{2}π)$ |
| $f(ax+b)$ | $a^nf^{(n)}(ax+b)$ |
| $u(x)v(x)$ | $\sum_{i=0}^{n}C_{n}^{n-i}u^{(n-i)}(x)v^{(i)}(x)$ |


### 隐函数及参数方程的导数

- 隐函数
> 例子：$e^y+xy-e=0，两边对x求导：e^yy'+(y+xy')=0，分离变量：y'=-\frac{y}{x+e^y}$

- 参数方程
> $y'=\frac{dy}{dx}=\frac{\frac{dy}{dt}}{\frac{dx}{dt}}$

## 微分中值定理与导数的应用

### 微分中值定理

#### 拉格朗日中值定理

> $f(x)$满足：1：在$[a,b]$连续；2：在(a,b)可导
> 则：$∃ξ∈(a,b)，st.f(b)-f(a)=f'(ξ)(b-a)$

#### 柯西中值定理

> $f(x)和F(x)$满足：1：在$[a,b]$连续；2：在$(a,b)$可导；3：$∀x∈(a,b)，F'(x)\neq 0$
> 则：$∃ξ∈(a,b)，st.\frac{f(b)-f(a)}{F(b)-F(a)}=\frac{f'(ξ)}{F'(ξ)}$

### 洛必达法则

> 设：
> (1) $\lim\limits_{x\Rightarrow a}f(x)=\lim\limits_{x\Rightarrow a}F(x)=0$
> (2) $a$的某去心领域内，$f'(x)，F'(x)$都存在且$F'(x)\neq 0$
> (3) $\lim\limits_{x\Rightarrow a}\frac{f'(x)}{F'(x)}$存在或为无穷大
> 则：$\lim\limits_{x\Rightarrow a}\frac{f(x)}{F(x)}=\lim\limits_{x\Rightarrow a}\frac{f'(x)}{F'(x)}$

### 泰勒公式

$f(x)$在$x_0$处具有$n$阶导$\Rightarrow ∃x∈U(x_0)，st. f(x)=\sum_{i=0}^{n}\frac{f^{(i)}(x_0)}{i!}(x-x_0)^i+o((x-x_0)^n)$

取$x_0=0$得**n阶局部迈克劳林公式：**$f(x)=\sum_{i=0}^{n}\frac{f^{(i)}(0)}{i!}x^i+o(x^n)$ 

#### 常用公式

- $\frac{1}{1-x}=\sum_{i=0}^{n}x^i+o(x^n)=\frac{1-x^n}{1-x}+o(x^n)$
- 推出$\frac{a}{1-x}=\frac{a(1-x^n)}{1-x}+o(x^n)$，即**a为首项，x为公比的等比数列的前n项和**
- $e^x=\sum_{i=0}^{n}\frac{x^i}{i!}+o(x^n)$（借此可推出欧拉公式）
- $sinx=\sum_{i=0}^{n}\frac{(-1)^i}{(2i+1)!}+o(x^{2n+1})$
- $cosx=\sum_{i=0}^{n}\frac{(-1)^i}{(2i)!}+o(x^{2n})$

### 单调性与极值、凹凸区间与拐点

$f(x)在[a,b]连续，在(a,b)内具有一阶和二阶导，那么：$
- $若∀x∈(a,b)，f''(x)>0\Rightarrow f(x)在[a,b]是上凹的$
- $若∀x∈(a,b)，f''(x)<0\Rightarrow f(x)在[a,b]是上凸的$
- $若∃x_0∈(a,b)，f''(x)在x_0左右异号\Rightarrow (x_0,f(x_0))是拐点$

**注意：凹凸区间包含端点，拐点是坐标**

单调性的考虑点：驻点（二阶导为0）、不可导点；（**两者都是x=x_0**）

极值：先求单调性，列表求极值

## 不定积分

### 换元积分法

$\int u_v'v'_xdx=\int u_v'dv=u+C$

#### 常用公式

- $\int\frac{1}{x}=ln|x|+C$
- $\int tanxdx=\int\frac{1}{cosx}sinxdx=-\int\frac{1}{cosx}dcosx=-ln|cosx|+C$
- $\int cotxdx=\int\frac{1}{sinx}cosxdx=\int\frac{1}{sinx}dsinx=ln|sinx|+C$
- $\int secxdx=ln|secx+tanx|+C$
- $\int cscxdx=-ln|cscx+tanx|+C$

推导：
>$$\begin{split}
\int secxdx&=\int\frac{1}{cosx}dx\\\\
&=\int\frac{1}{cos^2x}cosxdx\\\\
&=\int\frac{1}{1-sin^2x}dsinx\\\\
&=\frac{1}{2}\int\frac{(1-sinx)+(1+sinx)}{(1-sinx)(1+sinx)}dsinx\\\\
&=\frac{1}{2}\int(\frac{1}{1+sinx}+\frac{1}{1-sinx})dsinx\\\\
&=\frac{1}{2}(ln|1+sinx|-ln|1-sinx|)+C\\\\
&=ln|\frac{\sqrt{1+sinx}}{\sqrt{1-sinx}}|+C\\\\
&=ln|\frac{\sqrt{(1+sinx)^2}}{\sqrt{(1+sinx)(1-sinx)}}|+C\\\\
&=ln|\frac{1+sinx}{cosx}|+C\\\\
&=ln|secx+tanx|+C
\end{split}$$

### 分部积分法

$\int udv=uv-\int vdu$：**对反幂三指，排前作u排后作v**

## 定积分

### 定积分的换元积分法和分部积分法

- 换元公式$\int_{a}^{b}(x)dx=\int_{α}^{β}f(φ(t))φ'(t)dt$
- 分部积分公式$\int_{a}^{b}udv=(u(b)v(b)-u(a)v(a))-\int_{a}^{b}vdu$

### 反常积分

$\frac{d\int_{α(x)}^{β(x)}f(t)dt}{dx}=β'(x)f(β(x))-α'(x)f(α(x))$

### 定积分几何

曲线弧长：
- 直角：$s=\int_{a}^{b}\sqrt{1+y'^2}dx$
- 参数：$s=\int_{a}^{b}\sqrt{φ'^2(t)+ψ'^2(t)}dt$
- 极坐标：$s=\int_{a}^{b}\sqrt{ρ^2(θ)+ρ'^2(θ)}dθ$

## 微分方程

### 一阶线性微分方程

$y'+P(x)y=Q(x)\Rightarrow y=(C+\int Q(x)e^{\int P(x)dx}dx)e^{-\int P(x)dx}$

### 可降阶的高阶微分方程

#### $y^{(n)}=f(x)$（只含$x$型）

> $$\begin{split}
&y''=x+sinx\\\\
令：&p=y'，y''=p'=x+sinx\\\\
两侧积分得：&p=\frac{x^2}{2}-cosx+C_1=y'\\\\
两侧积分得：&y=\frac{x^3}{6}-sinx+C_1x+C_2\\\\
\end{split}$$


#### $y''=f(x,y')$（含$x，y$型）

> $$\begin{split}
&y''=y'+x\\\\
令：&p=y'，y''=p'\\\\
原方程化为：&p'-p=x\\\\
一阶线性微分方程公式：&y'+P(x)y=Q(x)\Rightarrow y=(C+\int Q(x)e^{\int P(x)dx}dx)e^{-\int P(x)dx}\\\\
代入：&P(x)=-1，Q(x)=x，C=C_1\\\\
可得：&p=(C_1+\int xe^{\int (-1)dx}dx)e^{-\int (-1)dx}\\\\
&\ =(C_1+\int xe^{-x}dx)e^x\\\\
&\ =(C_1-\int xde^{-x})e^x\\\\
&\ =(C_1-(xe^{-x}-\int e^{-x}dx))e^x\\\\
&\ =(-xe^{-x}-e^{-x}+C_1)e^x\\\\
&\ =-x-1+C_1e^x\\\\
即：&y'=p=-x-1+C_1e^x\\\\
两侧积分得：&y=\int (C_1e^x-x-1)dx=C_1e^x-\frac{x^2}{2}-x+C^2
\end{split}$$

#### $y''=f(y,y')$（不含$x$型）

> $$\begin{split}
&yy''+2y'^2=0\\\\
令：&p=y'，y''=p'=\frac{dp}{dy}\frac{dy}{dx}=\frac{dp}{dy}p\\\\
原方程化为：&y\frac{dp}{dy}p+2p^2=0\\\\
分离变量得：&\frac{dp}{p}=-2\frac{dy}{y}\\\\
两侧积分得：&ln|p|=ln\frac{1}{y^2}+C_1\\\\
即：&p=\frac{C_0}{y^2}=y'=\frac{dy}{dx}\\\\
分离变量得：&y^2dy=C_0dx\\\\
两侧积分得：&y^3=C_1x+C_2
\end{split}$$

### 常系数齐次线性微分方程

#### 二阶

令$y''+py'+qy=0\Rightarrow 特征方程：r^2+pr+q=0，考虑其解的情况：$

| 特征方程的解 | 通解 |
|-|-|
| $r_1\neq r_2∈R$ | $y=C_1e^{r_1x}+C_2e^{r_2x}$ |
| $r_1=r_2=r∈R$ | $y=C_1e^{rx}+C_2xe^{rx}$ |
| $r_1=α+iβ，r_2=α-iβ$ | $y=C_1e^{αx}cosβx+C_2e^{αx}sinβx$ |

#### 高阶

第$i$重根就乘$x^i$

| 特征方程的解 | 通解 |
|-|-|
| $r_1=-1，r_2=r_3=i，r_4=r_5=-i$ | $y=C_1e^{-x}+C_2cosx+C_3sinx+C_4xcosx+C_5xsinx$ |
| $r_1=r_2=r_3=r$ | $y=C_1e^{rx}+C_2xe^{rx}+C_3x^2e^{rx}$ |

### 常系数非齐次线性微分方程

$y''+py'+qy=P_m(x)e^{λx}\Rightarrow 特解y^*=x^kQ_m(x)e^{λx}（k：λ是特征方程的k重根）$

一般做法：
- 先令$y''+py'+qy=0$，求其特征方程的根
- 右侧化为$P_m(x)e^{λx}$（有$sinx或cosx时λ$设为复数），得到$λ$是$k$重根
- 设出特解$y^*=x^kQ_m(x)e^{λx}（有sinx或cosx时为\overline{y}）$
- $y^*代入y$解出$Q_m(x)$的系数，得到特解（有$cosx时为\overline{y}$的实部；有$sinx时为\overline{y}$的虚部）
