---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Grid System

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

在时空数据挖掘的任务中，对地理空间进行划分是一个重要步骤，它帮助我们把复杂的地理区域分割成更小、更容易管理的部分。

在地理空间划分的过程中，有一些常用的方法：
- 按规则划分：把城市分成大小相同（相近）的方块（多边形）
- 根据相邻地区的特点，把相似的区块聚在一起
- 使用图（Graph）的方式把交通、社交等关系表示出来

一般在机器学习任务中的数据预处理部分，我们拿到轨迹点序列后，会用网格系统将轨迹点转换为网格的id。有几个原因：
1. 轨迹点的坐标是连续型的数字，粒度太细，放到空间里面会非常稀疏。两个经纬度在小数点后第7位之后的变化在空间中引起的位移不大，所以可以将他们划分到1个网格以此减轻数据稀疏带来的问题。
2. 在机器学习任务里面需要针对位置构造一些统计性的特征，只有抽象到网格凑齐很多个点之后才能进行统计，否则对一个经纬度进行统计，因为稀疏的性质，这种统计几乎没有意义。
3. 在深度学习任务中，大部分算法是通过`EmbeddingLayer`对位置进行表示的，而不是对经纬度点进行表示。

`TrajDL`提供了一个网格模块，可以用来将一个区域划分成多个网格，并且将经纬度转换为网格的id。

在`TrajDL`里面有两个核心概念与网格系统相关，`Boundary`和`GridSystem`，前者表示一个区域的边界，可以定位一个或多个`Point`是否在这个区域内；后者表示一个网格系统工具，可以将区域划分成网格，然后将`Point`转换为`Location`。

+++

## Boundary

我们先看一下`Boundary`。

```{code-cell} ipython3
from trajdl.trajdl_cpp import RectangleBoundary

boundary = RectangleBoundary(min_x=0, min_y=0, max_x=10, max_y=10)
print(boundary)
```

这个`Boundary`是一个类似矩形的boundary，通过指定最大最小的x和y值可以确定出一个矩形的范围。

```{note}

说是“类似矩形”是因为如果将`x`作为经度，`y`作为纬度，这个矩形在地球表面实际是一个小的球面，边界是弧形的，不是矩形。

```

有了这么一个`Boundary`，我们可以判断任意一个`(x, y)`是否在这个区域内。

```{code-cell} ipython3
boundary.in_boundary(0, 0)
```

```{code-cell} ipython3
boundary.in_boundary(0, 10)
```

```{code-cell} ipython3
import numpy as np

np.random.seed(42)

# 这里的np.ndarray的第一维是x，第二维是y
coords = np.random.uniform(low=-5, high=15, size=(100, 2))
boundary.in_boundary_np(coords)
```

```{code-cell} ipython3
boundary.to_tuple()
```

```{code-cell} ipython3
RectangleBoundary.from_tuple((0.0, 0.0, 10.0, 10.0))
```

## GridSystem

有了`Boundary`之后，`TrajDL`可以利用`Boundary`实现网格系统，下面是一个`SimpleGridSystem`的示例。

```{code-cell} ipython3
from trajdl.grid.base import SimpleGridSystem

boundary = RectangleBoundary(min_x=0, min_y=0, max_x=10, max_y=10)
grid = SimpleGridSystem(boundary=boundary, step_x=100.0, step_y=100.0)
print(grid)
```

这个`SimpleGridSystem`是将一个`RectangleBoundary`作为边界，然后以`step_x`和`step_y`分别作为两个方向上的步长对区域进行划分。

```{code-cell} ipython3
len(grid)
```

可以看到，上面的网格系统只有1个网格，因为步长设置的太大了。

```{code-cell} ipython3
grid = SimpleGridSystem(boundary=boundary, step_x=1.0, step_y=1.0)
print(len(grid))
print(grid.num_x_grids)
print(grid.num_y_grids)
```

我们可以看到，上面的新网格系统只有10行，10列了。我们现在尝试定位几个坐标看看。

```{code-cell} ipython3
grid.locate(x=0.1, y=0.1)
```

```{code-cell} ipython3
grid.locate(x=1, y=1)
```

```{code-cell} ipython3
:tags: [raises-exception]

grid.locate(x=10, y=10)
```

````{attention}

不论是`Boundary`还是`GridSystem`，都遵循左闭右开的规则，也就是一个**点**只有满足下面的要求，才算属于这个`Boundary`或者网格。

```{math}
min_x \leq x \lt max_x \\

min_y \leq y \lt max_y 
```

````

```{code-cell} ipython3
grid.locate(x=9.9, y=9.9)
```

```{code-cell} ipython3
# 通过网格系统也可以判断一个点是否在一个区域内
grid.in_boundary(9.9, 10)
```

当然，网格系统也支持向量化等操作

```{code-cell} ipython3
coords = np.random.uniform(low=-5, high=15, size=(10, 2))
print(coords)
grid.in_boundary_np(coords)
```

```{code-cell} ipython3
grid.locate_unsafe_np(coords)
```

```{attention}

使用向量化操作的时候，`locate_unsafe_np`就表示了当前操作不是安全的，所谓的不安全是指当一个坐标不在`Boundary`范围内的时候，这个方法得到的值可能是有问题的，比如上述的那些负值，这些点实际上并不在区域内，因此需要配合上面的`in_boundary_np`方法来使用。

```

+++

```{tip}

本文介绍了`GridSystem`，网格系统在`TrajDL`中主要提供轨迹数据转换成位置的功能，这在时空数据挖掘的任务中很常见。

```

```{code-cell} ipython3

```
