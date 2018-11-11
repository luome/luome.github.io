---
title: 几个改善写代码体验的python库
updated: 2018-11-11 16:00
---
在写神经神经网络的时候，发现了几个被人广泛应用、但又很不起眼的工具，可能这些工具与神经网络的关系并不是很大，但是，它们可以在一定程度上改善写代码的体验。
## TQDM
在训练一个神经网络的时候，一定会有epoch与batch的循环代码，在长循环中，往往让人等的不耐烦，python tqdm库是一个让你将训练过程的进度可视化的一个工具。
tqdm给我的使用体验十分的好，它可以在显示循环的次数和循环消耗的时间，特别是在计算量大，需要特别久的时候。
tqdm是一个十分简单的库，用户只需要封装任意的迭代器**tqdm(iterator)**即可完成进度条。
tqdm有三种十分通用的方法：


- **Iterable-based**：将任意可迭代对象通过tqdm()封装即可。


```python
text = ""
for char in tqdm(["a", "b", "c", "d"]):
    text = text  + char
# 类似用法很多，也十分方便。
for i in tqdm(range(100)):
    print(i)
```


- **Manual**: 手动控制可以通过一个with语句来完成:


```python
with tqdm(total=100) as pbar:
    for i in range(10)
        pbar.update(10)
```


- **Module**: tqdm在脚本或者命令行中的应用是十分方便的，只用在管道中插入tqdm(或者python -m tqdm),它会将stdin传递给stdout的同时将进度打印在stderr。


## ARGPARSE
argparse实际上是Python内置的一个用于命令行选项与参数解析的模块。之所以会频繁的使用到它是因为，神经网络中有很多**超参数**，以及控制训练、测试，利用argparse免去了每次都去代码中改超参数的麻烦（特别是在模型需要调参的时候），以及是运行训练还是测试代码。
argparse的使用主要有三个步骤：
- 创建ArgumentParser()对象
- 调用add_argument()方法添加参数
- 使用parse_args()解析添加参数


argparse在使用上是十分简单的，可以去[官网](https://docs.python.org/3/library/argparse.html)看看文档。


## IPDB
ipdb是一个打断点调试程序的库，在
```python
from ipdb import set_trace

# 在断点处使用
set_trace()
```
运行程序之后在set_trace()处会中断程序，然后进入ipython进行调试。
当然ipdb没有上面两个好用，因为ipdb可能一些不太喜欢用IDE的人会用，只是个人习惯的选择。而tqdm和argparse在实际中的使用更有用。

## END
今天周日刚好没事就随便写了写，我觉得我以后应该把文章的重心转移到一些深度学习算法和一些更深的思考上去。