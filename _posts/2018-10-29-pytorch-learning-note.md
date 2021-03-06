---
title: PyTorch学习日志
updated: 2018-9-29 21:20
---
以前学习吴恩达的deeplearning.ai代码用的是tensorflow与keras，就学习了一下tensorflow，tensorflow的代码很繁琐，在语言设计上也不太贴合python的风格。PyTorch在最近炙手可热，于是我在上周按照官网的tutorial写了一下，跑了一个demo试了一下。在周末的时候花了两天时间看了一下PyTorch的api文档，今天终于初略地把文档看完了，下面是一些初学的感受。
## AUTOGRAD MECHANICS 
AUTOGRAD 是用BP算法自动计算梯度的方式，官网说没有必要完全掌握autograd,但是熟悉autograd能够帮助编写更高效、整洁、易于debug的程序。

每一个张量都有一个flag: requires_grad, 将其设置为False是能够细分排除子图。这个在冻结部分模型时，例如如果想要调整别人预训练好的网络，只用设置冻结部分的requires_grad为False即可。想了一下，这个在transfer learning应该能够有用处。

```python
x = torch.zeros(1, requires_grad=True)
with torch.no_grad():
    y = x * 2
print(y.requires_grad) # False
```

## MODULES

torch的文档比tensorflow简洁很多。*torch*模块中定义了一些与随机采样，序列化，并行，基础的数学运算相关的接口，这些接口似乎和numpy中的很像。顺便一提的是很多pytorch操作都支持numpy的**boardcast**语义。

*torch.nn*与*torch.nn.functional*也具有相似的接口。在定义一个神经网络的时候，通常通过继承nn.Module来实现，在继承时需要实现forward()方法。在深度学习中的很多层，例如卷积，池化，padding，序列模型等等都能够在torch.nn中找到相应的类。而*torch.nn.functional*与前者有一样的接口来实现神经网络的各种算法。刚开始，我还不知道它这样设计的目的，后来去看了一下源码，发现torch.nn中forword()里就是调用nn.functional中的函数实现。但是当你需要维护其中的参数状态时，应该用torch.nn，如conv2d之类的，而没有就可以用torch.nn.functional中的来定义，像relu,pooling之类的。

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

*torch.optim*模块中实现了一些优化器，像adam, SGD这些优化参数的算法都可以通过这个模块中的类来实现，而且这个模块中有很多文献中常用的算法实现。使用方式：
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for input, target in dataset:
    optimizer.zero_grad() # set grads to zero
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward() # bp
    optimizer.step() # update the parameters
```

*torch.distributions*这个模块是用来实现各种概率分布的模块，也是很有用的一个模块，比如ExponentialFamily, Binomial之类的概率分布，在统计学上很有用，还在哪看到有些数学统计工具的库就是基于这个模块写的。毕竟人工智能就是某种意义上的统计学。

还有很多其他的模块，用来操作张量，使用cuda计算，分布式，ONNX相关和其他工具，好像还并入了caffe2的代码。

## REMEMBER TO SAVE YOUR MODEL

在训练完模型时，记得将训练好的模型保存为一个文件供测试使用，或者接着上一次没有训练完成的继续训练。包括在你准备训练的时候，一定要检查保存模型的代码有没有写，否则白白消耗了计算资源。
```python
# 保存参数
torch.save(the_model.state_dict(), PATH)
# 恢复
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))

# 一种保存整个模型的方法
torch.save(the_model, PATH)
the_model = torch.load(PATH)

# 在实际使用中，比如以下一段代码
# 在训练中保存需要保存的参数到某个路径
if(iteration % save_every == 0):
    directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration':iteration,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
# 加载参数时
if loadFilename:
    checkpoint = torch.load(loadFilename)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__init__ = checkpoint['voc_dict']
```
## END OF THE BEGINNING

PyTorch现在发展到快1.0版本，官方发布了[road to 1.0](https://pytorch.org/blog/the-road-to-1_0/)，上面写了PyTorch的1.0发展展望。用官方的原话说有*stability of interface*,也开始有*production support*，已经很成熟了，不过我也不是很清楚，毕竟看不懂python以下的底层代码。PyTorch的设计上有很多numpy-like接口，而且与tensorflow相比最大的不同是动态图，不像tensorflow是静态图框架，tensorflow的session设计让我觉得是最恼火的。最近也只能在网上看看代码，然后写一些和跑一些简单的模型。不管是啥，希望自己能够尽快去掌握这个框架然后能复现文献中的算法吧。
