# MobileNetV3

## 目录


- [1. 简介]()
- [2. 复现流程]()
    - [2.1 reprod logger简介]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 生成伪数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 数据加载对齐]()
    - [4.2 模型前向对齐]()
    - [4.3 评估指标对齐]()
    - [4.4 反向梯度初次对齐]()
    - [4.5 训练对齐]()


## 1. 简介

* 本部分内容包含基于 [MobileNetV3](https://arxiv.org/abs/1905.02244) 的复现对齐过程。

## 2. 复现流程
在论文复现中我们可以根据网络训练的流程，将对齐流程划分为数据加载对齐、模型前向对齐、评估指标对齐、反向梯度对齐和训练对齐。其中不同对齐部分我们会在下方详细介绍。
在对齐验证的流程中，我们依靠 reprod logger 日志工具查看 paddle 和官方同样输入下的输出是否相同，这样的查看方式具有标准统一，比较过程方便等优势。

### 2.1 reprod logger 简介
Reprod logger 是一个用于 numpy 数据记录和对比工具，通过传入需要对比的两个 numpy 数组就可以在指定的规则下得到数据之差是否满足期望的结论。其主要接口的说明可以看它的 [github 主页](https://github.com/WenmuZhou/reprod_log)

## 3. 准备数据和环境
在进行我们的对齐验证之前，我们需要准备运行环境、用于输入的伪数据、paddle 模型参数和官方模型权重参数。

### 3.1 准备环境
* 克隆本项目

```bash
https://github.com/PaddlePaddle/models.git
cd model/tutorials/mobilenetv3_prod/Step1
```

* 安装paddlepaddle [Paddle安装指南](https://www.paddlepaddle.org.cn/)

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 生成伪数据
为了保证模型对齐不会受到数据的影响，我们生成一组数据作为两个模型的输入。
伪数据可以通过如下代码生成，我们在本地目录下也提供了好的伪数据（./data/fake_*.npy）。

```python
def gen_fake_data():
    fake_data = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)
```

### 3.3 准备模型
为了保证模型前向对齐不受到参数不一致的影响，我们使用相同的权重参数对模型进行初始化。

生成相同权重参数分为以下 2 步：
1. 随机初始化官方模型参数并保存成 mobilenet_v3_small-047dcff4.pth
2. 将 model.pth 通过 ./torch2paddle.py 生成mv3_small_paddle.pdparams

转换模型时，torch 和 paddle 存在参数需要转换的部分，主要是bn层、全连接层、num_batches_tracked等，转换脚本(./torch2paddle.py)中已经标出。

## 4. 开始使用
准备好数据之后，我们通过下面的拆解步骤进行复现对齐。

### 4.1 数据加载对齐

【**运行文件**】

【**获得结果**】
```python
python test_forward.py
```
验证结果满足预期，验证通过：

```bash
[2021/10/14 18:09:15] root INFO: logits:
[2021/10/14 18:09:15] root INFO:    mean diff: check passed: True, value: 5.600350050372072e-07
[2021/10/14 18:09:15] root INFO: diff check passed
```

### 4.2 模型前向对齐
前向对齐的验证过程如下图所示:

<div align="center">
    <img src="./images/step1_graph.png" width=500">
</div>

由图可以看到，其验证标准在于输入相同伪数据、且两个模型参数相同的情况下，paddle 模型产出的 logit 是否和官方模型一致。


【**运行文件**】

【**获得结果**】

### 4.3 评估指标对齐

【**运行文件**】

【**获得结果**】

### 4.4 反向梯度对齐
【**运行文件**】

【**获得结果**】

### 4.5 训练对齐

【**运行文件**】

【**获得结果**】
