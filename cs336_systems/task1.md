### 1.1 Profiling and Benchmarking (性能分析与基准测试)

在实施任何优化之前，首先对我们的程序进行性能分析（profile）以了解它在何处消耗资源（例如时间和内存）是很有帮助的。否则，我们可能会冒着优化模型中并未占用大量时间或内存的部分的风险，从而无法看到可衡量的端到端改进。

我们将实现三种性能评估路径：(a) 使用 Python 标准库进行简单的端到端基准测试，为我们的前向和反向传播计时；(b) 使用 NVIDIA Nsight Systems 工具分析计算性能，以了解时间是如何在 CPU 和 GPU 的各项操作中分配的；(c) 分析内存使用情况。

#### 1.1.1 设置 - 导入你的基础 Transformer 模型

让我们首先确保你能加载上一个作业中的模型。在上一个作业中，我们在一个 Python 包中设置了模型，以便稍后可以轻松导入。我们已在 `./cs336-basics` 文件夹中添加了模型的工作人员实现版本，并在 `pyproject.toml` 文件中指向了它。通过像往常一样调用 `uv run [command]`，`uv` 将自动定位这个本地的 `cs336-basics` 包。如果你想使用自己实现的模型，可以修改 `pyproject.toml` 文件以指向你自己的包。

你可以通过以下方式测试是否能导入你的模型：

```bash
~$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
Built cs336-systems @ file:///path/to/systems/dir
Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>>
```

作业 1 中的相关模块现在应该可用了（例如，对于 `model.py`，你可以使用 `import cs336_basics.model` 来导入）。

#### 1.1.2 模型尺寸 (Model Sizing)

在整个作业中，我们将对模型进行基准测试和性能分析，以更好地了解它们的性能。为了直观地了解事物在规模扩大时发生的变化，我们将使用并参考以下模型配置。对于所有模型，我们将使用 10,000 的词表大小和 4 的批次大小（batch size），并采用不同的上下文长度。本次作业（以及后续作业）将要求在表格中呈现大量结果。我们强烈建议你在代码中自动生成作业报告所需的表格，因为在 LaTeX 或 Markdown 中手动格式化表格可能会非常繁琐。请参考 `pandas.DataFrame.to_latex()` 和 `pandas.DataFrame.to_markdown()`，或者编写你自己的函数，从你首选的表格表示形式生成它们。

| 规模 (Size) | d_model | d_ff | num_layers | num_heads |
| :--- | :--- | :--- | :--- | :--- |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

*表 1：不同模型尺寸的规格*

#### 1.1.3 端到端基准测试 (End-to-End Benchmarking)

现在我们将实现一个简单的性能评估脚本。我们将测试模型的许多变体（更改精度、交换层等），因此**让你的脚本支持通过命令行参数启用这些变体将是非常值得的**，这会使后续的运行变得容易。我们还**强烈建议在 Slurm 上使用 `sbatch` 或 `submitit` 对基准测试超参数（如模型大小、上下文长度等）进行参数扫描（sweeps）**，以实现快速迭代。

首先，让我们通过对前向和反向传播计时来进行最简单的模型性能分析。由于我们只测量速度和内存，因此将使用随机权重和数据。

测量性能是微妙的——一些常见的陷阱可能导致我们测量不到想要的结果。对于 GPU 代码的基准测试，需要注意的一点是 CUDA 调用是*异步的（asynchronous）*。当你调用 CUDA 内核时（例如当你调用 `torch.matmul` 时），函数调用会将控制权交还给你的代码，而不会等待矩阵乘法完成。这样，CPU 可以继续运行，而 GPU 则在计算矩阵乘法。另一方面，这意味着天真地测量 `torch.matmul` 调用返回需要多长时间，并不能告诉我们 GPU 实际运行矩阵乘法需要多长时间。在 PyTorch 中，我们可以调用 `torch.cuda.synchronize()` 来等待所有 GPU 内核完成，从而让我们获得更准确的 CUDA 内核运行时间测量值。考虑到这一点，让我们编写基本的性能分析基础设施。

> **问题 (benchmarking_script)：4 分**
> 
> (a) 编写一个脚本来对模型的前向和反向传播执行基本的端到端基准测试。具体来说，你的脚本应支持以下内容：
> *   给定超参数（例如，层数），初始化一个模型。
> *   生成一个随机数据批次。
> *   运行 $w$ 步预热步骤（在开始测量时间之前），然后对 $n$ 个步骤的执行时间进行计时（仅前向传播，或同时进行前向和反向传播，具体取决于参数）。在计时方面，你可以使用 Python 的 `timeit` 模块（例如，使用 `timeit` 函数，或使用 `timeit.default_timer()`，它提供系统最高分辨率的时钟，因此是比 `time.time()` 更好的基准测试默认选择）。
> *   在每步之后调用 `torch.cuda.synchronize()`。
> 
> **交付物 (Deliverable)：** 一个将使用给定超参数初始化 `basics` Transformer 模型、创建随机数据批次，并对前向和反向传播进行计时的脚本。
> 
> (b) 为 §1.1.2 中描述的模型尺寸的前向和反向传播计时。使用 5 个预热步，并计算 10 个测量步的平均计时和标准差。一次前向传播需要多长时间？反向传播呢？你是否发现测量结果之间存在高变异性，还是标准差很小？
> 
> **交付物：** 一到两句话的回答，包含你的计时结果。
> 
> (c) 基准测试的一个陷阱是不执行预热步骤。在不使用预热步的情况下重复你的分析。这对你的结果有什么影响？你认为这是为什么？此外，尝试运行带有 1 到 2 个预热步的脚本。为什么结果可能仍然不同？
> 
> **交付物：** 两到三句话的回答。

#### 1.1.4 Nsight Systems 性能分析器

端到端基准测试并不能告诉我们模型在前向和反向传播期间把时间和内存花在了哪里，因此无法暴露出具体的优化机会。要知道我们的程序在每个组件（例如，函数）中花费了多少时间，我们可以使用*性能分析器（profiler）*。执行分析器通过在函数开始和结束运行时插入保护指令来检测代码，从而可以提供函数级别的详细执行统计信息（如调用次数、平均耗时、在该函数上花费的累计时间等）。

标准的 Python 性能分析器（例如 `CProfile`）无法对 CUDA 内核进行分析，因为这些内核是在 GPU 上异步执行的。幸运的是，NVIDIA 提供了一个我们可以通过命令行界面 `nsys` 使用的分析器，我们已经为你安装好了。在作业的这一部分，你将使用 `nsys` 来分析 Transformer 模型的运行时间。使用 `nsys` 非常简单：我们只需在上一节的 Python 脚本命令前加上 `nsys profile` 即可运行。例如，你可以分析 `benchmark.py` 脚本并将输出写入 `result.nsys.rep` 文件，命令如下：

```bash
~$ uv run nsys profile -o result python benchmark.py
```

然后，你可以使用 NVIDIA Nsight Systems 桌面应用程序在本地机器上查看配置文件。在配置文件的 `CUDA API` 行中选择特定的 CUDA API 调用（在 CPU 上），将会在 `CUDA HW` 行中高亮显示所有对应的内核执行（在 GPU 上）。

我们鼓励你尝试 `nsys profile` 的各种命令行选项，以了解它的功能。值得注意的是，你可以使用 `--python-backtrace=cuda` 获取每个 CUDA API 调用的 Python 回溯栈，不过这可能会带来额外开销。你还可以使用 NVTX 范围（ranges）来注释你的代码，这将在配置文件的 `NVTX` 行中显示为块（blocks），捕获所有 CUDA API 调用及相关的内核执行。特别需要指出的是，**你应该使用 NVTX 范围在基准测试脚本中忽略预热步骤**（通过在配置文件中的 `NVTX` 行应用过滤器）。你还可以隔离出哪些内核负责模型的前向和反向传播，甚至可以通过如下方式注释你的实现，来隔离哪些内核负责自注意力层的不同部分：

```python
...
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    ... # Q, K, V, mask
)
    ...
    with nvtx.range("computing attention scores"):
        ... # 计算 Q 和 K 之间的注意力分数

    with nvtx.range("computing softmax"):
        ... # 计算注意力分数的 softmax

    with nvtx.range("final matmul"):
        ... # 计算输出投影
    
    return ...
```

你可以在基准测试脚本中将原始实现替换为带有注释的版本，方法如下：

```python
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
```

最后，你可以使用带有 `nsys` 的 `--pytorch` 命令行选项，使用 NVTX 范围自动注释对 PyTorch C++ API 的调用。

> **问题 (nsys_profile)：5 分**
> 
> 使用 `nsys` 分析表 1 中描述的每种模型尺寸的前向传播、反向传播和优化器步，分别使用 128、256、512 和 1024 的上下文长度（对于较大的模型，在某些上下文长度下你可能会遇到内存不足的错误，如果发生这种情况，只需在报告中注明即可）。
> 
> (a) 你的前向传播总共花费了多少时间？这与我们之前使用 Python 标准库测量的时间一致吗？
> **交付物：** 一到两句话的回答。
> 
> (b) 在前向传播期间，哪个 CUDA 内核占用的累计 GPU 时间最多？在模型的一次前向传播中，这个内核被调用了多少次？这与你同时进行前向和反向传播时占用运行时间最多的内核是同一个吗？（提示：查看“Stats Systems View”下的“CUDA GPU Kernel Summary”，并使用 NVTX 范围进行过滤，以确定模型的哪些部分负责哪些内核。）
> **交付物：** 一到两句话的回答。
> 
> (c) 尽管绝大多数的浮点运算（FLOPs）发生在矩阵乘法中，但你会注意到其他几个内核仍然占据了总运行时间的相当大一部分。除了矩阵乘法之外，你还看到哪些内核在前向传播中占据了不可忽视的 CUDA 运行时间？
> **交付物：** 一到两句话的回答。
> 
> (d) 使用你的 AdamW 实现分析运行一个完整训练步的情况（即，前向传播、计算损失并运行反向传播，最后运行优化器步，就像你在训练期间做的那样）。与仅进行推理（仅前向传播）相比，在矩阵乘法上花费的时间比例发生了怎样的变化？其他内核呢？
> **交付物：** 一到两句话的回答。
> 
> (e) 比较在前向传播期间，自注意力层中 softmax 操作与矩阵乘法操作的运行时间。运行时间上的差异与浮点运算（FLOPs）上的差异相比如何？
> **交付物：** 一到两句话的回答。

#### 1.1.5 混合精度 (Mixed Precision)

到作业的这一步为止，我们一直在使用 FP32 精度运行——所有模型参数和激活值都具有 `torch.float32` 数据类型。然而，现代 NVIDIA GPU 包含专门的 GPU 核心（Tensor Cores，张量核心），用于加速较低精度下的矩阵乘法。例如，NVIDIA A100 的规格表指出其在 FP32 下的最大吞吐量为 19.5 TFLOP/s，而在 FP16（半精度浮点数）或 BF16（脑浮点数）下的最大吞吐量显著更高，达到 312 TFLOP/s。因此，使用较低精度的数据类型应该有助于我们加速训练和推理。

然而，天真地将我们的模型转换为较低精度的格式可能会导致模型准确率下降。例如，在实践中，许多梯度值通常太小，无法用 FP16 表示，因此在天真地使用 FP16 精度进行训练时会清零。为了应对这个问题，通常在使用 FP16 训练时使用*损失缩放（loss scaling）*——只需将损失乘以一个缩放因子，从而增加梯度幅度，使其不会下溢为零。此外，FP16 的动态范围低于 FP32，这可能导致溢出并表现为 NaN 损失。完整的 bfloat16 (BF16) 训练通常更稳定（因为 BF16 具有与 FP32 相同的动态范围），但与 FP32 相比仍可能影响最终的模型性能。

为了利用低精度数据类型带来的加速，通常的做法是使用*混合精度训练（mixed-precision training）*。在 PyTorch 中，这通过 `torch.autocast` 上下文管理器来实现。在这种情况下，某些操作（例如，矩阵乘法）在较低精度的数据类型中执行，而其他需要 FP32 完整动态范围的操作（例如，累加和归约）则保持不变。例如，以下代码在前向传播期间将自动识别哪些操作可以在较低精度下执行，并将这些操作转换为指定的数据类型：

```python
model : torch.nn.Module = ... # 例如你的 Transformer 模型
dtype : torch.dtype = ... # 例如 torch.float16
x : torch.Tensor = ... # 输入数据

with torch.autocast(device="cuda",dtype=dtype):
    y = model(x)
```

如上所述，即使被累加的张量本身已被降级转换，通常也最好将累加操作保持在更高的精度中。接下来的练习将帮助你建立关于为什么会这样的直观理解。

> **问题 (mixed_precision_accumulation)：1 分**
> 
> 运行以下代码并评论结果（的准确性）。
> 
> ```python
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float32)
> print(s)
> 
> s = torch.tensor(0,dtype=torch.float16)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float16)
> print(s)
> 
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float16)
> print(s)
> 
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     x = torch.tensor(0.01,dtype=torch.float16)
>     s += x.type(torch.float32)
> print(s)
> ```
> 
> **交付物：** 两到三句话的回答。

现在我们将混合精度首先应用于一个玩具模型以建立直觉，然后再应用于我们的基准测试脚本。

> **问题 (benchmarking_mixed_precision)：2 分**
> 
> (a) 考虑以下模型：
> 
> ```python
> class ToyModel(nn.Module):
>     def __init__(self, in_features: int, out_features: int):
>         super().__init__()
>         self.fc1 = nn.Linear(in_features, 10, bias=False)
>         self.ln = nn.LayerNorm(10)
>         self.fc2 = nn.Linear(10, out_features, bias=False)
>         self.relu = nn.ReLU()
> 
>     def forward(self, x):
>         x = self.relu(self.fc1(x))
>         x = self.ln(x)
>         x = self.fc2(x)
>         return x
> ```
> 
> 假设我们在 GPU 上训练该模型，且模型参数最初是 FP32 格式。我们希望使用 FP16 的 `autocast` 混合精度。以下各项的数据类型分别是什么：
> *   `autocast` 上下文中的模型参数，
> *   第一个前馈层的输出 (`ToyModel.fc1`)，
> *   层归一化的输出 (`ToyModel.ln`)，
> *   模型预测的 logits，
> *   损失 (loss)，
> *   以及模型的梯度？
> 
> **交付物：** 上述每个组件的数据类型。
> 
> (b) 你应该已经注意到，FP16 混合精度 `autocasting` 对层归一化层（layer normalization）的处理方式与前馈层不同。层归一化的哪些部分对混合精度敏感？如果我们使用 BF16 代替 FP16，我们是否还需要以不同的方式处理层归一化？为什么或为什么不？
> 
> **交付物：** 两到三句话的回答。
> 
> (c) 修改你的基准测试脚本，使其可以选择性地使用 BF16 混合精度运行模型。在有和没有混合精度的情况下，为 §1.1.2 中描述的每个语言模型尺寸的前向和反向传播计时。比较使用全精度与混合精度的结果，并评论随着模型大小变化出现的任何趋势。你可能会发现 `nullcontext`（无操作上下文管理器）很有用。
> 
> **交付物：** 两到三句话的回答，包含你的计时结果和评论。

#### 1.1.6 内存分析 (Profiling Memory)

到目前为止，我们一直在关注计算性能。现在我们将注意力转向*内存*，这是语言模型训练和推理中的另一个主要资源。PyTorch 也配备了一个强大的内存分析器，可以跟踪一段时间内的内存分配。

要使用内存分析器，你可以按如下方式修改基准测试脚本：

```python
... # 基准测试脚本中的预热阶段

# 开始记录内存历史。
torch.cuda.memory._record_memory_history(max_entries=1000000)

... # 你想要在基准测试脚本中进行性能分析的内容

# 保存一个 pickle 文件，以便被 PyTorch 的在线工具加载。
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# 停止记录历史。
torch.cuda.memory._record_memory_history(enabled=None)
```

这将输出一个名为 `memory_snapshot.pickle` 的文件，你可以将其加载到以下在线工具中：`https://pytorch.org/memory_viz`。此工具可让你查看总体内存使用时间线，以及进行过的每次独立内存分配，包括其大小以及指向导致该分配的代码的堆栈跟踪（stack trace）。要使用此工具，你应该在 Web 浏览器中打开上述链接，然后将 Pickle 文件拖放到页面上。

现在，你将使用 PyTorch 分析器来分析模型的内存使用情况。

> **问题 (memory_profiling)：4 分**
> 
> 使用表 1 中的 2.7B 模型，在上下文长度为 128、256 和 512 的情况下，分析你的前向传播、反向传播和优化器步。
> 
> (a) 在你的性能分析脚本中添加一个选项，以通过内存分析器运行模型。重用你之前的一些基础设施（例如，激活混合精度、加载特定模型尺寸等）可能会有所帮助。然后，运行你的脚本以获取 2.7B 模型在仅进行推理（仅前向传播）或执行完整训练步骤时的内存配置文件。你的内存时间线看起来像什么？你能根据看到的峰值判断出正在运行哪个阶段吗？
> 
> **交付物：** 两张从 `memory_viz` 工具获取的 2.7B 模型“活动内存时间线（Active memory timeline）”的图像：一张用于前向传播，另一张用于运行完整的训练步骤（前向和反向传播，然后是优化器步），以及两到三句话的回答。
> 
> (b) 在进行前向传播时，每个上下文长度的峰值内存使用量是多少？进行完整训练步骤时呢？
> 
> **交付物：** 一个表格，每个上下文长度对应两个数字。
> 
> (c) 找出 2.7B 模型在使用混合精度时，前向传播和完整优化器步的峰值内存使用量。混合精度是否显著影响内存使用？
> 
> **交付物：** 两到三句话的回答。
> 
> (d) 考虑 2.7B 模型。在我们的参考超参数下，单精度下 Transformer 残差流中的一个激活值张量的大小是多少？请以 MB 为单位给出此大小（即字节数除以 $1024^2$）。
> 
> **交付物：** 一到两句话的回答及你的推导过程。
> 
> (e) 现在仔细观察 `pytorch.org/memory_viz` 中 2.7B 模型执行前向传播时的内存快照的“活动内存时间线”。当你降低“Detail（细节）”级别时，该工具会将最小的分配隐藏到相应的级别（例如，将“Detail”设置为 10% 将仅显示最大的 10% 的分配）。显示的最大的内存分配大小是多少？查看堆栈跟踪，你能判断出这些分配来自哪里吗？
> 
> **交付物：** 一到两句话的回答。