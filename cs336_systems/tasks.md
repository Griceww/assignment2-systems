### 1.2 使用 FlashAttention-2 优化注意力机制 (Optimizing Attention with FlashAttention-2)

#### 1.2.1 PyTorch 注意力机制基准测试

您的性能分析结果可能表明，您的注意力层在显存和计算方面都有优化的空间。在较高层面上，注意力操作由一次矩阵乘法、接着是 softmax，然后再进行一次矩阵乘法组成：

$$
Attention(Q, K, V) = \text{softmax}\left(\text{mask}\left(\frac{Q^\top K}{\sqrt{d_k}}\right)\right)V \quad (1)

$$

这种朴素的注意力实现需要为每个 batch（批次）/ head（注意力头）元素保存形状为 `seq_len × seq_len` 的注意力分数矩阵，当序列长度很长时，该矩阵会变得非常大，导致在处理长输入或输出的任务时出现显存溢出（out-of-memory，简称OOM）错误。我们将根据 FlashAttention-2 论文实现一个注意力 Kernel，它通过分块（tiles）计算注意力，避免显式地物化（materializing） `seq_len × seq_len` 的注意力分数矩阵，从而能够扩展到更长的序列长度。

> **问题 (pytorch_attention)：2 分**
>
> (a) 在不同规模下对您的注意力机制实现进行基准测试。编写一个脚本来执行以下操作：
>
> - (a) 将批大小（batch size）固定为 8，并且不使用多头注意力（即移除 head 维度）。
> - (a) 遍历 head 嵌入维度 $d_{\text{model}}$ `[16, 32, 64, 128]` 和序列长度 `[256, 1024, 4096, 8192, 16384]` 的笛卡尔积（cartesian product）。
> - (c) 为相应的维度大小创建随机输入 $Q, K, V$。
> - (d) 使用这些输入，对 100 次注意力前向传播进行计时。
> - (e) 测量反向传播开始前使用了多少显存，并对 100 次反向传播进行计时。
> - (f) 确保进行预热（warm up），并在每次前向/反向传播后调用 `torch.cuda.synchronize()`。
>
> 报告您在这些配置下获得的计时时间（或记录是否出现了显存溢出错误）。在多大的规模下您遇到了显存溢出错误？在出现显存溢出错误的最基础（最小）配置中，请列出注意力的显存使用情况的计算过程（您可以使用作业 1 中关于 Transformer 显存使用的公式）。反向传播节省的显存如何随序列长度变化？您会如何做来消除这一显存开销？
>
> **交付物**：一张包含您的计时结果的表格，显存使用情况的计算过程，以及 1-2 段的文字回答。

---

### 1.3 JIT 编译的注意力机制基准测试 (Benchmarking JIT-Compiled Attention)

自 2.0 版本起，PyTorch 附带了一个强大的即时编译器（just-in-time compiler），它会自动尝试对 PyTorch 函数应用多种优化：关于简介请参阅 [PyTorch 教程](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)。特别是，它会通过动态分析您的计算图，尝试自动生成融合的 Triton Kernel（fused Triton kernels）。使用 PyTorch 编译器的接口非常简单。例如，如果我们想将其应用到模型的某一个单一层上，我们可以这样写：

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer)
```

现在，`compiled_layer` 在功能上的行为和 `layer` 完全相同（例如，均具备前向和反向传播功能）。我们也可以使用 `torch.compile(model)` 编译整个 PyTorch 模型，甚至可以编译调用 PyTorch 操作的普通 Python 函数。

> **问题 (torch_compile)：2 分**
>
> (a) 扩展您的注意力基准测试脚本，加入 PyTorch 注意力实现的编译版本，并在与上述 `pytorch_attention` 问题相同的配置下，比较其与未编译版本的性能。
>
> **交付物**：一张比较您编译的注意力模块与上述 `pytorch_attention` 问题中未编译版本的前向和反向传播计时的表格。
>
> (b) 现在，在您的端到端基准测试脚本中编译整个 Transformer 模型。前向传播的性能发生了什么变化？前向、反向传播以及优化器步骤的组合性能又如何？
>
> **交付物**：一张比较您原始（vanilla）和编译后 Transformer 模型的表格。

鉴于我们观察到的相对于序列长度的扩展行为，我们需要进行显著的改进才能处理大序列。即使使用了 `torch.compile`，当前的实现在长序列下仍然受到极其糟糕的内存访问模式的困扰。为此，我们将用 Triton 编写一个 FlashAttention-2 的实现，在其中我们将对内存的访问方式以及计算时机拥有更多的控制权。

#### 1.3.1 示例 - 加权求和 (Example - Weighted Sum)

为了介绍您需要了解的 Triton 知识以及它如何与 PyTorch 交互，我们将通过一个“加权求和”操作的示例 Kernel 进行讲解。关于快速上手 Triton 的更多资源，请参考 Triton 的官方教程。但请注意，这些教程并未采用方便的“块指针（block pointer）”新抽象，我们将在下文中逐步介绍。

给定一个输入矩阵 $X$，我们将使其各项乘以一个列方向的权重向量 $w$，并对每行求和，从而得到 $X$ 和 $w$ 的矩阵-向量乘积。我们将首先完成此操作的前向传播，然后为其编写反向传播的 Triton Kernel。

**前向传播**
我们 Kernel 的前向传播仅仅是以下基于广播机制的内积（broadcasted inner product）：

```python
def weighted_sum(x, weight):
    # 此处假设 x 具有 n 维形状 [..., D]，而 weight 具有一维形状 [D]
    return (weight * x).sum(axis=-1)
```

在编写 Triton Kernel 时，我们会让每个程序实例（可能并行运行）计算 $x$ 某几行组成的一个分块（tile）的加权和，并将相应的标量输出写入输出张量中。在 Triton 中，一个程序实例（program instance）是指运行同一程序的一组线程块（thread blocks），这些线程块可以在 GPU 上并行执行。我们不将张量作为参数传入，而是传入指向它们第一个元素的指针，以及每个张量的步幅（strides），这些步幅告诉我们如何沿轴进行移动。

我们可以使用这些步幅加载与当前运行实例正在求和的 $x$ 行块相对应的张量，通过程序 ID（program ID）来划分工作（即，实例 $i$ 将处理 $x$ 的第 $i$ 个行块）。在这个简单的例子中，Triton 和 PyTorch 之间前向传播的主要区别在于需要进行指针算术（pointer arithmetic）和显式的加载/存储。我们将使用 `tl.make_block_ptr` 提供的块指针（block pointer）抽象来极大地简化指针算术，尽管这意味着我们需要编写一些设置代码来准备这些块指针。

关于分块（tiling）以及如何推进块指针的示意图，请参阅图 1（*原文档插图*）。上面的加权求和函数在 Triton 中如下所示：

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,       # 输入指针
    output_ptr,              # 输出指针
    x_stride_row, x_stride_dim, # 步幅（Strides）告诉我们如何在张量的每个轴上移动一个元素
    weight_stride_dim,       # 可能为 1
    output_stride_row,       # 可能为 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # 分块的形状必须在编译时已知
):
    # 每个实例将计算 x 的几行组成的一个分块的加权和。
    # `tl.program_id` 为我们提供了一种检查当前运行在哪个线程块中的方法
    row_tile_idx = tl.program_id(0)

    # 块指针为我们提供了一种从 N 维内存区域中进行选择，
    # 并移动我们的选择区域的方法。
    # 块指针必须知道：
    # - 指向张量第一个元素的指针
    # - 张量的整体形状，以处理越界访问
    # - 正确使用内存布局所需的每个维度的步幅
    # - 起始块的 N 维坐标，即 "offsets"（偏移量）
    # - 一次加载/存储使用的块形状
    # - 内存中维度的顺序，从主到次 (major to minor)
    #   axes (= np.argsort(strides)) 用于优化，在 H100 上特别有用

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 初始化一个缓冲区用于写入
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前的块指针
        # 由于 ROWS_TILE_SIZE 可能无法整除 ROWS，且 D_TILE_SIZE 可能无法整除 D，
        # 我们需要对两个维度进行边界检查 (boundary checks)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        # 计算该行的加权和。
        output += tl.sum(row * weight[None, :], axis=1)

        # 将指针移动到下一个分块。
        # 这些是 (rows, columns) 的坐标增量
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # 在最后一个维度上移动 D_TILE_SIZE
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # 移动 D_TILE_SIZE

    # 将输出写入输出块指针（每行一个标量）。
    # 由于 ROWS_TILE_SIZE 可能无法整除 ROWS，我们需要边界检查
    tl.store(output_block_ptr, output, boundary_check=(0,))
```

现在，让我们将这个 Kernel 封装到一个 PyTorch 的 Autograd 函数中，使其能够与 PyTorch 进行互操作（即，接受 Tensor 作为输入，输出一个 Tensor，并在反向传播期间与 autograd 引擎协作）：

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 缓存 x 和 weight 以便在反向传播中使用，当我们
        # 仅接收到关于输出张量的梯度时，需要它们
        # 来计算关于 x 和 weight 的梯度。
        D, output_dims = x.shape[-1], x.shape[:-1]

        # 将输入张量重塑（Reshape）为 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 粗略地在嵌入维度上循环 16 次
        ctx.ROWS_TILE_SIZE = 16  # 每个线程每次处理 16 个 batch 元素
        ctx.input_shape = input_shape

        # 需要初始化空的张量来存放结果。请注意，这些元素未必为 0！
        y = torch.empty(output_dims, device=x.device)

        # 在我们的 1D 网格中以 n 个实例启动我们的 kernel。
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])
```

请注意，当我们使用 `weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)]` 调用 Triton Kernel 时，我们通过传递元组 `(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)` 定义了一个由线程块组成的所谓“启动网格（launch grid）”。然后，我们可以在 Kernel 中使用 `tl.program_id(0)` 来访问线程块的索引。

**反向传播**
既然我们正在定义自己的 Kernel，我们还需要编写自己的反向传播函数。

在前向传播中，我们给定了层的输入，并需要计算其输出。在反向传播中，请回想一下，我们将得到目标函数相对于我们输出的梯度，并需要计算相对于我们每个输入的梯度。在我们的例子中，操作的输入是一个矩阵 $x \in \mathbb{R}^{n \times h}$ 和一个权重向量 $w \in \mathbb{R}^h$。为了简便，我们将该操作称为 $f(x, w)$，其值域为 $\mathbb{R}^n$。然后，假设我们已知 $\nabla_{f(x,w)}\mathcal{L}$，即损失 $\mathcal{L}$ 相对于层输出的梯度，我们可以应用多元链式法则获得相对于 $x$ 和 $w$ 的梯度的以下表达式：

$$
(\nabla_x\mathcal{L})*{ij} = \sum*{k=1}^n \frac{\partial f(x, w)*k}{\partial x*{ij}} (\nabla_{f(x,w)}\mathcal{L})*k = w_j \cdot (\nabla*{f(x,w)}\mathcal{L})_i \quad (2)

$$

$$
(\nabla_w\mathcal{L})*j = \sum*{i=1}^n \frac{\partial f(x, w)*i}{\partial w_j} (\nabla*{f(x,w)}\mathcal{L})*i = \sum*{i=1}^n x_{ij} \cdot (\nabla_{f(x,w)}\mathcal{L})_i \quad (3)

$$

这给出了计算反向传播的简单公式。为了获得关于 $x$ 的反向步骤，我们应用公式 2，并取 $w$ 和 $\nabla_{f(x,w)}\mathcal{L}$ 的外积。为了计算关于 $w$ 的反向步骤（即 $(\nabla_w\mathcal{L})_j$），我们必须将输入梯度与相应的输出行相乘。

我们的反向传播 Kernel 将从定义所有块指针开始，然后计算 $\nabla_x\mathcal{L}$：

```python
@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,                 # 输入
    grad_output_ptr,                   # 梯度输入
    grad_x_ptr, partial_grad_weight_ptr, # 梯度输出
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs（输入）
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

        # 计算 grad_x 的外积
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 为 grad_weight 的结果尽可能归约多行
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # 在维度 0 上绝不会越界

        # 将指针沿 D 维度移动到下一个分块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
```

计算梯度 $\nabla_x$ 很简单，我们将结果写入输出张量相应的分块中。然而，计算 $\nabla_w$ 稍微更具挑战性。每个 Kernel 实例负责 $x$ 的一个行块，但现在我们需要对 $x$ 的多行进行求和。我们不会直接在反向传播 Kernel 中进行这个全局求和，而是假设 `partial_grad_weight_ptr` 包含一个 `n_row_tiles × H` 的矩阵，其中第一个维度仅在 $x$ 的一个行块内进行了归约（reduced）。我们在将数据写入该张量之前，在当前的行块内进行归约。在 Kernel 外部，我们使用 `torch.sum` 对 $\nabla_w$ 进行最终归约，汇总每个行块的结果^1。接下来 `autograd.Function` 的最后一部分就相对简单了：

> ^1 或者，当然，我们也可以专门为此编写自己的 Kernel。

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ... (之前已定义)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE # 这些不必须一致
        n_rows, D = x.shape

        # 我们的策略是让每个线程块首先写入一个部分缓冲区（partial buffer），
        # 然后我们在该缓冲区上进行归约，得到最终的梯度。
        partial_grad_weight = torch.empty((cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
```

最后，我们现在可以得到一个与 `torch.nn.functional` 中实现方式非常相似的函数：

```python
f_weightedsum = WeightedSumFunc.apply
```

现在，在两个 PyTorch 张量 $x$ 和 $w$ 上调用 `f_weightedsum`，将产生如下张量：

```python
tensor([ 90.8563, -93.6815, -80.8884, ...,  103.4840, -21.4634, -24.0192],
       device='cuda:0', grad_fn=<WeightedSumFuncBackward>)
```

请注意附加到该张量上的 `grad_fn` —— 这表明 PyTorch 知道当该张量出现在计算图中时，在反向传播期间应该调用什么。这就完成了我们对加权求和操作的 Triton 实现。

#### 1.3.2 FlashAttention-2 前向传播 (FlashAttention-2 Forward Pass)

您将使用大幅改进的、遵循 FlashAttention-2 [Dao, 2023] 的 Triton 实现来替换您的 PyTorch 注意力实现。FlashAttention-2 采用了一些技巧来按分块（in tiles）计算前向传播，这允许高效的内存访问模式，并避免了在全局内存中物化（materialize）整个注意力矩阵的需要。

在进入本节之前，我们强烈建议您至少阅读原始的 FlashAttention 论文 [Dao et al., 2022]，这将使您直观理解使 FlashAttention 能够高效计算注意力的核心技术：跨分块以在线（online）方式计算 softmax（这是一种在 [Milakov and Gimelshein, 2018] 中提出的技术）。我们还推荐阅读 He [2022]，以进一步了解 GPU 实际执行 PyTorch 代码的方式。

**理解普通（vanilla）注意力机制中的低效之处。** 回顾一下，注意力机制的前向传播（暂且忽略掩码 masking）可以写为：

$$
\mathbf{S} = \mathbf{QK}^\top / \sqrt{d} \quad (4)

$$

$$
\mathbf{P}_{ij} = \text{softmax}_j(\mathbf{S})_{ij} \quad (5)

$$

$$
\mathbf{O} = \mathbf{PV} \quad (6)

$$

标准的反向传播为：

$$
\mathbf{dV} = \mathbf{P}^\top \mathbf{dO} \quad (7)

$$

$$
\mathbf{dP} = \mathbf{dO}\mathbf{V}^\top \quad (8)

$$

$$
\mathbf{dS}_i = \text{dsoftmax}(\mathbf{dP}_i) = \left( \text{diag}(\mathbf{P}_i) - \mathbf{P}_i\mathbf{P}_i^\top \right) \mathbf{dP}_i \quad (9)

$$

$$
\mathbf{dQ} = \mathbf{dS}\mathbf{K} / \sqrt{d} \quad (10)

$$

$$
\mathbf{dK} = \mathbf{dS}^\top \mathbf{Q} / \sqrt{d} \quad (11)

$$

正如我们所见，反向传播依赖于前向传播中产生的一些非常大的激活张量（activations）。例如，计算公式 (7) 中的 $\mathbf{dV}$ 需要用到 $\mathbf{P}$，即形状为 `(batch_size, n_heads, seq_len, seq_len)` 的注意力分数矩阵——这个激活矩阵的大小随序列长度呈平方增长，这解释了我们在面对长序列长度进行注意力基准测试时遇到的显存问题。在普通注意力机制的前向和反向传播过程中，我们付出了极大的显存 I/O 代价，用于在片上（on-chip）SRAM 和 GPU HBM（高带宽内存）之间传输 $\mathbf{P}$ 和其他大型激活。在标准实现中会有数次这样的传输：例如，标准的反向传播实现会在计算公式 (7) 和 (9) 时从 HBM 读取 $\mathbf{P}$。

FlashAttention 的主要目标是避免将注意力矩阵写入 HBM 或从中读取，从而降低 I/O 和峰值显存成本。我们通过三种技术来实现这一点：分块（tiling）、重计算（recomputation）和算子融合（operator fusion）。

**分块 (Tiling)。** 为了避免读写 HBM 中的注意力矩阵，我们无需访问整个输入即可计算 softmax 归约。具体来说，我们重构了注意力计算过程，将输入拆分为块（tiles），并在输入块上进行多次传递，从而逐步执行 softmax 归约。

**重计算 (Recomputation)。** 我们避免在 HBM 中存储形状为 `(batch_size, n_heads, seq_len, seq_len)` 的大型中间注意力矩阵。取而代之的是，我们会在 HBM 中保存某些“激活检查点（activation checkpoints）”，然后在反向传播期间重新计算前向传播的一部分，以获得我们在计算梯度时所需的其他激活。FlashAttention-2 还存储了注意力分数的 logsumexp（对数和的指数）向量 $L$，它将用于简化反向传播计算。$L$ 的表达式为：

$$
L_i = \log \left( \sum_j \exp(\mathbf{S}_{ij}) \right) \quad (12)

$$

在最终的 Kernel 中，我们将以在线（online）的方式计算此值，但最终结果应当一致。结合使用分块和重计算，我们的内存 I/O 和峰值显存占用就不再依赖于 `sequence_length` 的平方，因此我们可以使用更长的序列长度。

**算子融合 (Operator fusion)。** 最后，我们将通过在单个 Kernel 中执行所有操作来避免对注意力矩阵和其他中间激活的重复内存 I/O —— 这被称为算子融合或 Kernel 融合。我们将编写一个单独的 Triton Kernel 用于前向传播，以执行注意力机制所涉及的所有操作，并在 HBM 和 SRAM 之间进行极其有限的数据传输。算子融合部分得益于重计算的支持，因为我们可以避免原本在每次将中间激活保存到 HBM 时所需付出的常规内存 I/O 代价。

如果想更直观地理解这些技术，请查看 FlashAttention 论文 [Dao et al., 2022, Dao, 2023]。

**带重计算的反向传播。** 利用 $L$，我们可以进行适当的重计算并高效地计算反向传播。在开始反向传播之前，我们提前计算出 $D = \text{rowsum}(\mathbf{O} \circ \mathbf{dO})$（其中 $\circ$ 是逐元素乘法）并将其预存入全局内存。该值等同于 $\text{rowsum}(\mathbf{P} \circ \mathbf{dP})$，因为 $\mathbf{P}\mathbf{dP}^\top = \mathbf{P}(\mathbf{dO}\mathbf{V}^\top)^\top = (\mathbf{P}\mathbf{V})\mathbf{dO}^\top = \mathbf{O}\mathbf{dO}^\top$（且对于任何矩阵 $\mathbf{A}$ 和 $\mathbf{B}$，都有 $\text{rowsum}(\mathbf{A} \circ \mathbf{B}) = \text{diag}(\mathbf{A}\mathbf{B}^\top)$）。有了向量 $L$ 和 $D$，反向传播计算就无需再显式执行 softmax。完整的反向传播计算过程现在变为：

$$
\mathbf{S} = \mathbf{QK}^\top / \sqrt{d} \quad (13)

$$

$$
\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - L_i) \quad (14)

$$

$$
\mathbf{dV} = \mathbf{P}^\top \mathbf{dO} \quad (15)

$$

$$
\mathbf{dP} = \mathbf{dO}\mathbf{V}^\top \quad (16)

$$

$$
\mathbf{dS}_{ij} = \mathbf{P}_{ij} \circ (\mathbf{dP}_{ij} - D_i) \quad (17)

$$

$$
\mathbf{dQ} = \mathbf{dS}\mathbf{K} / \sqrt{d} \quad (18)

$$

$$
\mathbf{dK} = \mathbf{dS}^\top \mathbf{Q} / \sqrt{d} \quad (19)

$$

我们可以看到，上述操作序列不需要我们在前向传播期间将注意力分数 $\mathbf{P}$ 保存在 HBM 中——我们在公式 (13) 和 (14) 中通过激活 $\mathbf{Q}, \mathbf{K}$ 和 $L$ 重新计算了它。

**FlashAttention 前向传播细节。** 现在我们对 FlashAttention-2 中使用的技术有了高层理解，我们将深入探讨您将要实现的 FA2 前向传播 Kernel 的细节。为了避免向 HBM 读写注意力矩阵，我们希望使用分块技术，即独立于其他分块计算输出的每一个分块。这就要求我们能够计算 $\mathbf{P}$ 的分块，最好在两个维度上都进行分块（针对 Query 和 Key）。

然而，当对 $\mathbf{S}$ 应用 softmax 时，我们需要归约 $\mathbf{S}$ 的整行以计算 softmax 的分母，这意味着我们不能直接分块计算 $\mathbf{P}$。FlashAttention-2 通过使用*在线 softmax (online softmax)* 解决了这个问题。在下文中，我们将使用下标索引 $i$ 来表示当前的 Query 块，使用上标索引 $(j)$ 表示当前的 Key 块。沿 Query 维度的分块大小为 $B_q$，沿 Key 维度的分块大小为 $B_k$。我们不在隐层维度 $d$ 上进行分块。

我们还需要保留一些逐行的运行值（running values），即 $m_i^{(j)} \in \mathbb{R}^{B_q}$ 和 $l_i^{(j)} \in \mathbb{R}^{B_q}$。逐行值 $m_i^{(j)}$ 是一个运行最大值，记录它的目的是为了能以数值稳定的方式计算 softmax（回想一下我们在作业 1 的 softmax 实现中使用过此技巧）。我们将随着每一组新的 $\mathbf{S}$ 行分块（当 $j$ 增加时）来更新 $m_i^{(j)}$。使用运行最大值，我们可以将未归一化的 softmax 值（分子）计算为 $\tilde{\mathbf{P}}*i^{(j)} = \exp\left(\mathbf{S}*{ij} - m_i^{(j)}\right)$。$l_i^{(j)}$ 是 softmax 分母的一个运行代理（running proxy），并会使用未归一化的 softmax 值进行更新：$l_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) l_i^{(j-1)} + \text{rowsum}(\tilde{\mathbf{P}}_i^{(j)})$。当我们最终写入输出时，我们需要使用 $l_i^{(T_k)}$（这是在处理所有 Key 块之后 $l_i^{(j)}$ 的最终值）来完成对其的归一化。算法 1 展示了应当在 GPU 上实现的前向传播。

---

## **算法 1** FlashAttention-2 前向传播

**要求:** $\mathbf{Q} \in \mathbb{R}^{N_q \times d}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_k \times d}$，分块大小 $B_q, B_k$
将 $\mathbf{Q}$ 拆分为 $T_q = \lceil \frac{N_q}{B_q} \rceil$ 个大小为 $B_q \times d$ 的分块 $\mathbf{Q}*1, \dots, \mathbf{Q}*{T_q}$
将 $\mathbf{K}, \mathbf{V}$ 拆分为 $T_k = \lceil \frac{N_k}{B_k} \rceil$ 个大小为 $B_k \times d$ 的分块 $\mathbf{K}^{(1)}, \dots, \mathbf{K}^{(T_k)}$ 和 $\mathbf{V}^{(1)}, \dots, \mathbf{V}^{(T_k)}$
**for** $i = 1, \dots, T_q$ **do**
    从全局内存加载 $\mathbf{Q}_i$
    初始化 $\mathbf{O}_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q \times d}, l_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q}, m_i^{(0)} = -\infty \in \mathbb{R}^{B_q}$
    **for** $j = 1, \dots, T_k$ **do**
        从全局内存加载 $\mathbf{K}^{(j)}, \mathbf{V}^{(j)}$
        计算 softmax 前的注意力分数分块 $\mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i (\mathbf{K}^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$
        计算 $m_i^{(j)} = \max\left(m_i^{(j-1)}, \text{rowmax}\left(\mathbf{S}_i^{(j)}\right)\right) \in \mathbb{R}^{B_q}$
        计算 $\tilde{\mathbf{P}}_i^{(j)} = \exp\left(\mathbf{S}_i^{(j)} - m_i^{(j)}\right) \in \mathbb{R}^{B_q \times B_k}$
        计算 $l_i^{(j)} = \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) l_i^{(j-1)} + \text{rowsum}\left(\tilde{\mathbf{P}}_i^{(j)}\right) \in \mathbb{R}^{B_q}$
        计算 $\mathbf{O}_i^{(j)} = \text{diag}\left(\exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\right) \mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_i^{(j)}\mathbf{V}^{(j)}$
    **end for**
    计算 $\mathbf{O}_i = \text{diag}\left(l_i^{(T_k)}\right)^{-1} \mathbf{O}_i^{(T_k)}$
    计算 $L_i = m_i^{(T_k)} + \log\left(l_i^{(T_k)}\right)$
    将 $\mathbf{O}_i$ 作为 $\mathbf{O}$ 的第 $i$ 个分块写入全局内存。
    将 $L_i$ 作为 $L$ 的第 $i$ 个分块写入全局内存。
**end for**
返回输出 $\mathbf{O}$ 和 logsumexp $L$。

在进入在 Triton 中实现前向传播的环节之前，我们在下方为您收集了一些编写 Triton Kernel 的通用提示和技巧。

> **Triton 提示与技巧**
>
> - 在 Triton 中可以使用 `tl.device_print` 进行调试：[链接](https://triton-lang.org/main/python-api/generated/triton.language.device_print.html)。有一个设置 `TRITON_INTERPRET=1` 可用于在 CPU 上运行 Triton 解释器，尽管我们发现它存在一些 bug。
> - 定义块指针时，确保它们的偏移量（offsets）是正确的，并且块的偏移量乘以了适当的分块大小。
> - 线程块的启动网格（launch grid）的设置方式为
>   `kernel_fn[(launch_grid_d1, launch_grid_d2, ...)](...arguments...)`
>   在 `torch.autograd.Function` 子类的方法中进行调用，正如我们在加权求和示例中所见。
> - 使用 `tl.dot` 执行矩阵乘法。
> - 要推进（advance）块指针，请使用 `*_block_ptr = *_block_ptr.advance(...)`。

> **问题 (flash_forward)：15 分**
>
> (a) 编写一个实现 FlashAttention-2 前向传播的纯 PyTorch（不含 Triton）的 `autograd.Function`。这会比常规的 PyTorch 实现慢很多，但将有助于您调试稍后编写的 Triton Kernel。
>
> 您的实现应接受输入 `Q`, `K`, 和 `V` 以及一个 `is_causal` 标志，并产生输出 `O` 和 logsumexp 值 `L`。对于此任务您可以忽略 `is_causal` 标志。`autograd.Function` 的 `forward` 方法随后应该使用 `save_for_backward` 缓存 `L, Q, K, V, O` 以为反向传播做准备，并返回 `O`。记住，`autograd.Function` 类的 `forward` 实现总是将其上下文 `ctx` 作为第一个参数。任何 `autograd.Function` 类都需要实现 `backward` 方法，但目前您可以让它直接抛出 `NotImplementedError` 异常。如果您需要用于比对的对照结果，可以在 PyTorch 中实现公式 4 到 6 和 12 来比较您的输出。
>
> 其接口为 `def forward(ctx, Q, K, V, is_causal=False)`。由您自行确定分块大小，但请确保它们至少为 16 × 16。我们总是会使用干净的 2 的幂次且至少为 16 的维度大小来测试您的代码，因此您无需担心越界访问的问题。
>
> **交付物**：一个在 `forward` 方法中实现 FlashAttention-2 的 `torch.autograd.Function` 子类。为了测试您的代码，请实现 `[adapters.get_flashattention_autograd_function_pytorch]`。然后，使用 `uv run pytest -k test_flash_forward_pass_pytorch` 运行测试，并确保您的实现通过该测试。
>
> (b) 遵循算法 1 为 FlashAttention-2 的前向传播编写 Triton Kernel。然后，编写 `torch.autograd.Function` 的另一个子类，在 `forward` 方法中调用这个（已融合的）Kernel，而不是在 PyTorch 中计算结果。一些针对具体问题的提示：
>
> - 为了调试，我们建议将您执行的每一步 Triton 操作结果与您在部分 (a) 中编写的分块 PyTorch 实现进行比较。
> - 您的启动网格（launch grid）应设置为 `(Tq, batch_size)`，这意味着每个 Triton 程序实例将仅加载单个 batch 索引中的元素，且仅读取/写入 `Q`、`O` 和 `L` 的单个 Query 块。
> - Kernel 内部应只有一个单一的循环，用于遍历 Key 块 $1 \leq j \leq T_k$。
> - 在循环末尾推进（Advance）块指针。
> - 使用下方的函数声明（通过我们给您的这个块指针，您应当能够推断出其余指针的设置）：
>
> ```python
> @triton.jit
> def flash_fwd_kernel(
>     Q_ptr, K_ptr, V_ptr,
>     O_ptr, L_ptr,
>     stride_qb, stride_qq, stride_qd,
>     stride_kb, stride_kk, stride_kd,
>     stride_vb, stride_vk, stride_vd,
>     stride_ob, stride_oq, stride_od,
>     stride_lb, stride_lq,
>     N_QUERIES, N_KEYS,
>     scale,
>     D: tl.constexpr,
>     Q_TILE_SIZE: tl.constexpr,
>     K_TILE_SIZE: tl.constexpr,
> ):
>     # 程序索引
>     query_tile_index = tl.program_id(0)
>     batch_index = tl.program_id(1)
>
>     # 将每个指针偏移相应的 batch 索引
>     # 乘以每个张量的 batch 步幅
>     Q_block_ptr = tl.make_block_ptr(
>         Q_ptr + batch_index * stride_qb,
>         shape=(N_QUERIES, D),
>         strides=(stride_qq, stride_qd),
>         offsets=(query_tile_index * Q_TILE_SIZE, 0),
>         block_shape=(Q_TILE_SIZE, D),
>         order=(1, 0),
>     )
>     ...
> ```
>
> 其中 `scale` 为 $\frac{1}{\sqrt{d}}$，`Q_TILE_SIZE` 和 `K_TILE_SIZE` 分别为 $B_q$ 和 $B_k$。您可以在之后对这些值进行调优。
>
> 以下附加指南可能有助于您避免精度问题：
>
> - 片上缓冲区（on chip buffers，如 $\mathbf{O}_i, l, m$）应具有 `dtype tl.float32`。如果您正累加到一个输出缓冲区，请使用 `acc` 参数（即 `acc = tl.dot(..., acc=acc)`）。
> - 在乘法之前，将 $\tilde{\mathbf{P}}_i^{(j)}$ 强制转换（Cast）为 $\mathbf{V}^{(j)}$ 的 dtype，并将 $\mathbf{O}_i$ 强制转换为适当的 dtype，然后再将其写入全局内存。强制转换可以通过 `tensor.to` 完成。您可以通过 `tensor.dtype` 获取张量的 dtype，并通过 `*_block_ptr.type.element_ty` 获取块指针/指针的 dtype。
>
> **交付物**：一个在前向传播阶段利用您的 Triton Kernel 实现 FlashAttention-2 的 `torch.autograd.Function` 子类。实现 `[adapters.get_flash_autograd_function_triton]`。然后，使用 `uv run pytest -k test_flash_forward_pass_triton` 运行测试并确保您的实现通过该测试。
>
> (c) 向您的 `autograd.Function` 实现添加一个标志作为最后一个参数以支持因果掩码（causal masking）。这应该是一个布尔标志，当设置为 `True` 时，它会启用基于索引的比较来进行因果掩蔽。您的 Triton Kernel 应该具备相应的附加参数 `is_causal: tl.constexpr`（这是必需的类型注解）。在 Triton 中，为 Query 和 Key 构造适当的索引向量，并将它们进行比较以形成一个 $B_q \times B_k$ 的方形掩码矩阵。对于被掩蔽掉的元素，在注意力分数矩阵 $\mathbf{S}_i^{(j)}$ 的对应元素中加上常数 `-1e6`。确保在反向传播中使用 `ctx.is_causal = is_causal` 保存掩码标志。
>
> **交付物**：为您的 `torch.autograd.Function` 子类添加一个额外的标志，使用您的 Triton Kernel 实现带因果掩蔽的 FlashAttention-2 前向传播。确保该标志是可选的，且默认值为 `False`，以便之前的测试仍然能够通过。

**使用重计算实现反向传播。** 注意，与公式 7 到 11 中显示的标准反向传播不同，我们可以利用重计算来避免在如公式 13 到 19 所示的反向传播中执行 softmax 操作。这意味着我们可以使用平凡的（trivial）Kernel 来计算反向传播，不需要任何在线（online）技巧。因此，对于这一部分，您可以通过在一个常规的 PyTorch 函数（不使用 Triton）上调用 `torch.compile` 来实现反向传播。

> **问题 (flash_backward)：5 分**
>
> 使用 PyTorch（不使用 Triton）并借助 `torch.compile` 为您的 FlashAttention-2 `autograd.Function` 实现反向传播。您的实现应将 `Q`, `K`, `V`, `O`, `dO`, 和 `L` 张量作为输出（注：按上下文实为作为输入变量处理），并返回 `dQ`, `dK` 和 `dV`。记得计算并使用 $D$ 向量。您可以遵循公式 13 到 19 的计算过程。
>
> **交付物**：要测试您的实现，请运行 `uv run pytest -k test_flash_backward`。

现在让我们将您（部分利用）Triton 实现的 FlashAttention-2 的性能与常规注意力机制的 PyTorch 实现性能进行比较。

> **问题 (flash_benchmarking)：5 分**
>
> (a) 使用 `triton.testing.do_bench` 编写一个基准测试脚本，比较您（部分利用）Triton 实现的 FlashAttention-2 前向和反向传播的性能与常规 PyTorch 实现（即不使用 FlashAttention）的性能。
>
> 具体而言，您将报告一个表格，其中包括针对您的 Triton 和 PyTorch 实现的前向、反向以及端到端（前向-反向）传播的延迟时间（latencies）。在开始基准测试之前随机生成所有必要的输入，并在单张 H100 GPU 上运行该基准测试。始终使用批大小为 1 并启用因果掩蔽。扫略（Sweep over）以下参数网格的笛卡尔积：序列长度取 128 到 65536 之间各种 2 的幂次，嵌入维度大小取 16 到 128 之间各种 2 的幂次，精度使用 `torch.bfloat16` 和 `torch.float32`。您可能需要根据输入大小来调整分块大小。
>
> **交付物**：一张对比结果表格，展示使用上述设置对您的 FlashAttention-2 实现与 PyTorch 实现进行性能比较，并报告前向、反向和端到端的延迟时间。

#### 1.3.3 FlashAttention-2 排行榜 (Leaderboard)

作业 2 的排行榜将测试您实现的 FlashAttention-2 的速度（包括前向和反向传播）。我们向您发起挑战，鼓励您使用能想到的任何技巧来进一步提升实现的性能。限制条件是您不能改变函数的输入/输出格式，并且必须使用 Triton（很遗憾，不能使用 CUDA）。您的输入将使用带因果掩蔽的 BF16 进行测试，且它必须通过与您常规实现相同的测试。该实现必须是您自己编写的，不允许使用预先存在的开源实现。我们将在 H100 环境下针对一个样本进行计时测量，该样本批大小为 1，Query、Key 和 Value 的序列长度为 16,384，且 $d_{\text{model}} = 1024$，含 16 个注意力头。我们将验证排名前 5-10 的提交在正确性与性能上的表现。我们用来为您实现进行计时的测试代码如下：

```python
def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttention2.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)
```

出于测试目的，您可以将重复次数（rep）和预热时间（warmup，单位为毫秒）缩短。以下是一些改进思路：

- 针对您的 Kernel 调优分块大小（使用 Triton autotune 实现此目的！）
- 调优额外的 Triton 配置参数
- 直接在 Triton 中实现反向传播，而不仅仅依赖 `torch.compile`（见下方的 1.3.4 节）
- 在反向传播中对您的输入进行两次遍历（passes），一次用于 $\mathbf{dQ}$，另一次用于 $\mathbf{dK}$ 和 $\mathbf{dV}$，以避免在线程块之间进行原子操作或同步操作。
- 在进行因果掩蔽时提早终止程序实例，跳过那些必定全为零的所有分块
- 将无需掩蔽的块与在平铺对角线上的块分开处理，在计算前者时完全不需要比较索引，在计算后者时只需比较一次索引
- 在 H100 上使用 TMA（Tensor Memory Accelerator，张量内存加速器）功能，遵循类似这篇教程的模式。

将您的最佳成绩提交至排行榜：
`github.com/stanford-cs336/assignment2-systems-leaderboard`

#### 1.3.4 选做 (OPTIONAL)：Triton 反向传播

如果您有兴趣获得更多使用 Triton 的实践经验，和/或希望在排行榜上提交更快的成绩，我们在下方提供了采用分块机制的 FlashAttention-2 反向传播逻辑，您可以自行在 Triton 中实现它。算法 2 展示了应当如何在 Triton 中实现的 FlashAttention-2 反向传播。这里一个核心技巧是计算两次 $\mathbf{P}$，一次在计算 $\mathbf{dQ}$ 的反向传播时，另一次用于计算 $\mathbf{dK}$ 和 $\mathbf{dV}$。这使得我们可以跳过线程块之间的同步。

---

## **算法 2** 分块的 FlashAttention-2 反向传播

**要求:** $\mathbf{Q}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N_q \times d}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_k \times d}, L \in \mathbb{R}^{N_q}$，分块大小 $B_q, B_k$
计算 $D = \text{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^{N_q}$
将 $\mathbf{Q}, \mathbf{O}, \mathbf{dO}$ 拆分为 $T_q = \lceil \frac{N_q}{B_q} \rceil$ 个分块 $\mathbf{Q}*1, \dots, \mathbf{Q}*{T_q}, \mathbf{O}*1, \dots, \mathbf{O}*{T_q}, \mathbf{dO}*1, \dots, \mathbf{dO}*{T_q}$，每个大小为 $B_q \times d$
将 $\mathbf{K}, \mathbf{V}$ 拆分为 $T_k = \lceil \frac{N_k}{B_k} \rceil$ 个分块 $\mathbf{K}^{(1)}, \dots, \mathbf{K}^{(T_k)}$ 和 $\mathbf{V}^{(1)}, \dots, \mathbf{V}^{(T_k)}$，每个大小为 $B_k \times d$
将 $L, D$ 拆分为 $T_q$ 个分块 $L_1, \dots, L_{T_q}$ 和 $D_1, \dots, D_{T_q}$，每个大小为 $B_q$
**for** $j = 1, \dots, T_k$ **do**
    从全局内存加载 $\mathbf{K}^{(j)}, \mathbf{V}^{(j)}$
    初始化 $\mathbf{dK}^{(j)} = \mathbf{dV}^{(j)} = \mathbf{0} \in \mathbb{R}^{B_k \times d}$
    **for** $i = 1, \dots, T_q$ **do**
        从全局内存加载 $\mathbf{Q}_i, \mathbf{O}_i, \mathbf{dO}_i, \mathbf{dQ}_i$
        计算注意力分数分块 $\mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i (\mathbf{K}^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$
        计算注意力概率 $\mathbf{P}_i^{(j)} = \exp\left(\mathbf{S}_i^{(j)} - L_i\right) \in \mathbb{R}^{B_q \times B_k}$
        计算 $\mathbf{dV}^{(j)} \mathrel{+}= (\mathbf{P}_i^{(j)})^\top \mathbf{dO}_i \in \mathbb{R}^{B_k \times d}$
        计算 $\mathbf{dP}_i^{(j)} = \mathbf{dO}_i (\mathbf{V}_j^\top) \in \mathbb{R}^{B_q \times B_k}$
        计算 $\mathbf{dS}_i^{(j)} = \mathbf{P}_i^{(j)} \circ \left(\mathbf{dP}_i^{(j)} - D_i\right) / \sqrt{d} \in \mathbb{R}^{B_q \times B_k}$
        从全局内存加载 $\mathbf{dQ}_i$，然后更新 $\mathbf{dQ}_i \mathrel{+}= \mathbf{dS}_i^{(j)} \mathbf{K}^{(j)} \in \mathbb{R}^{B_q \times d}$，并写回全局内存。为了正确性，这必须是原子操作（atomic）！
        计算 $\mathbf{dK}^{(j)} \mathrel{+}= (\mathbf{dS}_i^{(j)})^\top \mathbf{Q}_i \in \mathbb{R}^{B_k \times d}$。
    **end for**
    将 $\mathbf{dK}^{(j)}$ 和 $\mathbf{dV}^{(j)}$ 作为 $\mathbf{dK}$ 和 $\mathbf{dV}$ 的第 $j$ 个分块写入全局内存。
**end for**
返回 $\mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$。
