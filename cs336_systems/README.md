# `cs336_systems` 使用说明（benchmark + profiler）

这个目录的核心是 [`benchmark.py`](benchmark.py)。它用于完成 `tasks.md` 里 1.1.3（benchmarking）和 1.1.4（Nsight profiling）的实验。

## 1) `benchmark.py` 做什么

`benchmark.py` 会：

- 初始化 `cs336_basics.model.BasicsTransformerLM`（支持 `--preset` 或手动超参）。
- 生成随机 token batch。
- 执行 `warmup` 后进入测量。
- 在 CUDA 上每步调用 `torch.cuda.synchronize()`，计时器用 `timeit.default_timer()`。
- 支持以下模式：
  - `forward`
  - `forward_backward`
  - `both`（分别测 forward/backward）
  - `train_step`（forward + backward + optimizer.step）
- 支持 NVTX 标注（`--nvtx`）和 attention 细粒度标注（`--annotate-attention`）。

## 2) 快速开始（只做 benchmark）

在仓库根目录 `assignment2-systems` 下：

```bash
uv run python -m cs336_systems.benchmark \
  --presets small medium \
  --context-length 256 \
  --warmup 5 --steps 10 \
  --device cuda
```

常用参数：

- `--preset small|medium|...`
- `--presets small medium`（一次跑多个）
- `--mode forward|forward_backward|both|train_step`
- `--markdown` / `--latex-table`

## 3) 先处理 `nsys` 路径（建议每次开新终端先执行）

本机系统自带的 `nsys`（apt 的旧版）在 WSL 下有兼容性问题，当前项目使用本地新版二进制（2025.6.3）。
**注意：先 `cd assignment2-systems`，再执行下面命令。**

```bash
cd ~/stanford-cs336/assignment2-systems
export PATH="$PWD/.tools/nsight_systems-linux-x86_64-2025.6.3.343-archive/target-linux-x64:$PATH"
nsys --version
```

看到版本号后，再执行 profiler 命令。  
（如果想长期生效，可把 `export PATH=...` 写进 `~/.bashrc`。）

## 4) 如何进行 Nsight profiler（1.1.4）

单次示例（small, ctx=128, forward）：

```bash
cd ~/stanford-cs336/assignment2-systems
NSYS_BIN="$PWD/.tools/nsight_systems-linux-x86_64-2025.6.3.343-archive/target-linux-x64/nsys"
uv run "$NSYS_BIN" profile \
  -o nsys_reports/small_ctx128_forward \
  --force-overwrite=true \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  python -m cs336_systems.benchmark \
  --preset small --context-length 128 \
  --mode forward \
  --warmup 1 --steps 1 \
  --device cuda \
  --nvtx --annotate-attention
```

如果你看到 `GLIBC_PRIVATE` 的报错，通常说明你误用了系统的旧 `nsys`，请回到本节命令重新设置并用 `NSYS_BIN` 方式调用。

批量矩阵（small/medium × 128/256/512 × forward/forward_backward/train_step）已经跑过，结果在：

- `nsys_reports_2025/*.nsys-rep`（推荐使用这一批，已确认可见 CUDA HW）
- [`nsight_results.md`](nsight_results.md)
- `nsys_reports/nsight_measurement_summary.csv`
- `nsys_reports/nsight_cuda_api_top_summary.csv`

## 5) 当前已记录的 benchmark 结果（1.1.3 + torch.compile）

配置：`vocab_size=10000, batch_size=4, context_length=256, warmup=5, steps=10, mode=both, device=cuda`

| preset | compile_mode | forward_ms_mean | forward_ms_std | backward_ms_mean | backward_ms_std |
|---|---|---:|---:|---:|---:|
| small | none | 62.9104 | 0.9321 | 135.9059 | 2.1685 |
| medium | none | 213.0854 | 0.4528 | 455.7558 | 0.9389 |
| small | compiled | 54.0965 | 1.5083 | 115.8190 | 2.1912 |
| medium | compiled | 178.2196 | 3.6269 | 375.8774 | 1.5380 |

编译版运行命令（与上表对应）：

```bash
uv run python -m cs336_systems.benchmark \
  --presets small medium \
  --context-length 256 \
  --warmup 5 --steps 10 \
  --mode both \
  --device cuda \
  --compile-mode compiled
```

简要结论：

- 在新版“训练态 forward 且逐步清图”的测量逻辑下，`small`/`medium` 都表现为 compiled 优于 none（forward 与 backward 均加速）。
- 相比之前，`forward_ms_std` 明显收敛，说明“只建图不清图”的不稳定因素已被消除，当前结果更适合用于 none vs compiled 对比结论。

## 6) `pytorch_attention` 结果摘要（1.2.1）

实验配置：`batch_size=8`，`d_model in {16,32,64,128}`，`seq_len in {256,1024,4096,8192,16384}`，`warmup=10`，`steps=100`。

### 关键结果表（none vs compiled）

| compile_mode | d_model | seq_len | status | forward_ms_mean | backward_ms_mean |
|:---|---:|---:|:---|---:|---:|
| none | 16 | 256 | ok | 0.532 | 1.088 |
| none | 16 | 1024 | ok | 3.086 | 7.393 |
| none | 16 | 4096 | ok | 56.797 | 131.064 |
| none | 16 | 8192 | ok | 1394.953 | 6183.790 |
| none | 16 | 16384 | oom | - | - |
| none | 32 | 256 | ok | 0.698 | 2.017 |
| none | 32 | 1024 | ok | 3.216 | 7.581 |
| none | 32 | 4096 | ok | 62.115 | 141.088 |
| none | 32 | 8192 | ok | 1113.409 | 4684.886 |
| none | 32 | 16384 | oom | - | - |
| none | 64 | 256 | ok | 0.625 | 1.844 |
| none | 64 | 1024 | ok | 3.484 | 8.172 |
| none | 64 | 4096 | ok | 66.968 | 146.206 |
| none | 64 | 8192 | ok | 1139.472 | 3797.958 |
| none | 64 | 16384 | oom | - | - |
| none | 128 | 256 | ok | 0.746 | 2.307 |
| none | 128 | 1024 | ok | 3.851 | 8.812 |
| none | 128 | 4096 | ok | 77.354 | 158.142 |
| none | 128 | 8192 | ok | 1162.469 | 13353.915 |
| none | 128 | 16384 | oom | - | - |
| compiled | 16 | 256 | ok | 0.296 | 1.056 |
| compiled | 16 | 1024 | ok | 1.971 | 4.122 |
| compiled | 16 | 4096 | ok | 30.537 | 73.326 |
| compiled | 16 | 8192 | ok | 135.283 | 4542.000 |
| compiled | 16 | 16384 | oom | - | - |
| compiled | 32 | 256 | ok | 0.541 | 1.526 |
| compiled | 32 | 1024 | ok | 2.233 | 4.720 |
| compiled | 32 | 4096 | ok | 39.522 | 80.546 |
| compiled | 32 | 8192 | ok | 165.909 | 5603.533 |
| compiled | 32 | 16384 | oom | - | - |
| compiled | 64 | 256 | ok | 0.759 | 1.765 |
| compiled | 64 | 1024 | ok | 2.411 | 5.021 |
| compiled | 64 | 4096 | ok | 39.807 | 85.855 |
| compiled | 64 | 8192 | ok | 173.643 | 5711.196 |
| compiled | 64 | 16384 | oom | - | - |
| compiled | 128 | 256 | ok | 0.823 | 1.665 |
| compiled | 128 | 1024 | ok | 3.894 | 5.894 |
| compiled | 128 | 4096 | ok | 50.997 | 98.940 |
| compiled | 128 | 8192 | ok | 215.458 | 5829.573 |
| compiled | 128 | 16384 | oom | - | - |

关键结论：

- OOM 边界一致：`none` 与 `compiled` 首次 OOM 均为 `d_model=16, seq_len=16384`。
- 在本次重测中，`compiled` 的前向在大多数配置上优于 `none`（长序列更明显）。
- 反向收益不稳定：多数配置更快，但在 `seq_len=8192` 的若干配置出现变慢。
- 日志出现 `torch._dynamo cache_size_limit` 警告，说明存在重编译，可能影响稳定收益。

## 7) Nsight 与 Mixed Precision 摘要（1.1.4 / 1.1.5）

### 1.1.4 Nsight 计时摘要（measurement_ms）

配置：`size in {small, medium}`，`context in {128, 256, 512}`，`mode in {forward, forward_backward, train_step}`，`warmup=1`，`steps=1`。

| size | context | mode | measurement_ms |
|---|---:|---|---:|
| small | 128 | forward | 30.00 |
| small | 256 | forward | 61.76 |
| small | 512 | forward | 134.41 |
| medium | 128 | forward | 95.73 |
| medium | 256 | forward | 199.52 |
| medium | 512 | forward | 602.29 |
| small | 128 | forward_backward | 99.56 |
| small | 256 | forward_backward | 200.13 |
| small | 512 | forward_backward | 460.65 |
| medium | 128 | forward_backward | 336.06 |
| medium | 256 | forward_backward | 669.21 |
| medium | 512 | forward_backward | 27920.24 |
| small | 128 | train_step | 170.49 |
| small | 256 | train_step | 271.02 |
| small | 512 | train_step | 501.53 |
| medium | 128 | train_step | 1918.05 |
| medium | 256 | train_step | 15405.80 |
| medium | 512 | train_step | 39617.87 |

### 1.1.5(c) BF16 Mixed Precision 摘要

配置：`size in {small, medium}`，`context_length=256`，`mode=both`，`warmup=5`，`steps=10`。

| size | precision | forward_mean_ms | backward_mean_ms |
|---|---|---:|---:|
| small | FP32 (`none`) | 69.5066 | 148.3687 |
| small | BF16 mixed | 38.9221 | 81.5825 |
| medium | FP32 (`none`) | 204.1871 | 446.6771 |
| medium | BF16 mixed | 127.0594 | 261.5562 |

对应加速比（FP32 / BF16）：small forward `1.79x`、small backward `1.82x`；medium forward `1.61x`、medium backward `1.71x`。
