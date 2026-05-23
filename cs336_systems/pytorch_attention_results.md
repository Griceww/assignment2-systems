# 1.2.1 `pytorch_attention` 基准测试记录

本文件记录 `cs336_systems.pytorch_attention_benchmark` 的运行方式和一次完整实验结果，便于直接粘贴到作业 writeup。

## 如何运行

在仓库目录 `assignment2-systems` 下执行：

```bash
cd ~/stanford-cs336/assignment2-systems
uv run python -m cs336_systems.pytorch_attention_benchmark \
  --device cuda \
  --dtype float32 \
  --batch-size 8 \
  --d-models 16 32 64 128 \
  --seq-lens 256 1024 4096 8192 16384 \
  --warmup-steps 10 \
  --measure-steps 100 \
  --compile-mode none
```

可选：先做快速自检（更短时间）：

```bash
uv run python -m cs336_systems.pytorch_attention_benchmark \
  --device cuda \
  --d-models 16 \
  --seq-lens 256 1024 \
  --warmup-steps 2 \
  --measure-steps 5
```

## 结果（`--compile-mode none`）

| compile_mode | d_model | seq_len | status | forward_ms_mean | backward_ms_mean | mem_before_bwd_MiB | peak_mem_MiB |
|:---|---:|---:|:---|---:|---:|---:|---:|
| none | 16 | 256 | ok | 0.532 | 1.088 | 20.836 | 28.962 |
| none | 16 | 1024 | ok | 3.086 | 7.393 | 83.344 | 211.845 |
| none | 16 | 4096 | ok | 56.797 | 131.064 | 1064.625 | 3114.626 |
| none | 16 | 8192 | ok | 1394.953 | 6183.790 | 4193.000 | 12389.001 |
| none | 16 | 16384 | oom | - | - | - | - |
| none | 32 | 256 | ok | 0.698 | 2.017 | 21.336 | 29.587 |
| none | 32 | 1024 | ok | 3.216 | 7.581 | 85.344 | 214.345 |
| none | 32 | 4096 | ok | 62.115 | 141.088 | 1072.625 | 3124.626 |
| none | 32 | 8192 | ok | 1113.409 | 4684.886 | 4209.000 | 12409.001 |
| none | 32 | 16384 | oom | - | - | - | - |
| none | 64 | 256 | ok | 0.625 | 1.844 | 22.336 | 30.837 |
| none | 64 | 1024 | ok | 3.484 | 8.172 | 89.344 | 219.345 |
| none | 64 | 4096 | ok | 66.968 | 146.206 | 1088.625 | 3144.626 |
| none | 64 | 8192 | ok | 1139.472 | 3797.958 | 4241.000 | 12449.001 |
| none | 64 | 16384 | oom | - | - | - | - |
| none | 128 | 256 | ok | 0.746 | 2.307 | 24.336 | 33.337 |
| none | 128 | 1024 | ok | 3.851 | 8.812 | 97.344 | 229.345 |
| none | 128 | 4096 | ok | 77.354 | 158.142 | 1120.625 | 3184.626 |
| none | 128 | 8192 | ok | 1162.469 | 13353.915 | 4305.000 | 12529.001 |
| none | 128 | 16384 | oom | - | - | - | - |

## OOM 结论

- 首次出现 OOM 的配置：`d_model=16, seq_len=16384`。
- 在本次实验范围内，`seq_len=8192` 仍可运行，`seq_len=16384` 全部 OOM。

## 编译版结果（`--compile-mode compiled`）

运行命令：

```bash
uv run python -m cs336_systems.pytorch_attention_benchmark \
  --device cuda \
  --dtype float32 \
  --batch-size 8 \
  --d-models 16 32 64 128 \
  --seq-lens 256 1024 4096 8192 16384 \
  --warmup-steps 10 \
  --measure-steps 100 \
  --compile-mode compiled
```

| compile_mode | d_model | seq_len | status | forward_ms_mean | backward_ms_mean | mem_before_bwd_MiB | peak_mem_MiB |
|:---|---:|---:|:---|---:|---:|---:|---:|
| compiled | 16 | 256 | ok | 0.296 | 1.056 | 20.844 | 24.970 |
| compiled | 16 | 1024 | ok | 1.971 | 4.122 | 83.375 | 338.750 |
| compiled | 16 | 4096 | ok | 30.537 | 73.326 | 1064.750 | 2090.751 |
| compiled | 16 | 8192 | ok | 135.283 | 4542.000 | 4193.250 | 8293.251 |
| compiled | 16 | 16384 | oom | - | - | - | - |
| compiled | 32 | 256 | ok | 0.541 | 1.526 | 21.344 | 285.595 |
| compiled | 32 | 1024 | ok | 2.233 | 4.720 | 85.375 | 150.376 |
| compiled | 32 | 4096 | ok | 39.522 | 80.546 | 1072.750 | 2100.751 |
| compiled | 32 | 8192 | ok | 165.909 | 5603.533 | 4209.250 | 8313.251 |
| compiled | 32 | 16384 | oom | - | - | - | - |
| compiled | 64 | 256 | ok | 0.759 | 1.765 | 22.344 | 277.812 |
| compiled | 64 | 1024 | ok | 2.411 | 5.021 | 89.375 | 155.376 |
| compiled | 64 | 4096 | ok | 39.807 | 85.855 | 1088.750 | 2120.751 |
| compiled | 64 | 8192 | ok | 173.643 | 5711.196 | 4241.250 | 8353.251 |
| compiled | 64 | 16384 | oom | - | - | - | - |
| compiled | 128 | 256 | ok | 0.823 | 1.665 | 24.344 | 29.345 |
| compiled | 128 | 1024 | ok | 3.894 | 5.894 | 97.375 | 165.376 |
| compiled | 128 | 4096 | ok | 50.997 | 98.940 | 1120.750 | 2160.751 |
| compiled | 128 | 8192 | ok | 215.458 | 5829.573 | 4305.250 | 8433.251 |
| compiled | 128 | 16384 | oom | - | - | - | - |

对比结论（compiled vs vanilla）：

- OOM 边界不变，最小 OOM 仍为 `d_model=16, seq_len=16384`。
- 在本次重测中，`compiled` 的前向在多数配置上更快（长序列收益更明显）。
- 反向传播收益不稳定：多数配置变快，但 `d_model=32/64, seq_len=8192` 出现变慢。
- 编译模式下出现 `torch._dynamo cache_size_limit` 警告，说明存在较多重编译，可能影响稳定收益。
