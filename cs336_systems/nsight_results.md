# CS336 Systems Results (small + medium)

## 统一口径说明

- 由于本机资源限制，本页使用 `small` / `medium` 代替 2.7B 全量实验。
- 所有实验均在 `device=cuda` 下运行，且已启用 warm-up；统计仅针对 measurement 区间。
- Nsight 实验使用 NVTX 标注（`warmup/measurement/forward/backward/optimizer_step` + attention 子区间）。
- Memory profiling 使用 `torch.cuda.memory._record_memory_history` / `_dump_snapshot` / `max_memory_allocated`。

## 1.1.4 Nsight Systems（计时）

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

来源：`nsys_reports_2025/nsight_measurement_summary.csv`

结论（简述）：
- forward 时间随 model/context 单调上升，并与 Python 侧 benchmark 输出量级一致。
- `train_step` 显著慢于 `forward`，额外开销主要来自 backward + optimizer。
- 在当前 WSL 环境下，部分 report 无法稳定给出完整 kernel summary；NVTX 视图仍可用于阶段级归因。

## 1.1.5(c) BF16 Mixed Precision（速度）

配置：`size in {small, medium}`，`context_length=256`，`mode=both`，`warmup=5`，`steps=10`。

| size | precision | forward_mean_ms | backward_mean_ms |
|---|---|---:|---:|
| small | FP32 (`none`) | 69.5066 | 148.3687 |
| small | BF16 mixed | 38.9221 | 81.5825 |
| medium | FP32 (`none`) | 204.1871 | 446.6771 |
| medium | BF16 mixed | 127.0594 | 261.5562 |

对应加速比（FP32 / BF16）：small forward `1.79x`、small backward `1.82x`；medium forward `1.61x`、medium backward `1.71x`。

## Memory Profiling（tasks.md 197-242）

配置：`size in {small, medium}`，`context in {128, 256, 512}`，`mode in {forward, train_step}`，`precision in {none, bf16}`，`warmup=1`，`steps=1`。

原始汇总：`cs336_systems/memory_profile_matrix.log`  
结构化表：`cs336_systems/memory_profile_summary.csv`  
快照目录：`memory_snapshots/`

### Peak Memory：forward

| size | context | fp32_peak_mib | bf16_peak_mib | bf16_vs_fp32 |
|---|---:|---:|---:|---:|
| small | 128 | 560.24 | 771.65 | 1.377x |
| small | 256 | 618.10 | 785.14 | 1.270x |
| small | 512 | 734.48 | 903.19 | 1.230x |
| medium | 128 | 1684.36 | 2425.12 | 1.440x |
| medium | 256 | 1742.22 | 2438.39 | 1.400x |
| medium | 512 | 1920.84 | 2594.84 | 1.351x |

### Peak Memory：train_step

| size | context | fp32_peak_mib | bf16_peak_mib | bf16_vs_fp32 |
|---|---:|---:|---:|---:|
| small | 128 | 2241.34 | 2235.94 | 0.998x |
| small | 256 | 3081.34 | 2795.80 | 0.907x |
| small | 512 | 5148.40 | 4208.93 | 0.818x |
| medium | 128 | 6701.05 | 6763.85 | 1.009x |
| medium | 256 | 8690.82 | 8141.92 | 0.937x |
| medium | 512 | 14055.41 | 11805.28 | 0.840x |

### 对应 deliverable 的简要结论

- (a) timeline 形态：`forward` timeline 更短且单峰；`train_step` 通常可分辨为 forward 上升、backward 峰值、optimizer 阶段尾部高占用。可通过 snapshot 文件在 [pytorch memory_viz](https://pytorch.org/memory_viz) 中打开查看。
- (b) peak memory：在 FP32 下，`forward` 峰值约从 `560 MiB`（small-128）增长到 `1921 MiB`（medium-512）；`train_step` 峰值约从 `2241 MiB` 增长到 `14055 MiB`。
- (c) mixed precision 影响：对 `train_step`，BF16 在大 context 下显著降峰（约 `6%~18%`）；但在 `forward` 场景下本机观测到 BF16 峰值更高（`1.23x~1.44x`），说明仅看峰值时收益与具体执行路径/缓存行为有关，不能简单假设“BF16 一定降内存”。
