# 版本管理约定

## 为什么要有这一页

2026-11-18 之前，`phenotype-modules-v0` 没有版本控制，60 多个脚本和 73 条实验记录全靠文件覆盖写。结果是：

- `outputs/lam_refine/fold_0_lam1.0_mu1.0` 被 size40 的运行覆盖，原来的 size25 结果丢失；
- `66_pcst.py` 的输出目录写死，纯 z 版本与带频率项的版本互相覆盖；
- 无法回滚，也无法回答"这个数字是哪次改动产生的"。

从这里开始按下面四条做。

## 一、新方法开新文件，不改旧脚本

试新算法时新建脚本（编号顺延，如 84、85），不要直接改已经产出过结果的脚本。

必须改旧脚本时，只做**加法**：新增参数并给出默认值，默认值等于旧行为。例如

```python
parser.add_argument("--size", type=int, default=25)   # 默认等于旧行为
```

不允许在旧脚本里删参数、改默认值或改变输出路径格式。

## 二、输出目录带参数

凡是有可调参数并会落盘的脚本，输出目录必须把参数写进路径：

```python
out_dir = paths.out("lam_refine", f"fold_{args.fold}_lam{args.lam}_mu{args.mu}_size{args.size}")
```

否则换个参数就覆盖上一次的结果。

## 三、每完成一组实验提交一次

提交信息格式：`exp(E{编号}): 一句话结论`。例如

```
exp(E72): Pareto 按 p 选点五折 A 级 52.0% 对 56.2%，p=0.236
```

同时更新 `docs/EXPERIMENT_LOG.md`，写清脚本、输入、结果、结论四栏。

## 四、里程碑打 tag

一个阶段有可写结论时打 tag，便于整体回到某个状态：

```bat
git tag -a v0.1.0 -m "初始快照：主线三项主结果 + 数据链 v1"
git checkout v0.1.0 -- scripts/60_lam_refine.py   rem 只回滚单个文件
```

## 当前状态

- 初始快照：提交 153de2d，tag `v0.1.0`（2026-11-18）
- 入库内容：scripts（89）、src、configs、docs（含 27 份文档）、outputs 下的 json/csv 摘要
- 不入库：data/（3.5 GB）、outputs 下的 parquet/npz（51 MB）

## 已发生的覆盖事故（记录在案）

| 时间 | 对象 | 后果 | 现在怎么防 |
|---|---|---|---|
| 11-17 | `outputs/lam_refine/fold_0_lam1.0_mu1.0` | 被 size40 覆盖，丢失 size25 | 输出目录带 size 参数 |
| 11-16 | `outputs/pcst/fold_0` | 纯 z 版与 λ/μ 版互相覆盖 | 需要时按参数分目录 |

代码侧的错误随时可回滚；输出侧的覆盖只能靠"目录带参数"规则避免，这一点比 git 更重要。
