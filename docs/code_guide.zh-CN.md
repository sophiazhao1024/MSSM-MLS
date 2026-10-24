# 从输入到真实结果的代码导读

本文按计算流程解释仓库，配合根目录 `README.zh-CN.md` 的命令使用。这里只解释交付实现，不把没有公开的训练设置说成作者原来的选择。第一次阅读建议依次看 `domain.py → cards.py → backend.py → learning.py → baseline.py → cli.py`。

## 1. 先建立整体框架

```text
三个 MSSM 参数 x = [M1, mu, tan_beta]
          │
          ▼
cards.py 生成输入卡 → backend.py 调用真实物理程序
          │                FH / HB / HS / micrOMEGAs
          ▼
记录状态及真实观测量 [mh, mH, chi2_hs, omega]
          │
          ├─ 初始阶段：盒内随机取 1000 点，逐点评价
          │
          ▼
learning.py 使用累计真值训练
  四个回归器：[mh, mH, chi2_hs, ln(omega)]
  一个分类器：物理可计算 / 明确物理不可行
          │
          ▼
baseline.py 生成廉价候选
  分类筛选 → 回归预测 → exp 恢复 omega → Eq.(7) → 拒绝采样
          │
          ├─ 足够推荐：100 个推荐 + 5 个随机 → 调真实物理程序
          │                                          │
          │                         新真值加入累计数据┘
          └─ 候选上限仍不足：记录 proposal_exhausted 并停止
```

“回归器”预测连续数值，例如质量；“分类器”预测类别概率，例如能否形成有效物理结果；“代理模型”是对昂贵真实计算的快速近似。它们只帮助决定去算哪里，不能代替真实计算证明某个参数点符合实验。

随机扫描走同一真实物理链，但参数直接从盒内随机生成，不训练网络。

## 2. `src/mssm_mls/domain.py`

这个文件定义问题本身，与“神经网络怎么训练”分开。

| 内容 | 作用 |
|---|---|
| `PARAMETER_NAMES`、`BOUNDS` | 固定输入顺序和参数盒 |
| `FIXED_PARAMETERS` | 保存不扫描的 MSSM 参数 |
| `EXPECTED_TOOL_VERSIONS` | 论文对应物理程序版本 |
| `parameter_array(values)` | 把输入整理成 N×3 数组，检查形状、有限性和范围 |
| `parameter_dict(x)` | 把一行数组转成有名字的参数字典 |
| `sample(count, rng)` | 用传入随机数发生器在盒内均匀取点 |
| `validate_observables(values)` | 按固定顺序检查真实观测量有效性 |
| `score(values)` | 在物理单位下执行 Eq.(7)，返回对数权重和窗口判断 |

这里 N 表示一次送入多少个参数点。N×3 的每一行是 `[M1, mu, tan_beta]`，不是三个相互独立的实验。

论文使用的选点函数可写成：

```text
L(x) = exp(-0.15*mh) × I(123 < mH < 129)
                    × I(chi2_hs < 112.7273)
                    × I(0.08209 < omega < 0.15511)
```

`I(条件)` 在条件真时为 1，否则为 0。因此任何硬窗口不满足，整体权重就是零。程序返回 `log(L)`：窗口通过时为 `-0.15*mh`，窗口未通过时为负无穷。这样后面的接受比较可以避免极小指数直接计算带来的数值问题。

`mh` 是轻的 CP-even Higgs 质量，`mH` 是重的 CP-even Higgs 质量，不能交换。此模型用重的 H 对应约 125 GeV 的观测，质量窗口施加在 `mH` 上。

## 3. `src/mssm_mls/cards.py`

物理程序不能直接读取 Python 字典；这个文件负责把参数写成它认识的文本。

`write_fh_card(path, parameters, top_mass=172.)` 接收输出路径和三个扫描参数，结合 `FIXED_PARAMETERS` 写完整 FeynHiggs 原生输入卡。示例 `[7.3, 7600, 7.725]` 分别写到 M1、μ、tanβ 对应字段。质量按 GeV 写入，不对软质量擅自平方。实部按设置填写，虚部为零。

`MHp=155` 与 `MA0=-1` 采用带电 Higgs 质量输入。`MT=172`、第一二代 A 参数和部分开关来自已说明的历史示例/重建选择，不是论文逐项给出的完整卡。`Qt=Qb=Qtau=0` 遵循对应 FeynHiggs 输入约定。完整登记见同目录下的 `reconstruction_assumptions.txt`。

`write_options(path, overrides)` 生成受限制的 Fortran namelist 配置。它只允许明确支持的键和值，避免拼错字段后静默使用默认值。`DEFAULT_OPTIONS` 包含 FH flags、HS 数据集、质量误差、运行方式和 HB 检索类别。

这两个函数只写文件，不计算 Higgs 质量，也不判断实验允许区。

## 4. `physics/fh_hs_driver.F` 和 `physics/mo_driver.c`

这是 Python 与旧 Fortran/C 软件之间的桥。

`fh_hs_driver.F` 读取 FeynHiggs 输入卡和运行选项，调用对应库，输出机器可读的 Higgs 结果 JSON 与 SLHA 谱文件。它改编自 HiggsSignals 示例，原作者及 GPLv3-or-later 来源信息保留在文件和许可证中。

`mo_driver.c` 读取前一步的 SLHA，用 micrOMEGAs 计算暗物质丰度，输出 JSON。这里使用外部谱；不能悄悄让另一个谱生成器覆盖 FeynHiggs 的谱，再把它说成同一次设置。

SLHA 是粒子物理软件交换谱、耦合和衰变等信息的标准文本格式。驱动还保留诊断字段，便于确认 micrOMEGAs 读入的是预期质量和参数。

两个驱动都需要先与对应版本库编译链接。仓库中有源码不等于电脑上已经有可运行二进制。

## 5. `src/mssm_mls/backend.py`

`PhysicsBackend` 将上述物理链包装成统一接口。扫描器只需要调用 `evaluate(parameters, eval_id)`，不用处理 Fortran/C 的细节。

构造时读取 backend JSON。相对路径按配置文件所在目录解释。例如 `configs/backend.example.json` 中的 `../physics/bin/fh_hs_driver` 指向仓库的 `physics/bin/`。不符合历史版本时默认拒绝；显式允许替代版本的结果仍须标明替代，不能当作历史原版。

`doctor()` 检查程序是否存在且可执行、模板与指纹是否一致。它不调用一个真实参数点验证物理结果，故 `ready=true` 不能替代 `point`。

`evaluate()` 为每个参数点创建独立目录，写输入卡，运行驱动，读取 JSON 和 SLHA，并返回含 `status`、`observables`、`diagnostics`、`provenance` 等字段的字典。超时、损坏输出与明确的物理错误需要区分。真实有效行的核心结构是：

```text
status: ok
observables:
  mh:       轻 Higgs 质量，GeV
  mH:       重 Higgs 质量，GeV
  chi2_hs:  HiggsSignals χ²
  omega:    Ωh²，正的无量纲数
```

文件读写辅助函数拒绝非标准 JSON 的 NaN/Infinity。`adapter_sha256()` 等函数把物理映射源码和程序身份纳入指纹，防止后端悄悄变化却继续复用旧真值。

暖模板相关函数检查 CalcHEP 模型、已编译进程库与依赖身份。普通逐点目录可以写文件，共享暖模板必须保持不变。相关注释可能提及旧项目的 optimized scanner，但本仓库不提供该优化算法。

## 6. `src/mssm_mls/learning.py`

这个文件只负责训练与预测，不决定真实目标是否合格，也不调用物理程序。

### `make_network(seed)`

创建一个 `3 → 60 → 60 → 60 → 60 → 1` 网络。3 个输入是三个扫描参数；4 层隐藏层每层 60 个神经元；末层给一个数。`Linear` 做加权求和加偏置，`ReLU(z)=max(0,z)` 提供分段非线性。`seed` 固定本次初始化随机数，方便在相同环境下重复。

这个函数每调用一次创建一个独立网络，不共享权重。四个回归器和分类器总共调用五次。末层没有 ReLU，因此回归输出可为任意实数；预测出现不合理值由后续观测量/窗口检查处理，而不是临时截成允许值。

### `train_network(...)`

输入包括网络、训练输入 `x`、目标 `y`、epochs、学习率、batch 大小及随机种子。

- **epoch**：完整遍历一次当前全部训练样本。
- **batch**：一次计算损失并更新权重的样本组。
- **loss（损失）**：预测与训练答案的差异指标。
- **Adam**：根据梯度及其历史估计更新权重的优化器。

回归使用 MSE，即预测误差平方的平均。分类使用 `BCEWithLogitsLoss`，把 sigmoid 与二元交叉熵合并成数值稳定的计算。分类网络此时输出未经 sigmoid 的分数，预测阶段再转换为 0～1 概率。

每个 epoch 打乱样本顺序，按 batch 取出，依次执行清空旧梯度、前向预测、计算损失、反向传播、Adam 更新。每个样本都会被使用，没有额外验证集、dropout 或正则化。最后返回训练样本数、实际 epochs、末次训练损失与累计访问行数。这些末次损失用于记录，不用于额外早停。

70 条回归训练样本小于 batch=256，因此每个 epoch 只有一次回归更新。2000 epochs 在这种情况下是2000次更新，不能当成2000×256次更新。

### `Surrogates.__init__` 与 `inputs`

初始化保存训练设置和种子，检查值的范围，固定当前实现使用的 CPU 线程数。`inputs` 检查参数后转成 float32 张量。

float32 是数值的计算表示类型，**不是标准化**。例如 `[10,8000,6]` 仍约为 `[10,8000,6]`，没有变成零均值或 [-1,1]。

### `Surrogates.fit(records)`

接收累计的真实记录，按状态分两组：

| 训练用途 | 所用记录 |
|---|---|
| 回归 | 全部 `ok` 行 |
| 分类 | 全部 `ok` 与 `unphysical` 行 |
| 两者均排除 | `tool_error` 行 |

分类正标签为 `ok`，负标签为 `unphysical`。HB 排除但可计算的点仍是正类。预测值从来不作为新的真实训练标签。

回归真实标签先按 `[mh,mH,chi2_hs,omega]` 排列，只执行 `labels[:,3]=log(labels[:,3])`。例如教学观测量 `[20,126,40,0.1]` 变成 `[20,126,40,-2.302585...]`。其他列不变，没有减均值或除标准差。

然后分别训练四个独立回归器，再训练分类器。每次 `fit` 都重新建立网络和 Adam；旧累计真值保留，旧权重不延续。这是披露的重建设置，不能写成已经找到作者原来的初始化策略。

### `Surrogates.predict(x, threshold=0.5)`

先由分类器计算 `sigmoid` 概率，仅对达到阈值的行调用四个回归器。四列预测合并后，第四列取 `exp`，返回物理单位的 N×4 数组及 N 个分类概率。

分类器拒绝的行保持 NaN 预测。它们只是廉价候选被过滤，不产生新的真实 `unphysical` 记录，也不占真实评价预算。

注意 `threshold=0.5` 是人为指定的运行阈值，不是一个物理常数。网络说“有 0.8 概率”也不是经独立测试校准后的可靠性保证。

### `Surrogates.export()`

导出网络参数、fit 次数和预处理说明，供保存或诊断。它不包含精确恢复扫描所需的所有随机流、真实记录与物理身份，因此不能把导出的 `surrogates.pt` 当成完整续算 checkpoint。

## 7. `src/mssm_mls/baseline.py`

这是扫描流程的组织者。

`ScanSettings` 汇集预算、初始点数、每轮数量、训练参数、候选上限等，并检查合法性。例如预算必须不小于 initial，所以10点安装检查需同时写 `--budget 10 --initial 10`。

`rejection_sample()` 在参数盒均匀生成候选，用 `predict` 得到预测，交给 `score` 算权重。每个候选再抽一个 U∈[0,1)，满足 `log(U) <= log(Lhat)` 才接受，且要求权重是有限正值。这里固定包络 M=1，因为有效正 mh 下 `exp(-0.15*mh)≤1`。

它不把当前候选的最大权重当作新的归一化常数，不做 top-k，也不在推荐不足时改成随机补点。块大小65536只是批量计算方式，每轮候选上限20000000是有限保护。达到上限仍没有所需100点会抛 `ProposalExhausted`。

`run_scan()` 使用分开的随机流管理初始点、候选和额外探索。MLS 初始阶段完成后，它训练累计模型、选择推荐、补5个随机探索、调用真实评价并记录，直到预算结束或明确停止。随机方法直接生成并评价参数点。

`target_decision()` 区分 Eq.(7) 满足、HB 允许和最终目标。默认 `paper-higgs` 同时要求 Eq.(7) 与 HB=1；`eq7-only` 是单独命名的诊断策略。HB实验排除不是分类负例。

`target_summary()` 用同一真实尝试数作分母报告目标比例。便宜的网络候选不加入这个分母，网络预测也不加入成功数。

异常时保留已评价记录和诊断：推荐耗尽写 `proposal_exhausted`，其他异常可能写 `aborted`，中断会留下相应状态。默认连续5次技术失败触发停止，避免把持续坏掉的程序当成有意义扫描。

完整迭代边界 checkpoint 是底层接口。它检查配置、源码、依赖、物理身份和随机流，并把恢复结果写入新目录。当前 CLI 没有把它做成用户恢复命令；中途的任意日志并不天然可恢复。

## 8. `src/mssm_mls/execution.py`

这个文件处理真实结果缓存与历史导入，不改变神经网络或 Eq.(7)。

`CachedBackend` 用 SQLite 保存精确参数对应的真实 `ok`/`unphysical` 结果，要求物理配置、程序和适配器指纹相符。参数采用精确表示匹配，不把相近参数当作同一个点。技术错误不缓存成物理答案。

`import_run()` 需要核对历史记录及原始文件证据，不能只凭一个匿名观测量表相信它来自同一后端。公开脱敏的验证附件用于离线诊断，不替代完整缓存导入证据。

本文件有底层并发支持，但本包 CLI 给缓存层 `workers=1`，因此不能把代码中有线程/并发函数说成默认 MLS 已并行计算。

## 9. `src/mssm_mls/cli.py` 与入口

`__main__.py` 允许 `python -m mssm_mls`；安装后的 `mssm-mls-logomega` 调到同一个 `cli.main`。

| 子命令 | 输入 | 输出/作用 |
|---|---|---|
| `doctor` | 后端配置 | 路径与身份检查 |
| `point` | 一个三参数点、后端、输出目录 | 一次真实评价 |
| `scan` | 后端、settings、method、seed、budget等 | 多点扫描及完整记录 |
| `compare` | 多个运行目录 | 累计真实目标曲线与JSON |

`scan` 的 `--epochs` 等参数是显式覆盖入口，不应为了让演示通过而静默调整后再说仍是同一基线。`compare` 只能比较实际保存结果，不能补齐未完成的评价。

CLI 会在真实运行前做 `doctor`。对于技术错误或推荐耗尽，终端出现错误和非零退出码是有意提供的信号；应看结果文件中的停止原因，不能把它当成“进程结束所以实验完成”。

## 10. 配置与构建脚本

`configs/baseline.json` 是已冻结扫描默认值。`configs/backend.example.json` 是可移植的后端模板。`configs/sources.lock.json` 记录历史包URL、版本、归档SHA256和来源边界。

`scripts/setup_physics.py` 校验下载归档、安全解包、检查编译器、调用构建脚本并写本机配置。它拒绝覆盖未经验证的已有源码目录。`--download-only` 可只取得校验后的源码，不代表程序已构建。

`physics/build_fh_hs.sh` 构建 FH/HB/HS 和 Fortran 驱动；`scripts/build_mo.sh` 构建 micrOMEGAs 和 C 驱动。编译兼容调整有明确说明，不修改物理模型表达式来追求效率。

`scripts/warm_template.py` 从一个本机成功点创建带依赖指纹的暖模板。它保留生成进程库来减少重复开销，不给网络加训练样本。只有确认需要时才执行，先阅读 `--help`。

## 11. 验证脚本与证据文件

`tools/verify_release.py` 是快速发行检查：确认清单、文件哈希和公开验证数据的一致性。它不调用物理程序，也不代表论文效率已经复现。

`scripts/check_frozen_model.py` 加载 `validation/first_round/frozen_first_fit.pt`，检查已有第一轮网络及对应诊断，不重新训练。`weights_only=True` 限定加载权重数据，不把验证附件当任意 Python 程序执行。

`scripts/validate_saved_first_round.py` 是完整离线回放：核对seed=0初始坐标、读取公开保存真值、调用实际 `fit` 和 `rejection_sample`。它训练2000epochs并尝试最多2000万候选，不是几秒钟的软件测试。由于没有真实物理调用，其输出不能称为新完成的20000点实验。

`validation/initial1000_records.jsonl` 是清理私有路径后的公开派生记录，数值与状态字段保留。其来源文件哈希和公开文件哈希需要区分。`validation/first_round/report.json` 记录首轮已观察到的失败；其他来源和诊断文件说明结论适用范围。

`tests/` 使用小型人工夹具检查标签变换、接受条件、后端返回协议、缓存和恢复契约。它们服务于发现程序错误，不能靠测试通过证明真实物理预测正确。

## 12. 阅读代码时始终分清四件事

1. **源码可导入、测试通过**：软件基本结构与契约得到验证。
2. **真实单点通过物理程序**：这一具体坐标的后端调用和输出完成。
3. **MLS推荐成功并持续推进**：网络能推荐被实际评价的新点；当前保存首轮在这里失败。
4. **科学结果复现**：配置、效率口径、覆盖性及统计证据足够支持与论文比较；本版尚未达到这一结论。

把这些层次分别记录，别人就能看懂哪些是已经证实的事实，哪些仍需要找回历史设置或进一步实验。
