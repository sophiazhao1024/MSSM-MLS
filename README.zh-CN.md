# MSSM MLS 历史方法重建版使用说明

[English README](README.md) · [逐文件代码导读](docs/code_guide.zh-CN.md)

这是一份可整理上传 GitHub 的完整源码项目，用来重建论文 *Exploring Supersymmetry with Machine Learning* 的 Model-4 MSSM 扫描。项目文件夹名为 `mssm-mls-historical-reconstruction`，Python 发行名和命令名仍为 `mssm-mls-logomega`，Python 导入名为 `mssm_mls`。

**先说明最重要的边界：这不是找回的作者原始程序，也不是已经复现约 80% 效率的成品。** 它保留我们已经审计的“原始输入、仅丰度取 ln”实现和已经发现的首轮失败，提供可检查的源码、配置、数据及诊断入口。上传 GitHub 后应据此介绍项目，不能把打包完成写成科学复现完成。

## 1. 这一版能做什么

它包含随机扫描和 MLS 扫描的完整调用链，能够生成 MSSM 输入卡、调用对应年代的物理软件、读取真实观测量、训练四个回归器及一个分类器，并按论文 Eq.(7) 进行推荐选点。

输入为 `[M1, mu, tan_beta]` 的原始物理数值；单位依次为 GeV、GeV、无量纲。训练标签为 `[mh, mH, chi2_hs, ln(omega)]`。其中 `omega` 表示 Ωh²，`ln` 是自然对数。前三个标签保持原值，预测的第四项经过一次 `exp` 恢复丰度。

论文的依据和重建选择需要分开：四个独立回归器、4×60 隐藏层、Adam、最多 2000 epochs、1000 初始点、每轮 100 推荐点加 5 随机点有论文依据；固定恰好训练 2000 epochs、batch=256、lr=0.001、每轮重新初始化、分类阈值 0.5、候选上限以及完整物理输入卡并非全部由论文唯一指定。论文没有记载额外标准化，不等于能证明遗失源码从未使用。xBIT 是参考实现，也不能代替作者原程序。

## 2. 已知失败必须怎样理解

保存的初始 1000 条真值中有 70 条 `ok`、825 条 `unphysical`、105 条 `tool_error`。用原配置和 seed=0 训练后，在 20,000,000 个神经网络候选中：

| 项目 | 结果 |
|---|---:|
| 分类器放行 | 1,805,893 |
| 预测 Eq.(7) 权重大于零 | 0 |
| 接受的推荐点 | 0 / 100 |
| 首轮推荐状态 | `proposal_exhausted` |

这 2000 万是廉价候选预测数量，不是物理程序调用次数。此次保存真值回放没有新增真实物理评价，没有完成一次 1105 点或 20000 点扫描。

诊断进一步确认：这组训练后权重的 χ² 和 ln(Ωh²) 网络在整个参数盒内均无法进入目标窗口。因此，固定这些权重时，只增加候选也无效。这个结论针对这组网络，不代表任何不标准化的方法都会失败；也不能把失败唯一归因于某一个尚未隔离的训练设置。

正常 `scan` 不会偷偷读入附带的验证记录。重新运行真实物理程序时，原来的技术失败可能变化，因而不能保证新机器的首轮轨迹与归档完全相同。完整结果见 `validation/first_round/report.json` 和 `docs/` 中的证据说明。

本次 0.1.1 发行验证在 macOS arm64、Python 3.12.14 下通过 **124 项软件测试**；公开脱敏数据经过完整 2000 epochs 训练和 2000 万候选回放，训练及推荐报告与原始实验一致；保存权重检查也通过。本次打包没有新增物理调用或重新构建外部物理库。验证范围见 `docs/release_validation.json`；17 个数值/物理核心文件的逐字节保留检查见 `docs/core_preservation.json`。

## 3. 安装之前需要什么

先解压完整源码 ZIP，或取得完整 Git 仓库。仅安装 Python wheel 不包含仓库中的物理编译脚本和配置模板，不能代替完整源码目录。

建议使用 Python 3.12，并创建独立虚拟环境。不要把其他同样提供 `mssm_mls` 模块的旧项目安装在同一个环境中。目录路径避免空格，例如 `~/research/mssm-mls-historical-reconstruction`；旧 CalcHEP 会生成 shell/make 路径，不能可靠处理带空格的安装位置。

软件分两层安装：

1. **Python 层**：读代码、执行测试、检查保存网络、回放训练，只需要 Python 依赖。
2. **真实物理层**：运行 `point` 或 `scan` 还需要编译 FeynHiggs、HiggsBounds、HiggsSignals、micrOMEGAs 及本项目驱动。

macOS 需要 C/C++/Fortran 编译器、make、Command Line Tools 和可用 SDK。Linux 分支已提供，但不能据此声称你的 Linux 环境已经构建通过。原生 Windows 不是当前构建目标，可使用合适的 Linux 环境。第三方程序不是 Python `pip install` 自动获得的。

## 4. 先把 Python 层安装好

在终端进入解压后的目录，执行：

```bash
cd mssm-mls-historical-reconstruction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
mssm-mls-logomega --help
python tools/verify_release.py
```

`-e` 表示安装当前源码目录，修改源码后导入的就是这个目录中的代码。`.[test]` 同时安装测试依赖。以后重新打开终端，需要重新进入目录并执行 `source .venv/bin/activate`。

`requirements-tested.txt` 记录已验证环境中的精确版本。若要申请安装同一组依赖，可在独立环境中先执行 `python -m pip install -r requirements-tested.txt`，再执行 editable install。它不是对所有平台的安装成功保证。

`pytest` 检查软件契约，测试里的人工数据和假可执行文件不是真实 MSSM 结果。`verify_release.py` 检查发行清单和数据完整性。仓库附有 CI 配置不代表已经在 GitHub 上运行通过。

## 5. 不安装物理软件也能检查什么

最快的入口是检查归档的第一轮网络：

```bash
python scripts/check_frozen_model.py --output runs/frozen-check.json
```

它读取 `validation/first_round/frozen_first_fit.pt`，使用 `weights_only=True` 加载权重，复查保存网络的诊断结论；不重新训练，也不调用物理程序。它不能证明一套新编译的物理后端正确。

若希望完整重复“训练五个网络→尝试首轮推荐”，执行：

```bash
python scripts/validate_saved_first_round.py --project . \
  --records validation/initial1000_records.jsonl \
  --output runs/offline-first-round.json
```

这会重新训练并最多考察 2000 万候选，耗时明显更长。其预期历史结果是 `proposal_exhausted`；准确复现失败不等于完成扫描。

附带的 `initial1000_records.jsonl` 是已说明处理规则的公开派生数据：保留数值、状态及训练所需字段，清理机器私有路径。来源文件与公开文件的哈希分别登记，不能称二者逐字节相同。不要将这份附录当成新扫描自动增加的样本。

## 6. 编译真实物理后端

准备好系统编译器后：

```bash
python scripts/setup_physics.py
```

编译器不在默认 PATH 时：

```bash
python scripts/setup_physics.py --toolchain-bin /path/to/compiler/bin
```

脚本依据 `configs/sources.lock.json` 下载固定来源，校验 SHA256，编译历史程序和项目驱动，并生成 `configs/backend.local.json`。若它已经存在，脚本改写 `configs/backend.built.json`；后续命令必须使用脚本实际打印的配置路径。

| 程序 | 版本 | 作用 |
|---|---|---|
| FeynHiggs | 2.13.0 | MSSM Higgs 质量、耦合及相关输出 |
| HiggsBounds | 4.3.1 | Higgs 搜索排除检验 |
| HiggsSignals | 1.4.0 | Higgs 数据 χ² |
| micrOMEGAs | 4.3.2 | 暗物质丰度 Ωh² |

前三者使用固定发布归档；micrOMEGAs 使用固定 HEPcodes 历史镜像提交，尚未获得原官方压缩包的哈希来确认二者字节一致。版本号相同本身不足以证明全部历史设置一致。详细来源和物理补全选项见 `docs/reconstruction_assumptions.txt` 与来源登记。

编译日志保存在 `build/build_logs/`。旧程序可能把数据目录编进二进制，安装后移动目录可能需要重建。源码包不附第三方完整代码或已编译二进制；安装下载后还需遵循各软件的许可与引用要求，见 `LICENSE`、`THIRD_PARTY_NOTICES.txt`。

构建完成先执行：

```bash
mssm-mls-logomega doctor --backend configs/backend.local.json
```

`ready=true` 表示路径、可执行文件及身份检查通过，不表示真实单点成功，也不表示效率复现。生成的本机配置包含绝对路径和环境，已列入忽略规则，不应提交 GitHub。

## 7. 第一个真实单点

```bash
mssm-mls-logomega point --backend configs/backend.local.json \
  --M1 7.3 --mu 7600 --tan-beta 7.725 \
  --output runs/first-point
```

这是已有历史单点验证的坐标，仍应检查自己机器上的运行结果。不要把预期观测量硬编码为真实结果。`runs/first-point/point.json` 给汇总，`physics/point_000001/` 保留输入卡、SLHA、输出和日志。每次输出目录必须是新的，程序不会静默覆盖旧实验。

结果分三类：

| 状态 | 含义 | 机器学习用途 |
|---|---|---|
| `ok` | 物理计算有效，观测量可读 | 四个回归器训练数据；分类正例 |
| `unphysical` | 明确识别的物理不可行情况 | 分类负例；不训练观测量 |
| `tool_error` | 程序、超时、格式或数值技术问题 | 不训练任何网络 |

`ok` 不等于满足实验约束。HiggsBounds 排除点仍可以是 `ok`，不能把它当作“物理不可计算”的分类负例。

## 8. 随机扫描与 MLS 命令

先用 10 次真实随机评价检查安装：

```bash
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method random --seed 0 \
  --budget 10 --initial 10 --output runs/random-check-s0
```

这里同时写 `--initial 10` 是因为设置检查要求预算不小于初始点数。这是小规模安装检查，不是修改标准 MLS 初始 1000 点配置，也不是效率实验。

尝试完整首轮 MLS：

```bash
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method mls --seed 0 \
  --budget 1105 --output runs/mls-attempt-s0
```

它先评价 1000 个随机点，再训练，尝试接受 100 个推荐点，然后添加 5 个随机探索点。**已保存的真值回放在推荐阶段停止，本命令不能称为已经跑通的烟测。** 推荐耗尽后不会偷偷执行随机补点来推进后续训练。

完整预算的命令为：

```bash
mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method random --seed 0 \
  --budget 20000 --output runs/random-20000-s0

mssm-mls-logomega scan --backend configs/backend.local.json \
  --settings configs/baseline.json --method mls --seed 0 \
  --budget 20000 --output runs/mls-20000-s0
```

这些是运行入口，不是“本版已完成 20000 点”的证明。所有真实评价尝试，包括技术失败，都占逻辑预算。默认连续 5 次 `tool_error` 会停止运行。20000−1000 不能整除 105，末批按严格预算截为 95 推荐＋5 随机，这是明确的工程约定。

主要默认值在 `configs/baseline.json`：epochs=2000、lr=0.001、batch=256、分类阈值=0.5、候选块=65536、每轮候选上限=20000000。

## 9. 结果文件和效率口径

| 文件 | 内容 |
|---|---|
| `records.jsonl` | 每次真实评价的参数、状态、观测量、目标判定 |
| `training_history.json` | 每轮训练数量、损失和推荐统计 |
| `summary.json` | 运行状态、已评价数量、目标数量和累计比例 |
| `manifest.json` | 算法、种子、配置、变换方式及实现身份 |
| `physics_manifest.json` | 物理程序身份与后端配置 |
| `physics/point_*/` | 各点原始输入、外部程序输出、SLHA 和日志 |
| `execution.json` | 启用缓存时的复用与实际新增调用统计 |

默认最终目标策略叫 `paper-higgs`：真实 Eq.(7) 窗口满足且 `HB_result=1`。同时保留 `target_eq7`，以便区分 Eq.(7) 本身与加入 HB 后的数量。

`--target-policy eq7-only` 只是另行标注的诊断口径，不能用它隐去 HB 排除后再宣称复现全部 Higgs 约束。预测通过也不计为真实目标。

```bash
mssm-mls-logomega compare runs/random-20000-s0 runs/mls-20000-s0 \
  --output runs/comparison-s0
```

`compare` 生成曲线和 JSON；若 MLS 提前停止，曲线也只是到已完成的评价数。不能把部分曲线说成相同 20000 预算的完成结果。不同目标策略会被拒绝混合；还要人工核对物理设置、种子、版本、实际预算和候选上限。

## 10. 缓存、暖模板和恢复边界

重复做真实计算时可在 `scan` 加 `--cache runs/truth.sqlite`。缓存只复用精确参数和相同物理身份下的 `ok` 或明确 `unphysical` 结果；不会用近邻点、网络预测、技术失败充当真值。缓存命中仍计入逻辑评价预算，实际新增调用另记。不能用命中过缓存的耗时直接证明 MLS 加速。

`--reuse-run` 是向缓存导入可核查的历史结果，需要同时指定 `--cache`。它不是恢复旧进程或旧网络。公开的脱敏附件服务于离线诊断，不应当作具有完整原始文件证据的缓存导入来源。

`scripts/warm_template.py` 能从本机一个成功物理点创建带指纹的 CalcHEP 暖模板，减少重复生成工作。这是可选的运行工程措施，普通构建及单点命令不依赖它。若要使用先看脚本 `--help`，不要从另一台机器复制已编译模板。

底层 `run_scan` 有完整迭代边界 checkpoint 接口，但当前 CLI 没有恢复选项；不能把已有输出目录直接作为续算目录。`surrogates.pt` 是推理状态，也不是精确恢复整个扫描所需的 checkpoint。