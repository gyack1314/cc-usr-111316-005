# 未知灾害关联下的联合损失锐界

本项目保存气候行动研究的合成资料与领域口径。`analysis_data.io`读取JSON及核验原始文件身份，`analysis_data.domain`核对基本形状，`analysis_data.cli`展示数量。详细量纲、时间、契约和统计口径见[资料说明](docs/data-contract.md)。地区、家庭、合同、价格及排放数据均为合成，不代表真实政策评价或投资建议。

Python 3.11以上，无第三方依赖。项目根目录可运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
```

`inspect`只展示资料，不做拟合、求解或分析发布。`invalid-config.json`包含基本配置错误，其余清单在身份上完整，可能包含需领域分析判断的不可识别、无界、冲突或不适用情况。有理数以字符串保留，null不等于0，随机种子属于分析设计。

## 未知关联下的联合损失锐界

`python -m analysis_data.cli bounds data/normal.json` 在不假设维度独立的前提下，对每个阈值分别求 P(total>t) 与 P(total>=t) 的全可行域最小/最大锐界（纯标准库两阶段单纯形，精确有理算术）。每个界输出：

- 达到该界的联合分布，逐原子标注各灾害维状态与**原始年份位置**（可点开查看最坏情况下哪些灾害年份被配到一起）；
- 全部边际/事件/矩约束的残差；
- 可独立复算的对偶证书（弱对偶条件 + 原始/对偶强对偶相等）。

口径要点：上界是"在现有资料支持下最坏能到哪里"，不是预测；边际信息不足不会被包装成一个确定的联合风险数。矛盾资料（`contradictory-association.json`）返回 Farkas 不可行证据；预算耗尽只报当前可行界与证书间隙（状态 `gap`），绝不标锐界。格点过大时用列生成，定价枚举整个笛卡尔格点保证全局有效。独立性、共同分位耦合与给定真实联合表只作参照（报告中带假设警告），不替代可行域界。

可运行：

```sh
python -m unittest discover -s tests -v          # 含锐界、Farkas、列生成、合同、间隙测试
python -m analysis_data.cli bounds data/known-joint-event.json
```

Dockerfile固定Linux镜像；`docker build -t climate-analysis .`构建，`docker run --rm climate-analysis`显示摘要，`docker run --rm climate-analysis python -m unittest discover -s tests -v`运行已有读取检查。资料均可本地读取，运行不需要线上账户。
