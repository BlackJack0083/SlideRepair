# Slide Review Evaluation

评估以一个 injected PPT 为一个 case。Review agents 只能读取：

- `slide.pptx`
- `slide.png`
- ClientAgent 暴露的 `feedback_episode.json` 回复

`corruption.json`、GT YAML 和 GT PPTX 只在 workflow 完成后由 evaluator 读取。

## 主指标

### Detection Macro-F1

先把检测结果转成原子标签：

- scope error：`(error_type, scope_error_type, field)`
- value/claim error：`(error_type, target)`

对 gold annotations 定义的每个合法标签计算数据集级 F1，再对这些标签取宏平均。预测中的非法标签不构成新的 benchmark 类别，但会使 case-level exact match 失败。

### Exact Detection Accuracy

预测标签集合与 `corruption.json` 标签集合完全一致的 case 比例。多报或漏报任意标签均视为失败。

### Task Success Rate

repaired PPTX 与 GT PPTX 在以下内容上全部一致的 case 比例：

- slide 和 shape 数量
- shape 类型与位置尺寸
- textbox 文本
- chart/table 展示数据

数值比较允许绝对误差不超过 `0.5`，用于容纳 PPT 展示舍入。

## 阶段指标

阶段指标用于定位端到端失败来源，不替代主指标。

### Parser Accuracy

同时要求以下结果正确：

- element role
- caption 与 chart/table 的配对
- 从 PPTX 导出的 chart/table 数据

### Data Source Extraction Accuracy

比较 SlideAnalysisAgent 抽取的每个 caption datasource 与 injected YAML，包括 table、scope filters 和 `select_columns`。报告 table-level accuracy；全部 table 正确时 case success。

### Function Logic Execution Accuracy

将预测的 `calculation_logic` 与 GT datasource 组合后真实执行，并将结果与 GT chart/table 数据比较。这样只测 function logic，不让 datasource 抽取错误污染该阶段。

### Data Source Validation Accuracy

比较 DataSourceValidationAgent 输出的完整 final datasource 与 GT datasource。table、city、block、start_date、end_date 必须全部一致。

### Content Repair Accuracy

对 `corruption.json.operations` 中每个注入点，比较 repaired PPTX 对应文本或数据与 GT PPTX。最终报告正确注入点比例。

## 失败规则

pipeline 超时或异常的 case 不会从分母中删除。该 case 的 Task Success 和所有阶段指标记为 0，Detection 按空预测计算。

## 工作目录

数据集只读。Runner 将每个 case 的 PPTX 和 PNG 复制到：

```text
<output_dir>/work/<sample_id>__<injection_id>/
```

Parser CSV、tool artifacts 和 repaired PPTX 均写入该目录。每个 case 的完整 state、tool log、detected issues 和指标写入 `<output_dir>/<case_id>.json`。
