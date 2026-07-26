# Agent 目录说明

`method/agents` 按运行职责组织。某个工具如果只被一个 agent 调用，就放在该
agent 自己的目录里，避免形成难追踪的公共工具池。

## 子包

- `slide_parser/`
  - 入口：`SlideParserAgent`
  - 职责：抽取 PPTX 元素、标注可见角色、把 chart/table body 导出为 CSV，
    并生成 `ppt_representation`。
  - 局部工具：无。PPTX 抽取逻辑直接放在 `agent.py` 中，因为它不是
    LangChain tool，也只服务 parser。

- `slide_analysis/`
  - 入口：`SlideAnalysisAgent`
  - 职责：调用 LLM prompt 抽取 `summary.data_source`、
    `caption.data_source` 和每个表的 `calculation_logic`。
  - 局部工具：无。该阶段只调用配置好的 LLM client。

- `data_source_validation/`
  - 入口：`DataSourceValidationAgent`
  - 局部工具：`tools.py`
  - Agent 调用的 LangChain tool：`slot_query`
  - 职责：把可见 data-source 描述聚合成 `final_data_source`，用数据库工具
    验证 slots，并在需要时向共享 client 请求 scope 修正。

- `content_validation/`
  - 入口：`ContentValidationAgent`
  - 局部工具：`tools.py`
  - Agent 调用的 LangChain tool：`sql_retrieve`、`analysis_execute`、
    `compare_table`、`record_issue`、`ask_client`、`modify_chart`、
    `modify_table`、`modify_textbox`
  - 职责：基于验证后的 state 调用 SQL、function logic 和表格比较工具，
    由 ReAct agent 判断 chart/table、caption、summary 是否需要修复。
    可由证据直接确定的 Value 和 analytical Claim error 通过 `record_issue`
    记录并修复；Scope error 和 presentation-form Claim error 通过
    `ask_client` 请求澄清。

- `client/`
  - 入口：`ClientAgent`
  - 职责：benchmark client simulator。它把 agent 的显式请求和
    `feedback_episode.json` 匹配，并返回客户式 `response`。

## 共享文件

- `types.py`：agent 之间共享的 workflow dataclass。
- `../prompts/`：可直接编辑的 prompt 文本。
- `../pipeline/slide_review_workflow.py`：只负责编排，不持有某个 agent 的专用工具。
