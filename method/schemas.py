from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryFilter(BaseModel):
    """数据库查询槽位。

    Args:
        city: 城市名。
        block: 板块、片区或商圈名。
        start_date: 查询起始日期，格式为 YYYY-MM-DD。
        end_date: 查询结束日期，格式为 YYYY-MM-DD。
        table_name: 底层数据库表名。
    """

    city: str
    block: str
    start_date: str
    end_date: str
    table_name: str

    @property
    def sql_params(self) -> dict[str, str]:
        """转换为 DAO 使用的 SQL 参数。"""
        return {
            "city": self.city,
            "block": self.block,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "table_name": self.table_name,
        }


class BinningRule(BaseModel):
    """维度生成规则。

    `period` 用于把 `date_code` 转成年/月；`range` 用于把面积或价格字段
    切成可展示区间。

    Args:
        source_col: 底层原始数据列名。
        target_col: 计算后生成的展示维度列名。
        method: 维度生成方式。
        step: range 分箱步长。
        format_str: range 标签格式。
        time_granularity: period 时间粒度。
    """

    source_col: str = Field(..., description="Source column, e.g. dim_area")
    target_col: str = Field(..., description="Target column, e.g. area_range")
    method: Literal["range", "period"] = Field(
        ...,
        description="Binning method: range or period",
    )
    step: float | int | None = Field(None, description="Bin step, e.g. 20")
    format_str: str | None = Field(None, description="Format template, e.g. '{}-{}m²'")
    time_granularity: Literal["year", "month"] | None = Field(
        None,
        description="Time granularity",
    )


class MetricRule(BaseModel):
    """指标计算规则。

    Args:
        name: PPT 中展示的指标名，也是结果列名。
        source_col: 底层原始数值列名。
        agg_func: 聚合函数。
        filter_condition: 聚合前过滤条件，例如 `{"trade_sets": 1}`。
    """

    name: str = Field(..., description="Display metric name and result column name")
    source_col: str = Field(..., description="Source numeric column, e.g. supply_sets")
    agg_func: Literal["sum", "count", "mean", "max", "min"] = Field(
        "sum",
        description="Aggregation function",
    )
    filter_condition: dict[str, Any] | None = Field(None, description="Pre-filter condition")


class TableAnalysisConfig(BaseModel):
    """单个 chart/table 的可执行计算配置。

    Args:
        table_type: 表格结构类型。
        dimensions: 行/列维度生成规则。
        metrics: 指标计算规则。
        crosstab_row: 交叉表行维度。
        crosstab_col: 交叉表列维度。
    """

    model_config = ConfigDict(extra="ignore")

    table_type: Literal[
        "field-constraint",
        "constraint-field",
        "cross-constraint",
    ] = Field(..., description="表格结构类型")
    dimensions: list[BinningRule] = Field(default_factory=list, description="Dimension rules")
    metrics: list[MetricRule] = Field(default_factory=list, description="Metric rules")
    crosstab_row: str | None = Field(None, description="Crosstab row dimension")
    crosstab_col: str | None = Field(None, description="Crosstab column dimension")
