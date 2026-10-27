from pydantic import BaseModel, Field


class ConfigForm(BaseModel):
    """Plugin configuration for data evaluation.

    The data shape (features, target, date) is an AI component owned by the
    project and consumed as a ``datashape`` evaluation input at run time — it is
    intentionally NOT part of the plugin configuration.
    """

    frequency: str = Field(
        default="",
        title="Frequency",
        description="Data frequency of batch starts for time-series analysis (e.g., '30D', '1M')",
        examples=["30D", "1M", "7D"],
    )

    window_size: str = Field(
        default="",
        title="Window Size",
        description="Analysis window size (e.g., '90 days', '3 months')",
        examples=["90 days", "3 months"],
    )
