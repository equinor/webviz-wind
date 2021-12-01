from pathlib import Path

import orjson  # Import manually - bug in Dash  # pylint: disable=unused-import

import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
import webviz_core_components as wcc
from webviz_config import WebvizPluginABC
from webviz_config.webviz_store import webvizstore
from webviz_config.common_cache import CACHE


class TurbineModelling(WebvizPluginABC):
    """Insert documentation of plugin here. Is used automatically by `webviz docs'
    Alternative inputfile to the Basic-version made by Anders.
    ---
    * **`filter_cols`:** Dataframe columns that can be used to filter data.
    * **`filter_start`:** The value the filter is set to when it starts\
                          If not set the lowest value is used
    """

    def __init__(
        self,
        app,
        webviz_settings,
        input_file: Path,
        filter_col: str = "REAL",
        filter_start: any = None,
    ) -> None:
        super().__init__()

        self.theme = webviz_settings.theme
        self.input_file = input_file
        self.dataframe = get_data(input_file)
        self.columns = list(self.dataframe.columns)
        self.set_filter_input(filter_col, filter_start)
        self.set_callbacks(app)

    def set_filter_input(self, filter_col, filter_start):
        if filter_col in self.dataframe.columns:
            if (filter_start in self.dataframe[filter_col]) or (filter_start is None):
                self.filter_column = filter_col
                self.filter = np.sort(self.dataframe[self.filter_column].unique())
                if filter_start is None:
                    self.filter_start = np.min(self.filter)
                else:
                    self.filter_start = filter_start
            else:
                print(
                    f"The filter_start value {self.filter_start} is not in {filter_col}"
                )
                raise Exception
        else:
            print(f"The filter_col {filter_col} is not in {self.input_file}")
            raise Exception

    @property
    def layout(self):
        return wcc.FlexBox(
            children=[
                wcc.FlexColumn(
                    children=wcc.Frame(
                        style={"height": "90vh"},
                        children=[
                            wcc.Selectors(
                                label="Visualized output",
                                children=[
                                    wcc.Dropdown(
                                        id=self.uuid("color"),
                                        options=[
                                            {"label": i, "value": i}
                                            for i in self.columns
                                        ],
                                        value="NET",
                                    ),
                                ],
                            ),
                            wcc.Selectors(
                                label=f"Filter on {self.filter_column}",
                                children=[
                                    wcc.Dropdown(
                                        id=self.uuid("filter_value"),
                                        options=[
                                            {"label": i, "value": i}
                                            for i in self.filter
                                        ],
                                        value=self.filter_start,
                                    )
                                ],
                            ),
                            wcc.Selectors(
                                label="Plot type for sorted values",
                                children=[
                                    wcc.RadioItems(
                                        id=self.uuid("radio_button_value"),
                                        options=[
                                            {
                                                "label": "Dots and lines",
                                                "value": "Dots",
                                            },
                                            {
                                                "label": "Bars",
                                                "value": "Bars",
                                            },
                                        ],
                                        value="Dots",
                                    ),
                                ],
                            ),
                        ],
                    )
                ),
                wcc.FlexColumn(
                    flex=4,
                    children=[
                        wcc.Frame(
                            style={"height": "90vh"},
                            highlight=False,
                            color="white",
                            children=wcc.Graph(
                                style={"height": "85vh"},
                                id=self.uuid("figure"),
                            ),
                        ),
                        wcc.Frame(
                            style={"height": "90vh"},
                            highlight=False,
                            color="white",
                            children=wcc.Graph(
                                style={"height": "85vh"},
                                id=self.uuid("figure2"),
                            ),
                        ),
                    ],
                ),
            ],
        )

    def add_webvizstore(self):
        return [(get_data, [{"input_file": self.input_file}])]

    def set_callbacks(self, app) -> None:
        @app.callback(
            Output(component_id=self.uuid("figure"), component_property="figure"),
            [
                Input(component_id=self.uuid("color"), component_property="value"),
                Input(
                    component_id=self.uuid("filter_value"), component_property="value"
                ),
            ],
        )
        def _update_graph1(color_column, filter_value):
            # Same scale accross realizations
            low, high = minmax_dataframe_column(self.dataframe, color_column)
            # Filter the data, but only if the filter is changed
            df = filter_dataframe(self.dataframe, self.filter_column, filter_value)
            self.latest_filterval = filter_value

            fig = px.scatter(
                df,
                x="X",
                y="Y",
                color=color_column,
                hover_name="NAME",
                hover_data=["X", "Y", "GROSS", "NET", "WAKE_LOSS"],
                text="NAME",
                # range_color= (low, high)
            )
            fig.update_traces(textposition="bottom right")
            fig.update_traces(
                marker={"size": 15, "line": {"width": 2, "color": "DarkslateGrey"}}
            )
            fig["layout"].update(self.theme.plotly_theme["layout"])

            return fig

        @app.callback(
            Output(self.uuid("figure2"), "figure"),
            [
                Input(self.uuid("color"), "value"),
                Input(self.uuid("filter_value"), "value"),
                Input(self.uuid("radio_button_value"), "value"),
            ],
        )
        def _update_graph2(color_column, filter_value, plot_type):
            df = filter_dataframe(self.dataframe, self.filter_column, filter_value)
            df = df.sort_values(by=[color_column], ascending=False)
            values = df[color_column].values
            low = np.min(values)
            high = np.max(values)

            if plot_type == "Dots":
                fig = px.scatter(
                    df,
                    x="NAME",
                    y=color_column,
                    color=color_column,
                    hover_name="NAME",
                    hover_data=["GROSS", "NET", "WAKE_LOSS"],
                )
                fig.update_traces(
                    marker={"size": 15, "line": {"width": 2, "color": "DarkslateGrey"}}
                )
                for i in range(len(values)):
                    fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=values[i])
            else:
                fig = px.bar(
                    df,
                    x="NAME",
                    y=color_column,
                    color=color_column,
                    title="Ranking of turbines",
                )
            fig["layout"].update(self.theme.plotly_theme["layout"])
            fig.update_layout(yaxis_range=[low * 0.99, high * 1.01])

            return fig


@webvizstore
def get_data(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file)


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def filter_dataframe(df: pd.DataFrame, column: str, filter_value: int) -> pd.DataFrame:

    return df.loc[df[column] == filter_value]


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def minmax_dataframe_column(dframe: pd.DataFrame, column: str) -> (float, float):

    low = dframe[column].min()
    high = dframe[column].max()
    if (not isinstance(low, (int, float))) or (not isinstance(high, (int, float))):
        low, high = None, None

    return low, high
