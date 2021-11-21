from pathlib import Path

import orjson  # Import manually - bug in Dash  # pylint: disable=unused-import

import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import webviz_core_components as wcc
from webviz_config import WebvizPluginABC
from webviz_config.webviz_store import webvizstore


class TurbineModelling(WebvizPluginABC):
    """Insert documentation of plugin here. Is used automatically by `webviz docs`"""

    def __init__(self, app, webviz_settings, input_file: Path) -> None:
        super().__init__()

        self.theme = webviz_settings.theme
        self.input_file = input_file
        self.dataframe = get_data(input_file)
        self.set_callbacks(app)

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
                                    wcc.RadioItems(
                                        id=self.uuid("radio_button_value"),
                                        options=[
                                            {
                                                "label": "Average wake loss",
                                                "value": "Average wake loss",
                                            },
                                            {
                                                "label": "Average power",
                                                "value": "Average power",
                                            },
                                        ],
                                        value="Average wake loss",
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
                        )
                    ],
                ),
            ],
        )

    def add_webvizstore(self):
        return [(get_data, [{"input_file": self.input_file}])]

    def set_callbacks(self, app) -> None:
        @app.callback(
            Output(component_id=self.uuid("figure"), component_property="figure"),
            Input(
                component_id=self.uuid("radio_button_value"), component_property="value"
            ),
        )
        def _update_graph(value):
            fig = px.scatter(
                self.dataframe,
                x="X",
                y="Y",
                color=value,
                hover_name="Turbine label",
                hover_data=["X", "Y", "Average wake loss", "Average power"],
            )

            fig["layout"].update(self.theme.plotly_theme["layout"])

            return fig


@webvizstore
def get_data(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file)
