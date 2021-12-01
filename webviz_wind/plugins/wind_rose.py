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
import ast


class WindRose(WebvizPluginABC):
    """Insert documentation of plugin here. Is used automatically by `webviz docs'
       This is a demo-version showing how to display
       * ** wind rose from a csv-file with Weibull parameters (consider extend to xlsx)
       * ** wind rose calculated from a time series
       * ** Wind speed time series (just for demo. Should probabily be developed further)

    Flexibility
       ** Time period (years) wind rose calculation is based on (time series input only)
       ** Resolution (degrees) wind rose calculation is based on (time series input only)
       ** Resolution (wind speed) wind rose calculation is based on

    Input parameters. Both input files are needed.
     - WindRose:
        weibull_input_file: ./weibull_table.csv  #Weibull parameters
        timeseries_input_file: ./timeseries.csv  #Time series in a very specific format
        year: 1999                               #Optional: The year chosen when webviz starts (timeseries)
        nof_years: 1                             #Optional: Number of years chosen when webviz starts (timeseries)

    """

    def __init__(
        self,
        app,
        webviz_settings,
        weibull_input_file: Path = None,
        timeseries_input_file: Path = None,
        year: int = None,
        nof_years: int = None,
    ) -> None:

        super().__init__()

        self.theme = webviz_settings.theme
        self.weibull_input_file = weibull_input_file
        self.timeseries_input_file = weibull_input_file

        # Defaults: Put into separat class
        if year is None:
            self.year = 1999
        else:
            self.year = year
        if nof_years is None:
            self.nof_years = 1
        else:
            self.nof_years = nof_years

        if weibull_input_file is not None:
            self.weibull_df = get_data(weibull_input_file)
        else:
            self.weibull_df = None

        if timeseries_input_file is not None:
            self.timeseries_df = get_data(timeseries_input_file)
            self.timeseries_df, self.available_years = self.format_timeseries(
                self.timeseries_df
            )  # make more flexible for more formats
            self.first_year = np.min(self.available_years)
            self.last_year = np.max(self.available_years)

        self.set_callbacks(app)

    # Should it return values or just manipulate self.
    def format_timeseries(
        self, df
    ):  # NOTE Date, year, Time etc should be defined in a separate class
        df.columns = ["Date", "Time", "ws", "wd"]

        # Make dates to real dates and extract the years
        df["Date"] = pd.to_datetime(df["Date"])  # Now it is an actual date
        df["year"] = df["Date"].dt.year  # Extract the year
        df.drop("Time", inplace=True, axis=1)  # Remove time of day. Not needed

        available_years = df["year"].unique()
        return df, available_years

    # Put into a separat _layout_windrose-file
    @property
    def layout(self):
        return wcc.FlexBox(
            children=[
                wcc.FlexColumn(
                    children=wcc.Frame(
                        style={"height": "90vh"},
                        children=[
                            wcc.Selectors(
                                label="Time series: Start year",
                                children=[
                                    wcc.Dropdown(
                                        id=self.uuid("start_year"),
                                        options=[
                                            {"label": i, "value": i}
                                            for i in self.available_years
                                        ],
                                        value=self.first_year,
                                    ),
                                ],
                            ),
                            wcc.Selectors(
                                label="Time series: Number of years",
                                children=[
                                    wcc.Dropdown(
                                        id=self.uuid("nof_years"),
                                        options=[
                                            {"label": i, "value": i}
                                            for i in range(1, 5)
                                        ],
                                        value=1,
                                    ),
                                ],
                            ),
                            wcc.Selectors(
                                label=f"Time series: Wind direction resolution (degrees)",
                                children=[
                                    wcc.Dropdown(
                                        id=self.uuid("wd_resolution"),
                                        options=[
                                            {"label": i, "value": i}
                                            for i in [30, 15, 10, 5]
                                        ],
                                        value=30,
                                    )
                                ],
                            ),
                            wcc.Selectors(
                                label="Wind speed resolution",
                                children=[
                                    wcc.RadioItems(
                                        id=self.uuid("radio_button_value"),
                                        options=[
                                            {
                                                "label": "0-5-10-15-20-40",
                                                "value": "[0,5,10,15,20,40]",
                                            },
                                            {
                                                "label": "0-3-6-9-12-15-18-21-40",
                                                "value": "[0,3,6,9,12,15,18,21,40]",  # Represent list as string. Use the ast-module to transferre to proper list
                                            },
                                        ],
                                        # value = 1 #[0,5,10,15,20,40], #How to make this default choise at start-up? Does not appear
                                        value="[0,5,10,15,20,40]",  # Must be num or string. Hence using the ast-module to transfere to list
                                    ),
                                ],
                            ),
                        ],
                    )
                ),
                # Placement and sized not optimal. Should be placed side-by-side
                wcc.FlexColumn(
                    flex=4,
                    children=[
                        wcc.Frame(
                            style={"height": "90vh"},
                            highlight=False,
                            color="white",
                            children=wcc.Graph(
                                style={"height": "85vh"},
                                id=self.uuid("ts_wind_rose_figure"),
                            ),
                        ),
                        wcc.Frame(
                            style={"height": "90vh"},
                            highlight=False,
                            color="white",
                            children=wcc.Graph(
                                style={"height": "85vh"},
                                id=self.uuid("weibull_wind_rose_figure"),
                            ),
                        ),
                        # Just here for demo. Probably not a natural place - unless enhenced.
                        wcc.Frame(
                            style={"height": "65vh"},
                            highlight=False,
                            color="white",
                            children=wcc.Graph(
                                style={"height": "55vh"},
                                id=self.uuid("wind_speed_figure"),
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
            Output(
                component_id=self.uuid("ts_wind_rose_figure"),
                component_property="figure",
            ),
            [
                Input(
                    component_id=self.uuid("wd_resolution"), component_property="value"
                ),
                Input(
                    component_id=self.uuid("radio_button_value"),
                    component_property="value",
                ),
                Input(component_id=self.uuid("start_year"), component_property="value"),
                Input(component_id=self.uuid("nof_years"), component_property="value"),
            ],
        )
        def _update_timeseries_wind_rose(delta_wd, intervals, start_year, nof_years):
            intervals = ast.literal_eval(intervals)  # Going from string to proper list
            df = extract_years(
                self.timeseries_df, start_year, nof_years, "year"
            )  # year should be defined in a separate class
            df = wind_rose_from_timeseries(df, delta_wd, intervals)

            fig = px.bar_polar(
                df,
                r="freq",
                theta="wd",
                color="ws",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                title=f"Wind rose from time series: {nof_years} years starting from {start_year} ",
            )

            fig["layout"].update(self.theme.plotly_theme["layout"])
            return fig

        @app.callback(
            Output(
                component_id=self.uuid("wind_speed_figure"), component_property="figure"
            ),
            [
                Input(component_id=self.uuid("start_year"), component_property="value"),
                Input(component_id=self.uuid("nof_years"), component_property="value"),
            ],
        )
        def _update_timeseries(start_year, nof_years):
            df = extract_years(
                self.timeseries_df, start_year, nof_years, "year"
            )  # year should be defined in a separate class
            fig = px.line(df, x="timestep", y="ws", title="Wind speed time series")
            fig["layout"].update(self.theme.plotly_theme["layout"])
            return fig

        @app.callback(
            Output(
                component_id=self.uuid("weibull_wind_rose_figure"),
                component_property="figure",
            ),
            [
                Input(
                    component_id=self.uuid("radio_button_value"),
                    component_property="value",
                ),
            ],
        )
        def _update_weibull_wind_rose(intervals: str):

            intervals = ast.literal_eval(intervals)  # Going from string to proper list
            df = wind_rose_from_weibull(self.weibull_df, intervals)
            fig = px.bar_polar(
                df,
                r="freq",
                theta="wd",
                color="ws",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                title="Wind rose from weibull parameters",
            )

            fig["layout"].update(self.theme.plotly_theme["layout"])
            return fig

        @app.callback(
            Output(component_id=self.uuid("nof_years"), component_property="options"),
            [
                Input(component_id=self.uuid("start_year"), component_property="value"),
            ],
        )
        def _update_nof_years_to_choose(start_year: int):
            return [
                {"label": i, "value": i}
                for i in range(1, self.last_year - start_year + 1)
            ]

        @app.callback(
            Output(component_id=self.uuid("start_year"), component_property="options"),
            [
                Input(component_id=self.uuid("nof_years"), component_property="value"),
            ],
        )
        def _update_years_to_choose(nof_years: int):
            return [
                {"label": i, "value": i}
                for i in range(self.first_year, self.last_year - nof_years + 1)
            ]


@webvizstore
def get_data(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file)


@webvizstore
def read_timeseries(fn: Path) -> pd.DataFrame:
    # Get the data
    df = pd.read_csv(fn, names=["Date", "Time", "ws", "wd"], skiprows=1)
    # Make dates to real dates and extract the years
    df["Date"] = pd.to_datetime(df["Date"])  # Now it is an actual date
    df["year"] = df["Date"].dt.year  # Extract the year
    df.drop("Time", inplace=True, axis=1)  # Remove time of day. Not needed

    return df


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def filter_dataframe(df: pd.DataFrame, column: str, filter_value: int) -> pd.DataFrame:

    return df.loc[df[column] == filter_value]


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def extract_years(
    df: pd.DataFrame, start_year: int, nof_years: int, date_column: str
) -> pd.DataFrame:
    """Input: A DataFrame on the format. Additional columns are allowed
    year ws  wd
    1999 8.2 231
    1999 5.3 201
    """
    try:
        years = list(
            range(start_year, start_year + nof_years)
        )  # Years we are interested in
        print(f"Extracting the years: {years}")
        df = df.loc[df[date_column].isin(years)]  # Extract only the years of interest
        # print(df)
        df["timestep"] = [
            x * 1.0 / 24.0 for x in range(1, len(df) + 1)
        ]  # Number of days
        # print(df)
    except:
        raise Exception(
            f"Problems reading from year {start_year} to {start_year + nof_years}. Check your data"
        )

    return df


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def wind_rose_from_weibull(df: pd.DataFrame, intervals: list) -> pd.DataFrame:
    """Transfers a table with weibull parameters for each wind direction to proper format for plotting of wind roses
     It must be a csv-file on this format:
     wd,freq,a,k
     0,0.043151,8.350781,1.845081
     30,0.055251,8.944269,2.269696
     60,0.049886,8.418256,2.129145
     90,0.061758,9.808952,2.187215

     Output is a dataframe on this format
     wd   ws   freq
     0    0-5  0.013881
     0   5-10  0.018570
     0  10-15  0.008433
     0  15-20  0.001978
     0  20-40  0.000288
    30    0-5  0.012953
    30   5-10  0.027062
    30  10-15  0.013058
    30  15-20  0.002067

    """
    wlist = []
    wd = df["wd"].values

    for wd in df["wd"].values:  # Loop over each of the wind directions
        a, k, freq = df[df["wd"] == wd][["a", "k", "freq"]].to_numpy().flatten()
        for i in range(1, len(intervals)):  # Loop each wind speed interval
            name = (
                f"{intervals[i-1]}-{intervals[i]}"  # Name of that interval for plotting
            )
            wfrec = weibull_cdf(intervals[i], a, k) - weibull_cdf(
                intervals[i - 1], a, k
            )
            wlist.append([wd, name, wfrec * freq])

    # Make a dataframe out of the list of list containing
    wdf = pd.DataFrame(wlist, columns=["wd", "ws", "freq"])

    return wdf


def weibull_cdf(x, a, k):
    return 1 - np.exp(-((x / a) ** k))


@CACHE.memoize(timeout=CACHE.TIMEOUT)
def wind_rose_from_timeseries(
    df: pd.DataFrame, wd_delta: int, intervals: list
) -> pd.DataFrame:
    """Takes a dataframe on the format  (only ws, wd is needed)
     Date        ws     wd  year
    1999-01-01  12.77  156.9  1999
    1999-01-01  12.90  158.3  1999
    1999-01-01  12.66  160.5  1999
    1999-01-01  12.57  162.9  1999
    1999-01-01  12.74  168.5  1999

    and calculates the input for wind rose on this format
    ws   wd  freq
    0-5    0   0.014
    0-5   30   0.013
    0-5   60   0.012
    0-5   90   0.015

    Parameters:
    wd_delta: Spacing between ead wind direction to calculate.
              Must divide 360
    intervals: The intervals for wind speed used for calculation
    """
    if 360 % wd_delta != 0:
        print(
            f"ERROR: Delta wind direction must devide 360. You are trying to use {wd_delta}"
        )
        raise Exception

    wd = df["wd"].values
    ws = df["ws"].values
    tot0 = len(ws)
    wds = list(range(0, 360, wd_delta))
    wd_width = wd_delta / 2
    wd[
        wd > 360 - wd_width
    ] -= 360  # Turn angles in the last interval before 360 to negative-angles to make calculations later easier.

    # calculate windrose from timeseries
    # A dictionary is easier to work with when collecting the number of samples in each bin
    wdict = {}
    for i in wds:
        wdict[i] = {}
    #        for j in range(1, len(intervals)):
    #            wdict[i][f'{intervals[j-1]}-{intervals[j]}'] = 0

    tot = 0  # Total number of observations that goes into the wind rose calculations
    for (
        wdi
    ) in (
        wds
    ):  # For each wind direction and each wind speed interval: Calculate the number of observations
        # Find the interval of wind directions
        low = wdi - wd_width
        high = wdi + wd_width

        # The positions of the elements with wind directions in the wanted interval
        try:
            selection = np.where((wd > low) & (wd <= high))
            ws_sub = ws[
                selection
            ]  # Subset of wind speeds with the wanted wind directions
        except:
            ws_sub = []

        for i in range(1, len(intervals)):
            name = f"{intervals[i-1]}-{intervals[i]}"
            try:
                selection = np.where(
                    (ws_sub > intervals[i - 1]) & (ws_sub <= intervals[i])
                )  # Get the index subset
                nobs = np.shape(selection)[1]  # Get the number of observations
                wdict[wdi][name] = nobs
                tot += nobs
            except:
                wdict[wdi][name] = 0

    # Make a dataframe of the dictionary and reformat
    wdf = pd.DataFrame.from_dict(wdict)
    wdf = wdf.stack().reset_index()
    wdf = wdf.rename(columns={"level_0": "ws", "level_1": "wd", 0: "freq"})

    # Find frequency by dividing number of data points with the total number of data.
    if tot0 != tot:
        print(
            f"WARNING: The total number of observations {tot0} is not the same as in the rose diagram {tot}"
        )
    wdf["freq"] = wdf["freq"].div(tot)

    return wdf
