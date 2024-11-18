# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
import warnings

import beer_utils

warnings.filterwarnings("ignore")




# Run this app with `python simple_app.py` and
# visit http://127.0.0.1:8051/ in your web browser.

app = Dash(
    __name__,
    # suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

# Data stuff first, app stuff below.
data = pd.read_excel("TripelTasting.xlsx")
bieren = pd.read_excel('Bieren.xlsx')

# Preprocess MS Forms output, merge with bier metadata
df_all = beer_utils.preprocess(data)
df_all = df_all.merge(bieren, left_on="BeerID", right_on="ID", how='left')
df_all["Beer"] = df_all.Brewery + ' ' + df_all.BeerName
df_all.drop(columns=['ID'])

# set some parameters
navbar_height = "4rem"
sidebar_width = "24rem"


# see https://plotly.com/python/px-arguments/ for more options
# NAVBAR STYLE
NAVBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "height": navbar_height,
    "padding": "2rem 1rem",
    "background-color": "#001324",
}

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": navbar_height,
    "left": 0,
    "bottom": 0,
    "width": sidebar_width,
    "background-color": "#f8f9fa",
    "padding": "4rem 1rem 2rem",
    "overflow": "scroll"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "margin-top": navbar_height,
    "margin-left": sidebar_width,
}


# Create content

sidebar = html.Div(
    [
        html.H1("Sensory panel"),
        html.Hr(),
        html.H5("Select a beer!"),
        dcc.Dropdown(
            id="Beer",  
            value=df_all.Beer.values[0],
            options=[{"label": x, "value": x} for x in df_all.Beer.unique()],
            multi=False,
            optionHeight=40,
            style={"margin-bottom": "50px"},
        ),
        html.Div(id="tts-text", style={'whiteSpace': 'pre-line'}),
        html.Hr(),
        html.H4("Which analyses are included?"),
        # Datum selecteren wordt alleen een ding als er meerdere datums zijn
        # html.H5('Selecteer datumrange van proeven'),
        # dcc.RangeSlider(
        #     min=pd.Timestamp(2024,1,1,0).timestamp(),
        #     max=pd.Timestamp(2025,1,1,0).timestamp(),
        #     value=[pd.Timestamp(2024, 1, 1, 0).timestamp(), pd.Timestamp(2025,1,1,0).timestamp()],
        #     id="Datumrange",
        # ),
        html.H5("Only those that indicated true-to-style"),
        daq.BooleanSwitch(id="stijlvast-switch"),
        html.H5("Only those that indicated fresh"),
        daq.BooleanSwitch(id="fresh-switch"),
        html.Br(),html.Br(),html.Br(),
        html.Hr(),
        html.Hr(),
        html.H4("Color"),
        html.Div(id='color-text', style={'whiteSpace': 'pre-line'}),
        html.Hr(),
        html.Hr(),
        html.H4("General Feedback"),
        html.Div(id='feedback-text', style={'whiteSpace': 'pre-line'}),
        html.Br(),
        # dcc.Graph(id="graph_kleur")
    ],
    style=SIDEBAR_STYLE,
    id="sidebar",
)

content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(dcc.Graph(id="graph_aroma")), width=6),
                dbc.Col(
                    [
                        daq.BooleanSwitch(id="taste-switch", label={
                            'label':"Basic tastes: Bitter/Sweet or Sour/pH",
                            'style':{'font_size':128}
                        }
                            ),
                        html.Div(dcc.Graph(id="graph_basissmaak"))
                    ],
                    width=6
                )
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div(dcc.Graph(id="graph_smaakstoffen")), width=6),
                dbc.Col(html.Div(dcc.Graph(id="graph_gauges")), width=6),
            ]
        ),
        

    ],
    style=CONTENT_STYLE,
    id="content",
)


# NAVBAR

LOGO = "https://static.vecteezy.com/system/resources/previews/025/062/290/non_2x/cheers-beer-glasses-isolated-illustration-cheers-beer-glass-clipart-toasting-beer-glasses-beer-glasses-beer-glasses-party-time-illustration-cheers-toast-clipart-cheers-silhouette-glasses-icon-free-png.png"
# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

NAVBAR = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.A(
                                href="https://craftbeer.marcelhaas.com",
                                children=[
                                    html.Img(
                                        src=LOGO, alt="Tasting Craft Beer", height="80px"
                                    )
                                ],
                            ),
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    html.H2("Tasting Craft Beer", style={"color": "#f8f9fa"}),
                    align="center",
                    width=9,
                ),
            ],
        ),
    ],
    color="dark",
    dark=True,
    # className="mb-5",
    style=NAVBAR_STYLE,
)

# HERE WE BUILT THE APP LAYOUT
app.layout = html.Div([content, sidebar, NAVBAR])  #

# any change to the input will call the update_figure function and return figures with updated data
@app.callback(
    Output("graph_aroma", "figure"),
    Output("graph_basissmaak", "figure"),
    Output("graph_gauges", "figure"),
    Output("graph_smaakstoffen", "figure"),
    Output("feedback-text", "children"),
    Output("color-text", "children"),
    Input("Beer", "value"),
    Input("stijlvast-switch", "on"),
    Input("fresh-switch", "on"),
    Input("taste-switch", "on")
)
def update_figure_table(Beer, stijlvastswitch, freshswitch, tasteswitch):
    # If no function selected, make it Docent 4
    if not Beer:
        Beer = df_all.Beer.unique([0])

    if stijlvastswitch:
        df1 = df_all[df_all["True-to-style"] == 'Yes']
    else: df1 = df_all.copy()

    if freshswitch:
        df = df1[df1["Fresh"] == "Fresh"]
    else: df = df1.copy()
    
    # Create profiles per beer as well (aggregated)
    perbier = beer_utils.bieranalyse(df)
    ditbier = perbier[perbier.Beer == Beer]

    # preps en plotten per plaatje. Code in bier.py
    df_aroma = beer_utils.prep_aroma(df, ditbier)
    fig_aroma = beer_utils.plot_aroma(df_aroma)

    df_basissmaak = perbier[['Beer', 'Sweet', 'Bitter', 'Sour', 'sweetstd', 'bitterstd', 'pH']]
    fig_basissmaak = beer_utils.plot_basissmaak(df_basissmaak, Beer, tasteswitch)

    fig_gauges = beer_utils.plot_gauge(ditbier, Beer)


    df_smaakstoffen = beer_utils.aroma_profiel(ditbier.Attributes.values[0]).sort_values('Votes', ascending=True)
    fig_smaakstoffen = beer_utils.plot_smaakstoffen(df_smaakstoffen, ditbier.n_tasters.values[0])

    feedback = ditbier.Feedback.values[0]

    colort = beer_utils.colordesc(ditbier.Color.values[0])
    colortext= f"Measured color: {ditbier.EBC.values[0]} EBC \nPerceived as {colort}"

    return (
        fig_aroma,
        fig_basissmaak,
        fig_gauges,
        fig_smaakstoffen,
        feedback,
        colortext
    )

@app.callback(
    Output("tts-text", "children"),
    Input("Beer", "value"),
)
def perc_true(Beer):
    ntastings = len(df_all[df_all.Beer == Beer])
    ntrue = len(df_all[((df_all.Beer == Beer) & (df_all["True-to-style"] == "Yes"))])
    return f"Percentage true-to-style: {ntrue/ntastings:.0%}"

if __name__ == "__main__":
    app.run(debug=True)#, port=8051)