"""
ACAI Dashboard — app.py
Entry point. Run locally with: python dashboard/app.py
Deploy to PythonAnywhere by pointing WSGI config to `server`.

PythonAnywhere WSGI config:
    from dashboard.app import server as application
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, html, dcc, Input, Output
from layouts import overview, comparison

# ============================================================
# APP INIT
# ============================================================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# ============================================================
# ROOT LAYOUT
# ============================================================
app.layout = html.Div([

    html.Div([

        # header
        html.Div([
            html.H1("ACAI"),
            html.P("Automated Classroom Analysis & Insights"),
        ], id="header"),

        # tabs
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            children=[
                dcc.Tab(label="SESSION OVERVIEW", value="overview",
                        className="dash-tab", selected_className="dash-tab--selected"),
                dcc.Tab(label="COMPARISON", value="comparison",
                        className="dash-tab", selected_className="dash-tab--selected"),
            ],
            className="dash-tabs",
        ),

        html.Div(id="tab-content"),

    ], id="app-container"),

], style={"background": "#0d1b2a", "minHeight": "100vh"})


# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "overview":
        return overview.layout()
    if tab == "comparison":
        return comparison.layout()
    return html.Div("Tab not found.", style={"color": "#7a8fa6", "padding": "40px"})


# ============================================================
# REGISTER LAYOUT CALLBACKS
# ============================================================
overview.register_callbacks(app)
comparison.register_callbacks(app)


# ============================================================
# RUN LOCALLY
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)