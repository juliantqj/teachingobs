"""
ACAI Dashboard — layouts/comparison.py
Tab 2: Cross-session comparison.
"""

import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D

# ============================================================
# COLOURS
# ============================================================
BG       = "#0d1b2a"
SURFACE  = "#112236"
SURFACE2 = "#162b42"
ACCENT   = "#00b4d8"
TEXT     = "#e8edf2"
MUTED    = "#7a8fa6"
BORDER   = "#1e3a52"
WARNING  = "#f39c12"

SESSION_PALETTE = [
    "#00b4d8", "#2ecc71", "#f39c12", "#e74c3c",
    "#9b59b6", "#1abc9c", "#e67e22", "#3498db",
]

BLOOMS_ORDER = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]
AOI_ORDER    = ["Slides", "Students", "Computer", "Whiteboard"]


# ============================================================
# HELPERS
# ============================================================
def _empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color=MUTED, size=13))
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin=dict(l=0, r=0, t=0, b=0), height=240)
    return fig


def _base_layout(fig, height=320):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Arial"),
        height=height,
        margin=dict(l=50, r=20, t=20, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                    font=dict(color=TEXT, size=11)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
        barmode="group",
    )
    return fig


def _short_label(session_name: str) -> str:
    """Shorten session name for chart axes."""
    meta = D.parse_session_name(session_name)
    date = meta["date"]
    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
    return f"{meta['filename']}\n{date_fmt}"


# ============================================================
# CHARTS
# ============================================================
def make_teaching_style_comparison(sessions_data: dict):
    if not sessions_data:
        return _empty_fig()

    fig = go.Figure()
    styles = ["Active Gesturing", "Passive Teaching"]
    colours = {
        "Active Gesturing": ACCENT,
        "Passive Teaching": "#4e6d8c",
    }

    for style in styles:
        x_labels, y_vals = [], []
        for session_name, d in sessions_data.items():
            summary = D.get_teaching_style_summary(d["teaching_style"])
            x_labels.append(_short_label(session_name))
            y_vals.append(summary.get(style, 0))
        fig.add_trace(go.Bar(
            name=style, x=x_labels, y=y_vals,
            marker_color=colours[style],
            hovertemplate=f"{style}: %{{y}}%<extra></extra>",
        ))

    _base_layout(fig)
    fig.update_yaxes(title="% of session", range=[0, 100])
    return fig


def make_aoi_comparison(sessions_data: dict):
    if not sessions_data:
        return _empty_fig()

    fig = go.Figure()
    aoi_colours = {
        "Slides":     "#00b4d8",
        "Students":   "#2ecc71",
        "Computer":   "#f39c12",
        "Whiteboard": "#e74c3c",
    }

    for aoi in AOI_ORDER:
        x_labels, y_vals = [], []
        for session_name, d in sessions_data.items():
            summary = D.get_aoi_time_summary(d["teaching_style"])
            x_labels.append(_short_label(session_name))
            y_vals.append(summary.get(aoi, 0))
        fig.add_trace(go.Bar(
            name=aoi, x=x_labels, y=y_vals,
            marker_color=aoi_colours[aoi],
            hovertemplate=f"{aoi}: %{{y}}%<extra></extra>",
        ))

    _base_layout(fig)
    fig.update_yaxes(title="% of session")
    return fig


def make_blooms_comparison(sessions_data: dict):
    if not sessions_data:
        return _empty_fig()

    blooms_colours = {
        "Remember":   "#4e6d8c",
        "Understand": "#3a8fb5",
        "Apply":      "#00b4d8",
        "Analyse":    "#2ecc71",
        "Evaluate":   "#f39c12",
        "Create":     "#e74c3c",
    }

    fig = go.Figure()
    for level in BLOOMS_ORDER:
        x_labels, y_vals = [], []
        for session_name, d in sessions_data.items():
            summary = D.get_blooms_summary(d["blooms"])
            x_labels.append(_short_label(session_name))
            y_vals.append(summary.get(level, 0))
        fig.add_trace(go.Bar(
            name=level, x=x_labels, y=y_vals,
            marker_color=blooms_colours[level],
            hovertemplate=f"{level}: %{{y}}%<extra></extra>",
        ))

    _base_layout(fig)
    fig.update_yaxes(title="% of segments")
    return fig


def make_speech_metrics_comparison(sessions_data: dict):
    if not sessions_data:
        return _empty_fig()

    labels, rates, pauses, pause_counts = [], [], [], []
    for session_name, d in sessions_data.items():
        acoustic = D.get_acoustic_summary(d["report"])
        labels.append(_short_label(session_name))
        rates.append(acoustic.get("mean_speech_rate_wps", 0))
        pauses.append(acoustic.get("mean_pause_duration_sec", 0))
        pause_counts.append(acoustic.get("pause_count", 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mean speech rate (wps)", x=labels, y=rates,
        marker_color=ACCENT,
        hovertemplate="Rate: %{y:.2f} wps<extra></extra>",
    ))

    _base_layout(fig, height=260)
    fig.update_yaxes(title="Words per second")
    return fig


def make_pause_comparison(sessions_data: dict):
    if not sessions_data:
        return _empty_fig()

    labels, pauses = [], []
    for session_name, d in sessions_data.items():
        acoustic = D.get_acoustic_summary(d["report"])
        labels.append(_short_label(session_name))
        pauses.append(acoustic.get("mean_pause_duration_sec", 0))

    fig = go.Figure(go.Bar(
        x=labels, y=pauses,
        marker_color=WARNING,
        hovertemplate="Mean pause: %{y:.2f}s<extra></extra>",
    ))
    _base_layout(fig, height=260)
    fig.update_yaxes(title="Mean pause duration (s)")
    return fig


def make_radar_chart(sessions_data: dict):
    """Speech style radar — one polygon per session."""
    if not sessions_data:
        return _empty_fig()

    categories = [
        "Speech Rate", "Pause Frequency",
        "Question Count", "Clarity", "Hedge Frequency",
    ]

    def _normalise(val, min_v, max_v):
        if max_v == min_v:
            return 0.5
        return max(0, min(1, (val - min_v) / (max_v - min_v)))

    clarity_map   = {"Low": 0, "Medium": 0.5, "High": 1}
    freq_map      = {"Low": 0, "Medium": 0.5, "High": 1}

    all_rates     = []
    all_pauses    = []
    all_questions = []

    session_metrics = {}
    for session_name, d in sessions_data.items():
        acoustic   = D.get_acoustic_summary(d["report"])
        linguistic = D.get_linguistic_summary(d["report"])
        m = {
            "rate":     acoustic.get("mean_speech_rate_wps", 0),
            "pauses":   acoustic.get("pause_count", 0),
            "questions": linguistic.get("question_count_estimate", 0),
            "clarity":  clarity_map.get(linguistic.get("clarity_assessment", "Medium"), 0.5),
            "hedge":    freq_map.get(linguistic.get("hedge_frequency", "Medium"), 0.5),
        }
        session_metrics[session_name] = m
        all_rates.append(m["rate"])
        all_pauses.append(m["pauses"])
        all_questions.append(m["questions"])

    r_min, r_max = min(all_rates), max(all_rates)
    p_min, p_max = min(all_pauses), max(all_pauses)
    q_min, q_max = min(all_questions), max(all_questions)

    fig = go.Figure()
    for i, (session_name, m) in enumerate(session_metrics.items()):
        colour = SESSION_PALETTE[i % len(SESSION_PALETTE)]
        values = [
            _normalise(m["rate"],     r_min, r_max),
            _normalise(m["pauses"],   p_min, p_max),
            _normalise(m["questions"], q_min, q_max),
            m["clarity"],
            m["hedge"],
        ]
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself", name=D.parse_session_name(session_name)["filename"],
            line=dict(color=colour),
            fillcolor=f"rgba({int(colour[1:3], 16)}, {int(colour[3:5], 16)}, {int(colour[5:7], 16)}, 0.15)",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=BORDER, linecolor=BORDER,
                            tickfont=dict(color=MUTED, size=9)),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(color=TEXT, size=11)),
        ),
        paper_bgcolor=SURFACE, font=dict(color=TEXT),
        height=360,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        showlegend=True,
    )
    return fig


# ============================================================
# LAYOUT
# ============================================================
def layout():
    sessions = D.list_sessions()
    options = [{"label": s, "value": s} for s in sessions]

    return html.Div([

        # ---- selector bar ----
        html.Div([
            html.Label("SELECT SESSIONS TO COMPARE"),
            dcc.Dropdown(
                id="cmp-session-dropdown",
                options=options,
                value=sessions[:2] if len(sessions) >= 2 else sessions,
                multi=True,
                style={"minWidth": "480px"},
            ),
        ], className="selector-bar"),

        html.Div(id="cmp-content"),
    ])


# ============================================================
# CALLBACKS
# ============================================================
def register_callbacks(app):

    @app.callback(
        Output("cmp-content", "children"),
        Input("cmp-session-dropdown", "value"),
    )
    def update_comparison(selected_sessions):
        print("Selected sessions:", selected_sessions)
        if not selected_sessions or len(selected_sessions) < 2:
            return html.Div(
                "Select at least 2 sessions to compare.",
                className="no-data"
            )

        # diagnostic prints
        for s in selected_sessions:
            d = D.load_session_files(s)
            print(f"{s} — teaching_style rows: {len(d['teaching_style'])}")
            print(f"{s} — blooms rows: {len(d['blooms'])}")
            print(f"{s} — report keys: {list(d['report'].keys())}")

        # load all selected sessions
        sessions_data = {}
        for s in selected_sessions:
            sessions_data[s] = D.load_session_files(s)

        return html.Div([

            # teaching style
            html.Div([
                html.Div("TEACHING STYLE COMPARISON", className="card-title"),
                dcc.Graph(figure=make_teaching_style_comparison(sessions_data),
                          config={"displayModeBar": False}),
            ], className="card"),

            # AOI
            html.Div([
                html.Div("TIME IN REGION COMPARISON", className="card-title"),
                dcc.Graph(figure=make_aoi_comparison(sessions_data),
                          config={"displayModeBar": False}),
            ], className="card"),

            # Bloom's
            html.Div([
                html.Div("BLOOM'S TAXONOMY COMPARISON", className="card-title"),
                dcc.Graph(figure=make_blooms_comparison(sessions_data),
                          config={"displayModeBar": False}),
            ], className="card"),

            # speech metrics side by side
            html.Div([
                html.Div([
                    html.Div("MEAN SPEECH RATE", className="card-title"),
                    dcc.Graph(figure=make_speech_metrics_comparison(sessions_data),
                              config={"displayModeBar": False}),
                ], className="card"),
                html.Div([
                    html.Div("MEAN PAUSE DURATION", className="card-title"),
                    dcc.Graph(figure=make_pause_comparison(sessions_data),
                              config={"displayModeBar": False}),
                ], className="card"),
            ], className="two-col"),

            # radar
            html.Div([
                html.Div("SPEECH STYLE RADAR", className="card-title"),
                html.P(
                    "Normalised axes — values show relative position across selected sessions, not absolute scores.",
                    className="prose-muted",
                    style={"marginBottom": "12px"},
                ),
                dcc.Graph(figure=make_radar_chart(sessions_data),
                          config={"displayModeBar": False}),
            ], className="card"),

        ])