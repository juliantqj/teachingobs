"""
ACAI Dashboard — layouts/overview.py
Tab 1: Single session overview.
"""

import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, callback, Input, Output
import pandas as pd
import os

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import data as D

# ============================================================
# COLOUR PALETTE
# ============================================================
BG        = "#0d1b2a"
SURFACE   = "#112236"
SURFACE2  = "#162b42"
ACCENT    = "#00b4d8"
TEXT      = "#e8edf2"
MUTED     = "#7a8fa6"
BORDER    = "#1e3a52"
SUCCESS   = "#2ecc71"
WARNING   = "#f39c12"

BLOOMS_COLOURS = {
    "Remember":   "#4e6d8c",
    "Understand": "#3a8fb5",
    "Apply":      "#00b4d8",
    "Analyse":    "#2ecc71",
    "Evaluate":   "#f39c12",
    "Create":     "#e74c3c",
    "Unclassified": "#4a4a6a",
}

AOI_COLOURS = {
    "Slides":     "#00b4d8",
    "Students":   "#2ecc71",
    "Computer":   "#f39c12",
    "Whiteboard": "#e74c3c",
}

STYLE_COLOURS = {
    "Active Gesturing": ACCENT,
    "Passive Teaching": "#4e6d8c",
    "No Pose Detected": "#2a3a4a",
}

# ============================================================
# CHART HELPERS
# ============================================================
def _empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color=MUTED, size=13))
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin=dict(l=0, r=0, t=0, b=0), height=200)
    return fig


def _base_layout(fig, height=300, margin=None):
    m = margin or dict(l=40, r=20, t=20, b=40)
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Arial"),
        height=height, margin=m,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                    font=dict(color=TEXT, size=11)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED)),
    )
    return fig


# ============================================================
# INDIVIDUAL CHARTS
# ============================================================
def make_teaching_style_timeline(df: pd.DataFrame):
    if df.empty or "teaching_style" not in df.columns:
        return _empty_fig()

    fps_estimate = 30
    df = df.copy()
    df["time_sec"] = df["frame"] / fps_estimate if "frame" in df.columns else df.index / fps_estimate

    fig = go.Figure()
    for style, colour in STYLE_COLOURS.items():
        mask = df["teaching_style"] == style
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "time_sec"], y=[style] * mask.sum(),
            mode="markers", name=style,
            marker=dict(color=colour, size=3, symbol="square"),
        ))

    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT), height=180,
        margin=dict(l=120, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        xaxis=dict(title="Time (seconds)", gridcolor=BORDER,
                   linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT)),
        showlegend=False,
    )
    return fig


def make_aoi_pie(df: pd.DataFrame):
    summary = D.get_aoi_time_summary(df)
    if not summary:
        return _empty_fig()

    labels = list(summary.keys())
    values = list(summary.values())
    colours = [AOI_COLOURS.get(l, ACCENT) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colours, line=dict(color=BG, width=2)),
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT), height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        showlegend=True,
    )
    return fig


def make_cog_heatmap(df: pd.DataFrame, first_frame_path: str = None,
                     frame_w: int = None, frame_h: int = None):
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return _empty_fig()

    d = df.dropna(subset=["x", "y"])
    if d.empty:
        return _empty_fig()

    fig = go.Figure()

    # ---- background: first video frame ----
    if first_frame_path and os.path.exists(first_frame_path):
        import base64
        with open(first_frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        img_w = frame_w or int(d["x"].max() * 1.1)
        img_h = frame_h or int(d["y"].max() * 1.1)

        fig.add_layout_image(
            source=f"data:image/jpeg;base64,{img_b64}",
            x=0, y=0,
            xref="x", yref="y",
            sizex=img_w, sizey=img_h,
            sizing="stretch",
            opacity=0.4,
            layer="below",
        )

    # ---- heatmap contour overlay ----
    fig.add_trace(go.Histogram2dContour(
        x=d["x"], y=d["y"],
        colorscale=[[0, "rgba(0,0,0,0)"], [0.3, "rgba(0,119,168,0.4)"],
                    [1, "rgba(0,180,216,0.85)"]],
        contours=dict(showlabels=False),
        line=dict(width=0),
        ncontours=20,
        showscale=False,
    ))

    x_range = [0, frame_w or d["x"].max() * 1.05]
    y_range = [frame_h or d["y"].max() * 1.05, 0]   # inverted: image origin top-left

    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), height=340,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(range=x_range, showgrid=False, showticklabels=False,
                   linecolor=BORDER, zeroline=False),
        yaxis=dict(range=y_range, showgrid=False, showticklabels=False,
                   linecolor=BORDER, zeroline=False),
    )
    return fig


def make_blooms_bar(df: pd.DataFrame):
    summary = D.get_blooms_summary(df)
    if not summary:
        return _empty_fig()

    order = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create", "Unclassified"]
    labels = [l for l in order if l in summary]
    values = [summary[l] for l in labels]
    colours = [BLOOMS_COLOURS.get(l, ACCENT) for l in labels]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colours),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))
    _base_layout(fig, height=260, margin=dict(l=100, r=60, t=10, b=40))
    fig.update_xaxes(title="% of segments", range=[0, max(values) * 1.2])
    return fig


def make_blooms_timeline(df: pd.DataFrame):
    if df.empty or "blooms_level" not in df.columns:
        return _empty_fig()

    order = ["Remember", "Understand", "Apply", "Analyse", "Evaluate", "Create"]
    level_map = {l: i for i, l in enumerate(order)}
    d = df.copy()
    d["level_num"] = d["blooms_level"].map(level_map)
    d = d.dropna(subset=["level_num"])

    start_col = "start" if "start" in d.columns else None

    fig = go.Figure(go.Scatter(
        x=d[start_col] if start_col else d.index,
        y=d["level_num"],
        mode="markers+lines",
        marker=dict(
            color=[BLOOMS_COLOURS.get(l, ACCENT) for l in d["blooms_level"]],
            size=7,
        ),
        line=dict(color=BORDER, width=1),
        text=d["blooms_level"],
        hovertemplate="%{text}<extra></extra>",
    ))
    _base_layout(fig, height=220, margin=dict(l=90, r=20, t=10, b=40))
    fig.update_yaxes(
        tickvals=list(range(len(order))),
        ticktext=order,
        title="",
    )
    fig.update_xaxes(title="Time (seconds)" if start_col else "Segment")
    return fig


def make_speech_rate_chart(df: pd.DataFrame):
    if df.empty or "speech_rate_wps" not in df.columns:
        return _empty_fig()

    x_col = "start" if "start" in df.columns else df.index
    mean_rate = df["speech_rate_wps"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col] if isinstance(x_col, str) else x_col,
        y=df["speech_rate_wps"],
        mode="lines",
        line=dict(color=ACCENT, width=2),
        name="Speech rate",
        hovertemplate="Time: %{x}s<br>Rate: %{y:.2f} wps<extra></extra>",
    ))
    fig.add_hline(y=mean_rate, line_dash="dash",
                  line_color=WARNING, opacity=0.7,
                  annotation_text=f"Mean: {mean_rate:.2f}",
                  annotation_font_color=WARNING)

    _base_layout(fig, height=220)
    fig.update_xaxes(title="Time (seconds)")
    fig.update_yaxes(title="Words/sec")
    return fig


def make_pause_chart(df: pd.DataFrame):
    if df.empty or "pause_after_sec" not in df.columns:
        return _empty_fig()

    d = df[df["pause_after_sec"] > 0].copy()
    if d.empty:
        return _empty_fig("No pauses detected above threshold")

    x_col = "start" if "start" in d.columns else d.index

    fig = go.Figure(go.Bar(
        x=d[x_col] if isinstance(x_col, str) else x_col,
        y=d["pause_after_sec"],
        marker=dict(color=ACCENT, opacity=0.8),
        hovertemplate="Time: %{x}s<br>Pause: %{y:.2f}s<extra></extra>",
    ))
    _base_layout(fig, height=200)
    fig.update_xaxes(title="Time (seconds)")
    fig.update_yaxes(title="Pause duration (s)")
    return fig


# ============================================================
# LAYOUT
# ============================================================
def layout():
    sessions = D.list_sessions()
    options = [{"label": s, "value": s} for s in sessions]
    default = sessions[0] if sessions else None

    return html.Div([

        # ---- selector bar ----
        html.Div([
            html.Label("SESSION"),
            dcc.Dropdown(
                id="ov-session-dropdown",
                options=options,
                value=default,
                clearable=False,
                style={"minWidth": "320px"},
            ),
        ], className="selector-bar"),

        # ---- content (populated by callback) ----
        html.Div(id="ov-content"),

    ])


# ============================================================
# CALLBACKS
# ============================================================
def register_callbacks(app):

    @app.callback(
        Output("ov-content", "children"),
        Input("ov-session-dropdown", "value"),
    )
    def update_overview(session_name):
        if not session_name:
            return html.Div("Select a session to begin.", className="no-data")

        d = D.load_session_files(session_name)
        ts_df       = d["teaching_style"]
        cog_df      = d["cog"]
        blooms_df   = d["blooms"]
        acoustic_df = d["acoustic"]
        report      = d["report"]
        session_meta    = d.get("session_meta", {})
        first_frame_path = d.get("first_frame_path")

        content_summary  = D.get_content_summary(report)
        linguistic       = D.get_linguistic_summary(report)
        acoustic_metrics = D.get_acoustic_summary(report)
        style_summary    = D.get_teaching_style_summary(ts_df)

        # ---- real duration from session_meta ----
        duration_sec = session_meta.get("duration_sec")
        if duration_sec:
            mins = int(duration_sec // 60)
            secs = int(duration_sec % 60)
            duration_str = f"{mins}m {secs}s"
        else:
            duration_str = content_summary.get("lecture_duration_estimate", "—")

        # ---- metadata card ----
        meta = D.parse_session_name(session_name)
        date_str = f"{meta['date'][:4]}-{meta['date'][4:6]}-{meta['date'][6:]}" if len(meta.get("date","")) == 8 else meta.get("date", "—")

        metadata_card = html.Div([
            html.Div("SESSION DETAILS", className="card-title"),
            html.Div([
                html.Div([html.Div(meta["filename"], className="metric-value"),
                          html.Div("FILE", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(date_str, className="metric-value"),
                          html.Div("DATE", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(duration_str, className="metric-value"),
                          html.Div("DURATION", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(linguistic.get("teacher_talk_style", "—"), className="metric-value"),
                          html.Div("TALK STYLE", className="metric-label")], className="metric-tile"),
            ], className="metric-grid"),
        ], className="card")

        # ---- content summary card ----
        summary_card = html.Div([
            html.Div("LECTURE SUMMARY", className="card-title"),
            html.P(content_summary.get("content_summary", "No summary available."), className="prose"),
            html.Br(),
            html.Div("KEY TOPICS", className="card-title"),
            html.Div([
                html.Span(t, className="tag tag-accent")
                for t in content_summary.get("key_topics", [])
            ], className="tag-list"),
        ], className="card")

        # ---- teaching style section ----
        active_pct = style_summary.get("Active Gesturing", 0)
        passive_pct = style_summary.get("Passive Teaching", 0)

        teaching_card = html.Div([
            html.Div("TEACHING STYLE", className="card-title"),
            html.Div([
                html.Div([html.Div(f"{active_pct}%", className="metric-value"),
                          html.Div("ACTIVE GESTURING", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(f"{passive_pct}%", className="metric-value"),
                          html.Div("PASSIVE TEACHING", className="metric-label")], className="metric-tile"),
            ], className="metric-grid"),
            dcc.Graph(figure=make_teaching_style_timeline(ts_df),
                      config={"displayModeBar": False}),
        ], className="card")

        # ---- AOI + heatmap ----
        spatial_card = html.Div([
            html.Div("SPATIAL ANALYSIS", className="card-title"),
            html.Div([
                html.Div([
                    html.Div("TIME IN REGION", className="card-title"),
                    dcc.Graph(figure=make_aoi_pie(ts_df),
                              config={"displayModeBar": False}),
                ], className="card"),
                html.Div([
                    html.Div("CLASSROOM MOVEMENT HEATMAP", className="card-title"),
                    dcc.Graph(figure=make_cog_heatmap(
                                  cog_df,
                                  first_frame_path=first_frame_path,
                                  frame_w=session_meta.get("width"),
                                  frame_h=session_meta.get("height"),
                              ),
                              config={"displayModeBar": False}),
                ], className="card"),
            ], className="two-col"),
        ])

        # ---- blooms section ----
        blooms_card = html.Div([
            html.Div("BLOOM'S TAXONOMY", className="card-title"),
            html.Div([
                html.Div([
                    html.Div("DISTRIBUTION", className="card-title"),
                    dcc.Graph(figure=make_blooms_bar(blooms_df),
                              config={"displayModeBar": False}),
                ], className="card"),
                html.Div([
                    html.Div("COGNITIVE LEVEL OVER TIME", className="card-title"),
                    dcc.Graph(figure=make_blooms_timeline(blooms_df),
                              config={"displayModeBar": False}),
                ], className="card"),
            ], className="two-col"),
        ])

        # ---- speech section ----
        mean_rate = acoustic_metrics.get("mean_speech_rate_wps", 0)
        mean_pause = acoustic_metrics.get("mean_pause_duration_sec", 0)
        pause_count = acoustic_metrics.get("pause_count", 0)
        pitch_mean = acoustic_metrics.get("pitch_mean_hz", 0)

        speech_card = html.Div([
            html.Div("SPEECH & PROSODY", className="card-title"),
            html.Div([
                html.Div([html.Div(f"{mean_rate:.2f}", className="metric-value"),
                          html.Div("MEAN SPEECH RATE (wps)", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(f"{mean_pause:.2f}s", className="metric-value"),
                          html.Div("MEAN PAUSE DURATION", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(str(pause_count), className="metric-value"),
                          html.Div("TOTAL PAUSES", className="metric-label")], className="metric-tile"),
                html.Div([html.Div(f"{pitch_mean:.0f} Hz", className="metric-value"),
                          html.Div("MEAN PITCH", className="metric-label")], className="metric-tile"),
            ], className="metric-grid"),
            html.Div([
                html.Div([
                    html.Div("SPEECH RATE OVER TIME", className="card-title"),
                    dcc.Graph(figure=make_speech_rate_chart(acoustic_df),
                              config={"displayModeBar": False}),
                ], className="card"),
                html.Div([
                    html.Div("PAUSE DISTRIBUTION", className="card-title"),
                    dcc.Graph(figure=make_pause_chart(acoustic_df),
                              config={"displayModeBar": False}),
                ], className="card"),
            ], className="two-col"),
        ], className="card")

        # ---- linguistic card ----
        clarity = linguistic.get("clarity_assessment", "—")
        clarity_colour = {"High": SUCCESS, "Medium": WARNING, "Low": "#e74c3c"}.get(clarity, MUTED)

        linguistic_card = html.Div([
            html.Div("LINGUISTIC ANALYSIS", className="card-title"),
            html.Div([
                html.Div([
                    html.Div(clarity, className="metric-value",
                             style={"color": clarity_colour}),
                    html.Div("CLARITY", className="metric-label"),
                ], className="metric-tile"),
                html.Div([
                    html.Div(str(linguistic.get("question_count_estimate", "—")), className="metric-value"),
                    html.Div("QUESTIONS ASKED", className="metric-label"),
                ], className="metric-tile"),
                html.Div([
                    html.Div(linguistic.get("filler_word_frequency", "—"), className="metric-value"),
                    html.Div("FILLER WORD FREQ.", className="metric-label"),
                ], className="metric-tile"),
                html.Div([
                    html.Div(linguistic.get("hedge_frequency", "—"), className="metric-value"),
                    html.Div("HEDGE LANGUAGE FREQ.", className="metric-label"),
                ], className="metric-tile"),
            ], className="metric-grid"),
            html.P(linguistic.get("clarity_notes", ""), className="prose-muted"),
            html.Br(),
            html.Div([
                html.Div([
                    html.Div("STRENGTHS", className="card-title"),
                    html.Div([
                        html.Span(s, className="tag tag-accent")
                        for s in linguistic.get("communication_strengths", [])
                    ], className="tag-list"),
                ], style={"flex": "1"}),
                html.Div([
                    html.Div("AREAS FOR IMPROVEMENT", className="card-title"),
                    html.Div([
                        html.Span(s, className="tag")
                        for s in linguistic.get("areas_for_improvement", [])
                    ], className="tag-list"),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "24px"}),
        ], className="card")

        return html.Div([
            metadata_card,
            summary_card,
            teaching_card,
            spatial_card,
            blooms_card,
            speech_card,
            linguistic_card,
        ])
