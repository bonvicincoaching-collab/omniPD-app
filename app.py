import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

# =========================
# Costante di modello
TCPMAX = 1800  # secondi

# =========================
# Funzioni modello
def ompd_power(t, CP, W_prime, Pmax, A):
    t = np.array(t, dtype=float)
    base = (W_prime / t) * (1 - np.exp(-t * (Pmax - CP) / W_prime)) + CP
    P = np.where(t <= TCPMAX, base, base - A * np.log(t / TCPMAX))
    return P

def ompd_power_short(t, CP, W_prime, Pmax):
    t = np.array(t, dtype=float)
    return (W_prime / t) * (1 - np.exp(-t * (Pmax - CP) / W_prime)) + CP

def ompd_power_with_bias(t, CP, W_prime, Pmax, A, B):
    t = np.array(t, dtype=float)
    base = (W_prime / t) * (1 - np.exp(-t * (Pmax - CP) / W_prime)) + CP
    P = np.where(t <= TCPMAX, base, base - A * np.log(t / TCPMAX))
    return P + B

def w_eff(t, W_prime, CP, Pmax):
    return W_prime * (1 - np.exp(-t * (Pmax - CP) / W_prime))

def _format_time_label_custom(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs}s" if minutes else f"{secs}s"

def _format_time_label_custom_residuals(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        h = minutes // 60
        m = minutes % 60
        return f"{h}h" if m==0 else f"{h}h{m}m"
    return f"{minutes}m" if secs==0 else f"{minutes}m{secs}s"

# =========================
# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "OmPD Web App"

# =========================
# Layout
app.layout = html.Div([
    html.H2("OmPD Data Entry", style={"textAlign": "center"}),
    html.P("Inserisci almeno 4 punti dati: tempo (s) e potenza (W)", style={"textAlign": "center"}),

    # Tabella dati e riquadro parametri affiancati
    html.Div([
        html.Div(id="data-entry-div", style={"display":"inline-block", "verticalAlign":"top", "marginRight":"30px"}, children=[
            html.Table(id="data-entry-table",
                       children=[
                           html.Tr([html.Th("Time (s)"), html.Th("Power (W)")])
                       ]),
            html.Button("Add Row", id="add-row-btn", n_clicks=0, style={"marginTop":"5px"}),
            html.Button("Calcola", id="calculate-btn", n_clicks=0, style={"marginLeft": "10px"})
        ]),
        html.Div(id="params-box", style={
            "display":"inline-block", "verticalAlign":"top",
            "border":"1px solid gray", "padding":"10px", "minWidth":"150px", "height":"120px"
        }, children=[
            html.H4("Parametri stimati", style={"textAlign":"center"}),
            html.P("CP: -"),
            html.P("W': -"),
            html.P("Pmax: -"),
            html.P("A: -")
        ])
    ], style={"marginBottom":"20px"}),

    html.Div(id="error-msg", style={"color": "red", "marginTop": "10px"}),

    html.Hr(),

    # Grafici in un'unica sezione, con altezza maggiore
    html.Div(id="graphs-div")
], style={"width": "90%", "margin": "auto"})

# =========================
# Callback: aggiungi/rimuovi riga
@app.callback(
    Output("data-entry-table", "children"),
    Input("add-row-btn", "n_clicks"),
    Input({"type": "remove-btn", "index": dash.ALL}, "n_clicks"),
    State("data-entry-table", "children")
)
def update_rows(add_clicks, remove_clicks, rows):
    triggered = ctx.triggered_id

    # Prima apertura: crea 4 righe base se non ci sono
    if triggered is None and len(rows) == 1:  # solo header
        default_rows = []
        for i in range(4):
            row_children = [
                html.Td(dcc.Input(type="number", placeholder="Time", id={"type":"time-input","index":i}, style={"width":"100px"})),
                html.Td(dcc.Input(type="number", placeholder="Power", id={"type":"power-input","index":i}, style={"width":"100px"}))
            ]
            if i == 0:
                row_children.append(html.Td("sprint power, best 5-10s"))
            else:
                row_children.append(html.Td(""))
            default_rows.append(html.Tr(row_children))
        return [rows[0]] + default_rows

    # Aggiungi nuova riga
    if triggered == "add-row-btn":
        new_index = len(rows) - 1
        new_row = html.Tr([
            html.Td(dcc.Input(type="number", placeholder="Time", id={"type":"time-input","index":new_index}, style={"width":"100px"})),
            html.Td(dcc.Input(type="number", placeholder="Power", id={"type":"power-input","index":new_index}, style={"width":"100px"})),
            html.Td(html.Button("Remove", n_clicks=0, id={"type":"remove-btn","index":new_index}))
        ])
        return rows + [new_row]

    # Rimuovi riga
    if isinstance(triggered, dict) and triggered.get("type") == "remove-btn":
        remove_idx = triggered["index"]
        new_rows = [rows[0]]  # header
        for row in rows[1:]:
            last_cell = row['props']['children'][-1]  # ultima cella
            # controlla se contiene un bottone
            children_of_cell = last_cell.get('props', {}).get('children', None)
            if isinstance(children_of_cell, dict):
                btn_id = children_of_cell.get('props', {}).get('id', {})
                if isinstance(btn_id, dict) and btn_id.get("index") == remove_idx:
                    continue  # salta questa riga
            new_rows.append(row)
        return new_rows

    return rows



# =========================
# Callback: calcola e aggiorna grafici e parametri
@app.callback(
    Output("graphs-div", "children"),
    Output("params-box", "children"),
    Output("error-msg", "children"),
    Input("calculate-btn", "n_clicks"),
    State({"type":"time-input","index":dash.ALL}, "value"),
    State({"type":"power-input","index":dash.ALL}, "value")
)
def calculate_graphs(n_clicks, time_values, power_values):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, ""

    # Leggi i dati
    data = []
    for t, P in zip(time_values, power_values):
        if t is not None and P is not None:
            data.append((t, P))
    if len(data) < 4:
        return dash.no_update, dash.no_update, "Errore: inserire almeno 4 punti dati validi."

    df = pd.DataFrame(data, columns=["t","P"])

    # Fit OmPD standard
    initial_guess = [np.percentile(df["P"],30), 20000, df["P"].max(), 5]
    params, _ = curve_fit(ompd_power, df["t"].values, df["P"].values,
                          p0=initial_guess, maxfev=20000)
    CP, W_prime, Pmax, A = params

    # Fit OmPD con bias
    initial_guess_bias = [np.percentile(df["P"],30),20000,df["P"].max(),5,0]
    param_bounds = ([0,0,0,0,-100], [1000,50000,5000,100,100])
    params_bias, _ = curve_fit(ompd_power_with_bias,
                               df["t"].values.astype(float),
                               df["P"].values.astype(float),
                               p0=initial_guess_bias,
                               bounds=param_bounds,
                               maxfev=20000)
    CP_b, W_prime_b, Pmax_b, A_b, B_b = params_bias
    P_pred = ompd_power_with_bias(df["t"].values.astype(float), *params_bias)
    residuals = df["P"].values.astype(float) - P_pred
    RMSE = np.sqrt(np.mean(residuals**2))
    MAE = np.mean(np.abs(residuals))
    bias_real = B_b

    # W'eff
    T_plot_w = np.linspace(1, 3*60, 500)
    Weff_plot = w_eff(T_plot_w, W_prime, CP, Pmax)
    W_99 = 0.99 * W_prime
    t_99_idx = np.argmin(np.abs(Weff_plot - W_99))
    t_99 = T_plot_w[t_99_idx]
    w_99 = Weff_plot[t_99_idx]

    # =========================
    # Grafico OmPD
    T_plot = np.logspace(np.log10(1.0), np.log10(max(max(df["t"])*1.1, 180*60)), 500)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["t"], y=df["P"], mode='markers', name="Dati reali", marker=dict(symbol='x', size=10)))
    fig1.add_trace(go.Scatter(x=T_plot, y=ompd_power(T_plot,*params), mode='lines', name="OmPD"))
    fig1.add_trace(go.Scatter(x=T_plot[T_plot<=TCPMAX], y=ompd_power_short(T_plot[T_plot<=TCPMAX], CP, W_prime, Pmax),
                              mode='lines', name="Curva base t ≤ TCPMAX", line=dict(dash='dash', color='blue')))
    fig1.add_hline(y=CP, line=dict(color='red', dash='dash'), annotation_text="CP", annotation_position="top right")
    fig1.add_vline(x=TCPMAX, line=dict(color='blue', dash='dot'), annotation_text="TCPMAX", annotation_position="bottom left")
    fig1.update_xaxes(type='log', title_text="Time (s)")
    fig1.update_yaxes(title_text="Power (W)")
    fig1.update_layout(title="OmPD Curve", hovermode="x unified", height=700)  # altezza raddoppiata

    # =========================
    # Grafico Residuals
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["t"], y=residuals, mode='lines+markers', name="Residuals", marker=dict(symbol='x', size=8), line=dict(color='red')))
    fig2.add_hline(y=0, line=dict(color='black', dash='dash'))
    fig2.update_xaxes(type='log', title_text="Time (s)")
    fig2.update_yaxes(title_text="Residuals (W)")
    fig2.update_layout(title="Residuals", hovermode="x unified", height=700)

    # =========================
    # Grafico W'eff
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=T_plot_w, y=Weff_plot, mode='lines', name="W'eff", line=dict(color='green')))
    fig3.add_hline(y=w_99, line=dict(color='blue', dash='dash'))
    fig3.add_vline(x=t_99, line=dict(color='blue', dash='dash'))
    fig3.add_annotation(x=t_99, y=W_99, text=f"99% W'eff at {_format_time_label_custom(t_99)}", showarrow=True, arrowhead=2)
    fig3.update_xaxes(title_text="Time (s)")
    fig3.update_yaxes(title_text="W'eff (J)")
    fig3.update_layout(title="OmPD Effective W'", hovermode="x unified", height=700)

    # =========================
    # Riquadri: 3 box
    durations_s = [5*60, 10*60, 15*60, 20*60, 30*60]  # in secondi
    predicted_powers = [int(round(float(ompd_power(t, CP, W_prime, Pmax, A)))) for t in durations_s]

    params_content = html.Div([
        # Riquadro 1: Parametri stimati + t_99
        html.Div([
            html.H4("Parametri stimati", style={"textAlign":"center", "margin":"5px"}),
            html.P(f"CP: {int(round(CP))} W", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"W': {int(round(W_prime))} J", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"99% W'eff at {_format_time_label_custom(t_99)}", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"Pmax: {int(round(Pmax))} W", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"A: {A:.2f}", style={"margin":"2px", "fontSize":"14px"})
        ], style={"border":"1px solid gray", "padding":"10px", "marginRight":"10px", 
                  "width":"180px", "display":"inline-block", "verticalAlign":"top"}),

        # Riquadro 2: Valori teorici
        html.Div([
            html.H4("Valori teorici", style={"textAlign":"center", "margin":"5px"}),
            *[html.P(f"{t//60}m: {p} W", style={"margin":"2px", "fontSize":"14px"}) 
              for t,p in zip(durations_s, predicted_powers)]
        ], style={"border":"1px solid gray", "padding":"10px", "width":"150px", 
                  "display":"inline-block", "verticalAlign":"top"}),

        # Riquadro 3: Residual summary
        html.Div([
            html.H4("Residual summary", style={"textAlign":"center", "margin":"5px"}),
            html.P(f"RMSE: {RMSE:.2f} W", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"MAE: {MAE:.2f} W", style={"margin":"2px", "fontSize":"14px"}),
            html.P(f"Bias: {bias_real:.2f} W", style={"margin":"2px", "fontSize":"14px"})
        ], style={"border":"1px solid gray", "padding":"10px", "width":"150px", 
                  "display":"inline-block", "verticalAlign":"top"})
    ])

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ]), params_content, ""


# =========================
# Run App
if __name__ == "__main__":
    app.run(debug=True)
