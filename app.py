import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
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

# =========================
# Streamlit layout
st.title("Bonvi omniPD Web App")
st.write("Inserisci almeno 4 punti dati, tra cui sprint: tempo (s) e potenza (W)")

# =========================
# Input dati
st.subheader("Inserisci i dati")
num_rows = st.number_input("Numero di punti dati", min_value=4, max_value=20, value=4, step=1)

time_values = []
power_values = []

for i in range(num_rows):
    cols = st.columns(2)
    t = cols[0].number_input(f"Time (s) #{i+1}", min_value=1, value=none, step=1, format="%d")
    P = cols[1].number_input(f"Power (W) #{i+1}", min_value=1, value=none, step=1, format="%d")
    time_values.append(t)
    power_values.append(P)

# =========================
# Calcolo
if st.button("Calcola"):
    data = [(t, P) for t, P in zip(time_values, power_values) if t is not None and P is not None]
    
    if len(data) < 4:
        st.error("Errore: inserire almeno 4 punti dati validi.")
    else:
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
        fig1.update_layout(title="OmPD Curve", hovermode="x unified", height=700)

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
        # Calcolo valori teorici
        durations_s = [5*60, 10*60, 15*60, 20*60, 30*60]  # in secondi
        predicted_powers = [int(round(float(ompd_power(t, CP, W_prime, Pmax, A)))) for t in durations_s]

        # =========================
        # Mostra i 3 riquadri sopra i grafici
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Parametri stimati")
            st.write(f"CP: {int(round(CP))} W")
            st.write(f"W': {int(round(W_prime))} J")
            st.write(f"99% W'eff at {_format_time_label_custom(t_99)}")
            st.write(f"Pmax: {int(round(Pmax))} W")
            st.write(f"A: {A:.2f}")

        with col2:
            st.subheader("Residual summary")
            st.write(f"RMSE: {RMSE:.2f} W")
            st.write(f"MAE: {MAE:.2f} W")
            st.write(f"Bias: {bias_real:.2f} W")

        with col3:
            st.subheader("Valori teorici")
            for t, p in zip(durations_s, predicted_powers):
                minutes = t // 60
                st.write(f"{minutes}m: {p} W")

        # =========================
        # Mostra grafici sotto i riquadri
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)
