import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
import plotly.graph_objects as go
import base64

# =========================
# Imposta la modalità wide
st.set_page_config(
    page_title="Bonvi omniPD model calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# Titolo e info
st.title("Bonvi omniPD model calculator <3")
st.markdown(
    """
Inserisci almeno **4 punti dati** (tra cui **sprint**) tempo (s) e potenza (W).  
Oppure, più in basso ⬇️ puoi caricare un CSV con i dati.  
Per avere valori di **A** inserisci **MMP oltre i 30 minuti** (opzionale).
"""
)

# =========================
# PDF locali
pdf_files = ["tutorial_fast.pdf",
             "guida_rapida_modello.pdf"]
pdf_names = ["Tutorial fast: come usare il calcolatore",
             "Guida al profilo di potenza omniPD"]

for pdf_file, pdf_name in zip(pdf_files, pdf_names):
    with st.expander(f"{pdf_name} (clicca per visualizzare)"):
        # Visualizza PDF embedded
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'''
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="600px"
            type="application/pdf"
        ></iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Bottone per scaricare PDF
        with open(pdf_file, "rb") as f:
            st.download_button(
                label=f"Scarica {pdf_name}",
                data=f.read(),
                file_name=pdf_file,
                mime="application/pdf"
            )

# =========================
# Paper 
st.markdown(
    '<a href="https://pubmed.ncbi.nlm.nih.gov/32131692/" '
    'target="_blank" style="text-decoration:none; color:#47c3f0;">📄 Paper</a>',
    unsafe_allow_html=True
)

# =========================
# Input dati manuale
num_rows = st.number_input("Numero di punti dati", min_value=4, max_value=20, value=4, step=1)
time_values = []
power_values = []

for i in range(num_rows):
    cols = st.columns(2)
    label = "Sprint (1–10s)" if i == 0 else f"#{i}"
    t_str = cols[0].text_input(f"{label} – Time (s)", value="", key=f"time_{i}")
    P_str = cols[1].text_input(f"{label} – Power (W)", value="", key=f"power_{i}")
    try:
        t_val = int(t_str)
        P_val = int(P_str)
        if t_val > 0 and P_val > 0:
            time_values.append(t_val)
            power_values.append(P_val)
    except Exception:
        pass

# =========================
# Funzione di calcolo e visualizzazione
def calcola_e_mostra(time_values, power_values):
    df = pd.DataFrame({"t": time_values, "P": power_values})

    # Fit OmPD standard
    initial_guess = [np.percentile(df["P"],30), 20000, df["P"].max(), 5]
    params, _ = curve_fit(ompd_power, df["t"].values, df["P"].values, p0=initial_guess, maxfev=20000)
    CP, W_prime, Pmax, A = params

    # Fit OmPD con bias
    initial_guess_bias = [np.percentile(df["P"],30),20000,df["P"].max(),5,0]
    param_bounds = ([0,0,0,0,-100], [1000,50000,5000,100,100])
    params_bias, _ = curve_fit(
        ompd_power_with_bias,
        df["t"].values.astype(float),
        df["P"].values.astype(float),
        p0=initial_guess_bias,
        bounds=param_bounds,
        maxfev=20000
    )
    CP_b, W_prime_b, Pmax_b, A_b, B_b = params_bias
    P_pred = ompd_power_with_bias(df["t"].values.astype(float), *params_bias)
    residuals = df["P"].values.astype(float) - P_pred
    RMSE = np.sqrt(np.mean(residuals**2))
    MAE = np.mean(np.abs(residuals))
    bias_real = B_b

    # Salva parametri
    st.session_state["params_computed"] = {
        "CP_b": CP_b, "W_prime_b": W_prime_b, "Pmax_b": Pmax_b, "A_b": A_b, "B_b": B_b
    }

    # W'eff
    T_plot_w = np.linspace(1, 3*60, 500)
    Weff_plot = w_eff(T_plot_w, W_prime, CP, Pmax)
    W_99 = 0.99 * W_prime
    t_99_idx = np.argmin(np.abs(Weff_plot - W_99))
    t_99 = T_plot_w[t_99_idx]
    w_99 = Weff_plot[t_99_idx]

    # Valori teorici
    durations_s = [5*60, 10*60, 15*60, 20*60, 30*60]
    predicted_powers = [int(round(float(ompd_power(t, CP, W_prime, Pmax, A)))) for t in durations_s]

    # =========================
    # Mostra riquadri
    if "time_values_csv" in st.session_state and "power_values_csv" in st.session_state:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)
        col4 = None

    with col1:
        st.markdown("**Parametri stimati**")
        st.markdown(f"CP: {int(round(CP))} W")
        st.markdown(f"W': {int(round(W_prime))} J")
        st.markdown(f"99% W'eff at {_format_time_label_custom(t_99)}")
        st.markdown(f"Pmax: {int(round(Pmax))} W")
        st.markdown(f"A: {A:.2f}")

    with col2:
        st.markdown("**Residual summary**")
        st.markdown(f"RMSE: {RMSE:.2f} W")
        st.markdown(f"MAE: {MAE:.2f} W")
        st.markdown(f"Bias: {bias_real:.2f} W")

    with col3:
        st.markdown("**Valori teorici**")
        for t, p in zip(durations_s, predicted_powers):
            minutes = t // 60
            st.markdown(f"{minutes}m: {p} W")
 
    if col4 is not None:
         with col4:
             st.markdown("**Valori reali per stesse durate**")
             # Durate corrispondenti a quelle teoriche
             match_durations = durations_s  # 5,10,15,20,30 minuti in secondi
             # Per ciascuna durata, trova il valore reale dal CSV più vicino al tempo
             times_csv = np.array(st.session_state["time_values_csv"])
             powers_csv = np.array(st.session_state["power_values_csv"])
             for t in match_durations:
                 idx = np.argmin(np.abs(times_csv - t))
                 P_real = powers_csv[idx]
                 minutes = t // 60
                 st.markdown(f"{minutes}m: {int(round(P_real))} W")

    # =========================
    # Grafici
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
    st.plotly_chart(fig1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["t"], y=residuals, mode='lines+markers', name="Residuals",
                              marker=dict(symbol='x', size=8), line=dict(color='red')))
    fig2.add_hline(y=0, line=dict(color='black', dash='dash'))
    fig2.update_xaxes(type='log', title_text="Time (s)")
    fig2.update_yaxes(title_text="Residuals (W)")
    fig2.update_layout(title="Residuals", hovermode="x unified", height=700)
    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=T_plot_w, y=Weff_plot, mode='lines', name="W'eff", line=dict(color='green')))
    fig3.add_hline(y=w_99, line=dict(color='blue', dash='dash'))
    fig3.add_vline(x=t_99, line=dict(color='blue', dash='dash'))
    fig3.add_annotation(x=t_99, y=W_99, text=f"99% W'eff at {_format_time_label_custom(t_99)}",
                        showarrow=True, arrowhead=2)
    fig3.update_xaxes(title_text="Time (s)")
    fig3.update_yaxes(title_text="W'eff (J)")
    fig3.update_layout(title="OmPD Effective W'", hovermode="x unified", height=700)
    st.plotly_chart(fig3)

# =========================
# Pulsante Calcola
if st.button("Calcola"):
    if len(time_values) >= 4:
        calcola_e_mostra(time_values, power_values)

# =========================
# File uploader CSV con filtro tempo massimo
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Oppure carica un CSV con i dati")
st.markdown("""
    <style>
    div.stFileUploader > label > div > div {
        font-size: 12px;
        height: auto;
        line-height: normal;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("(opzionale) Carica un file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)
        st.dataframe(df_csv.head(20), height=200)

        col_time = st.selectbox("Seleziona la colonna TEMPO", options=df_csv.columns)
        col_power = st.selectbox("Seleziona la colonna POTENZA", options=df_csv.columns)

        # Inserimento filtro per tempo massimo
        max_time = st.number_input(
            "(opzionale) Valore massimo di tempo (s) da importare. **⚠ sconsigliati valori inferiori a 1200-1800s ⚠**",
            min_value=1,
            value=int(df_csv[col_time].max())
        )

        if st.button("Importa dati CSV e calcola", key="csv_btn"):
            # Pulizia e filtro dati
            df_valid = df_csv[[col_time, col_power]].dropna()
            df_valid = df_valid[df_valid[col_time] <= max_time]

            time_values_csv = df_valid[col_time].astype(float).tolist()
            power_values_csv = df_valid[col_power].astype(float).tolist()

            st.session_state["time_values_csv"] = time_values_csv
            st.session_state["power_values_csv"] = power_values_csv
            st.success(f"Dati importati: {len(time_values_csv)} punti (tempo ≤ {max_time}s)")

            # Calcolo e visualizzazione immediata
            calcola_e_mostra(time_values_csv, power_values_csv)

    except Exception as e:
        st.error(f"Errore durante la lettura del CSV: {e}")

# =========================
# Calcolatore rapido funzionante
st.markdown("<hr>", unsafe_allow_html=True)
col_calc = st.columns([1])[0]
t_str = col_calc.text_input("(opzionale) Inserisci t (in s) per sapere quanti watt riesci a fare", value="1200", key="t_calc_final")
try:
    t_calc = max(1, int(t_str))
except Exception:
    t_calc = 60

if "params_computed" in st.session_state:
    p = st.session_state["params_computed"]
    P_calc = ompd_power_with_bias(
        t_calc,
        p["CP_b"], p["W_prime_b"], p["Pmax_b"], p["A_b"], p["B_b"]
    )
    time_label = _format_time_label_custom(t_calc)
    
    # Mostra valore calcolato
    col_calc.markdown(f"**{time_label} → {int(round(P_calc))} W** | (per aggiornare premere \"Enter\" dopo aver cambiato t)")
    
    # Se c'è CSV, mostra sotto il valore reale più vicino
    if "time_values_csv" in st.session_state and "power_values_csv" in st.session_state:
        times_csv = np.array(st.session_state["time_values_csv"])
        powers_csv = np.array(st.session_state["power_values_csv"])
        idx = np.argmin(np.abs(times_csv - t_calc))
        P_real_csv = powers_csv[idx]
        col_calc.markdown(f"Valore reale CSV: {int(round(P_real_csv))} W")
else:
    col_calc.markdown("⚠ Per calcolare è necessario importare CSV O calcolare con i valori manuali")
