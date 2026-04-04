import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

# -------------------------
# FONCTIONS MATHÉMATIQUES & FINANCIÈRES
# -------------------------

def calculate_returns(df):
    """Calcule les rendements logarithmiques (Section 6.2)"""
    # Rigueur : On utilise le log pour l'additivité temporelle
    return np.log(df['Close'] / df['Close'].shift(1)).dropna()

def get_statistics(returns):
    """Calcule les moments d'ordre supérieur et la volatilité (Section 6.3)"""
    return {
        "Mean": returns.mean(),
        "Annual Vol": returns.std() * np.sqrt(252),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis()
    }

def run_backtest(df):
    """Logique de stratégie SMA 20/50 (Section 6.5)"""
    data = df.copy()
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    
    # Signaux : 1 si SMA20 > SMA50, sinon 0
    data['Signal'] = np.where(data['SMA20'] > data['SMA50'], 1, 0)
    
    # Rendements stratégie (rendement actif * position la veille)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Strategy_Ret'] = data['Signal'].shift(1) * data['Returns']
    
    # Courbe de capital (Base 1000 DH)
    data['Equity_Curve'] = 1000 * np.exp(data['Strategy_Ret'].cumsum())
    
    # Métriques
    total_return = (data['Equity_Curve'].iloc[-1] / 1000) - 1
    std_dev = data['Strategy_Ret'].std()
    sharpe = (data['Strategy_Ret'].mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
    
    peak = data['Equity_Curve'].cummax()
    drawdown = (data['Equity_Curve'] - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Equity Curve": data['Equity_Curve']
    }

# -------------------------
# INTERFACE STREAMLIT
# -------------------------

st.set_page_config(page_title="Finance Analytics Platform", layout="wide")
st.title("🏦 Plateforme d'Analyse Financière Quantitative")

# SIDEBAR
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Symbole de l'actif", "BTC-USD")
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("today"))

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    # Suppression du multi-index si présent avec yfinance récent
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("❌ Aucune donnée trouvée.")
    st.stop()

# ONGLETS
tab_price, tab_stats, tab_backtest = st.tabs(["📈 Analyse de Prix", "📊 Analyse Statistique", "🚀 Backtesting"])

# --- ONGLET PRIX ---
with tab_price:
    st.subheader(f"Évolution du prix — {ticker}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Prix Clôture"))
    fig_price.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Prix (DH)")
    st.plotly_chart(fig_price, use_container_width=True)

# --- ONGLET STATISTIQUES ---
with tab_stats:
    rets = calculate_returns(data)
    stats_vals = get_statistics(rets)
    
    st.subheader("Indicateurs de Risque et Moments")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Moyenne Log", f"{stats_vals['Mean']:.5f}")
    col_b.metric("Volatilité Ann.", f"{stats_vals['Annual Vol']:.2%}")
    col_c.metric("Skewness", f"{stats_vals['Skewness']:.2f}")
    col_d.metric("Kurtosis", f"{stats_vals['Kurtosis']:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        # Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=rets, nbinsx=50, name="Rendements", histnorm='probability density'))
        fig_dist.update_layout(title="Distribution des Rendements", template="plotly_dark")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with c2:
        # QQ-Plot pour la normalité
        qq = stats.probplot(rets, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Données"))
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0]*qq[1][0]+qq[1][1], mode="lines", name="Normale", line=dict(color="red")))
        fig_qq.update_layout(title="Q-Q Plot (Normalité)", template="plotly_dark")
        st.plotly_chart(fig_qq, use_container_width=True)

# --- ONGLET BACKTESTING ---
with tab_backtest:
    st.subheader("Performance Stratégie SMA 20/50")
    res = run_backtest(data)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Rendement Total", f"{res['Total Return']:.2%}")
    m2.metric("Ratio de Sharpe", f"{res['Sharpe Ratio']:.2f}")
    m3.metric("Max Drawdown", f"{res['Max Drawdown']:.2%}")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=res['Equity Curve'].index, y=res['Equity Curve'], fill='tozeroy', name="Capital"))
    fig_eq.update_layout(title="Évolution du Capital (Investissement initial: 1000 DH)", template="plotly_dark")
    st.plotly_chart(fig_eq, use_container_width=True)