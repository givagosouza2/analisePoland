# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Replicabilidade: ICC(2,k), SEM, MDC, Bland–Altman, Pearson", layout="wide")
st.title("📊 Replicabilidade entre Cinemática vs Smartphone")

st.markdown(
    """
**Entrada esperada:** um arquivo `.csv` / `.txt` com **cabeçalho** e **duas colunas numéricas**  
- Coluna 1: **Cinemática**
- Coluna 2: **Smartphone**

O app calcula:
- **ICC(2,k)** (two-way random, *absolute agreement*, *average measures*)
- **SEM** e **MDC95**
- **Bland–Altman** (viés e limites de concordância)
- **Correlação de Pearson (r)**
"""
)

# -------------------------
# Helpers
# -------------------------
def coerce_numeric(series: pd.Series) -> pd.Series:
    """Try to coerce numeric even if decimal commas exist."""
    s = series.astype(str).str.strip()
    # replace comma decimal with dot if needed
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def icc_2_k(data: np.ndarray) -> dict:
    """
    ICC(2,k): two-way random effects, absolute agreement, average measures
    Shrout & Fleiss convention.

    data: shape (n_subjects, k_raters)
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (n_subjects x k_raters).")
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 subjects and 2 raters (columns).")

    # Means
    mean_per_subject = np.mean(data, axis=1, keepdims=True)  # (n,1)
    mean_per_rater = np.mean(data, axis=0, keepdims=True)    # (1,k)
    grand_mean = np.mean(data)

    # Sum of squares
    SSR = k * np.sum((mean_per_subject - grand_mean) ** 2)  # rows/subjects
    SSC = n * np.sum((mean_per_rater - grand_mean) ** 2)    # columns/raters
    SSE = np.sum((data - mean_per_subject - mean_per_rater + grand_mean) ** 2)  # residual

    # Degrees of freedom
    dfR = n - 1
    dfC = k - 1
    dfE = (n - 1) * (k - 1)

    # Mean squares
    MSR = SSR / dfR
    MSC = SSC / dfC
    MSE = SSE / dfE

    # ICC(2,1) and ICC(2,k)
    icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + (k * (MSC - MSE) / n))
    icc2k = (MSR - MSE) / (MSR + ((MSC - MSE) / n))

    return {
        "n_subjects": n,
        "k_raters": k,
        "MSR": MSR,
        "MSC": MSC,
        "MSE": MSE,
        "ICC(2,1)": icc21,
        "ICC(2,k)": icc2k,
    }

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader("📁 Envie o arquivo (CSV/TXT)", type=["csv", "txt"])

left, right = st.columns([1, 1])

if uploaded:
    with st.spinner("Lendo arquivo..."):
        df = pd.read_csv(uploaded, sep=None, engine="python")

    st.subheader("Pré-visualização")
    st.dataframe(df.head(20), use_container_width=True)

    if df.shape[1] < 2:
        st.error("O arquivo precisa ter pelo menos 2 colunas.")
        st.stop()

    st.subheader("Mapeamento das colunas")
    col1, col2 = st.columns(2)
    with col1:
        col_cin = st.selectbox("Coluna da Cinemática", options=list(df.columns), index=0)
    with col2:
        default_idx = 1 if df.shape[1] > 1 else 0
        col_smart = st.selectbox("Coluna do Smartphone", options=list(df.columns), index=default_idx)

    x_raw = df[col_cin]
    y_raw = df[col_smart]

    x = coerce_numeric(x_raw)
    y = coerce_numeric(y_raw)

    valid = x.notna() & y.notna()
    n_removed = int((~valid).sum())

    x = x[valid].to_numpy()
    y = y[valid].to_numpy()

    if len(x) < 3:
        st.error("Após remover valores inválidos, restaram poucos pares (precisa de pelo menos 3).")
        st.stop()

    if n_removed > 0:
        st.warning(f"Foram removidas {n_removed} linhas por valores ausentes/não numéricos.")

    # Arrange data for ICC: n_subjects x k_raters
    data_icc = np.column_stack([x, y])

    # Calculations
    icc_out = icc_2_k(data_icc)
    icc2k = icc_out["ICC(2,k)"]

    # SEM choices
    st.subheader("Configurações de SEM/MDC")
    sem_basis = st.radio(
        "Base do SD para SEM",
        options=[
            "SD do valor médio ( (cinemática+smartphone)/2 )",
            "SD pooled (todas as medições das duas colunas empilhadas)"
        ],
        index=0,
        horizontal=True
    )

    mean_pair = (x + y) / 2.0
    if "médio" in sem_basis:
        sd_for_sem = float(np.std(mean_pair, ddof=1))
        sd_label = "SD da média dos instrumentos"
    else:
        stacked = np.concatenate([x, y])
        sd_for_sem = float(np.std(stacked, ddof=1))
        sd_label = "SD pooled (colunas empilhadas)"

    sem = sd_for_sem * np.sqrt(max(0.0, 1.0 - icc2k))
    mdc95 = 1.96 * np.sqrt(2) * sem

    # Pearson
    r = pearson_r(x, y)

    # Bland–Altman
    diff = y - x
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff

    # -------------------------
    # Outputs
    # -------------------------
    with left:
        st.subheader("Resultados (numéricos)")
        results = pd.DataFrame(
            {
                "Métrica": [
                    "N (pares válidos)",
                    "ICC(2,k)",
                    "ICC(2,1) (extra)",
                    sd_label,
                    "SEM",
                    "MDC95",
                    "Pearson r",
                    "Bland–Altman viés (média diff: smart - cin)",
                    "LoA inferior",
                    "LoA superior",
                ],
                "Valor": [
                    icc_out["n_subjects"],
                    icc2k,
                    icc_out["ICC(2,1)"],
                    sd_for_sem,
                    sem,
                    mdc95,
                    r,
                    bias,
                    loa_low,
                    loa_high,
                ],
            }
        )
        st.dataframe(results, use_container_width=True, hide_index=True)

        with st.expander("Detalhes do ICC (ANOVA)"):
            st.write(
                pd.DataFrame(
                    {
                        "Termo": ["MSR (sujeitos)", "MSC (instrumentos)", "MSE (erro)"],
                        "Valor": [icc_out["MSR"], icc_out["MSC"], icc_out["MSE"]],
                    }
                )
            )
            st.caption("ICC(2,k): two-way random, absolute agreement, average measures (k = nº de colunas).")

    with right:
        st.subheader("Gráficos")

        # Scatter (with identity line)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(x, y)
        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))
        ax1.plot([mn, mx], [mn, mx])
        ax1.set_xlabel("Cinemática")
        ax1.set_ylabel("Smartphone")
        ax1.set_title("Dispersão (com linha identidade)")
        st.pyplot(fig1, clear_figure=True)

        # Bland–Altman plot
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(mean_pair, diff)
        ax2.axhline(bias)
        ax2.axhline(loa_low)
        ax2.axhline(loa_high)
        ax2.set_xlabel("Média dos dois instrumentos")
        ax2.set_ylabel("Diferença (Smartphone - Cinemática)")
        ax2.set_title("Bland–Altman")
        st.pyplot(fig2, clear_figure=True)

        # Differences histogram (optional but useful)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.hist(diff, bins=20)
        ax3.set_xlabel("Diferença (Smartphone - Cinemática)")
        ax3.set_ylabel("Frequência")
        ax3.set_title("Distribuição das diferenças")
        st.pyplot(fig3, clear_figure=True)

    st.divider()
    st.subheader("Exportar (opcional)")
    export_df = pd.DataFrame(
        {
            "cinematica": x,
            "smartphone": y,
            "media": mean_pair,
            "diff_smart_minus_cin": diff,
        }
    )
    st.download_button(
        "⬇️ Baixar tabela com média e diferenças (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="replicabilidade_tabela.csv",
        mime="text/csv",
    )

else:
    st.info("Envie um arquivo para começar.")
