# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="Replicabilidade: ICC(2,k), SEM, MDC, Bland–Altman, Pearson",
    layout="wide",
)

st.title("📊 Replicabilidade: Cinemática vs Smartphone")

st.markdown(
    """
**Entrada esperada:** arquivo `.csv` ou `.txt` com **cabeçalho** e **duas colunas numéricas**  
- Coluna 1: **Cinemática**
- Coluna 2: **Smartphone**

O app calcula:
- **ICC(2,k)** (two-way random, *absolute agreement*, *average measures*) + **p-value** (H0: ICC=0)
- **IC 95% do ICC(2,k)** (bootstrap por sujeitos)
- **SEM** e **MDC95**
- **Bland–Altman** (viés e limites de concordância)
- **Correlação de Pearson (r)** + **p-value**
"""
)

# -------------------------
# Helpers
# -------------------------
def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce numeric values; supports decimal comma."""
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def icc_2_k(data: np.ndarray) -> dict:
    """
    ICC(2,k): two-way random effects, absolute agreement, average measures
    Shrout & Fleiss convention.

    Also returns an F-test p-value for H0: ICC = 0 (MSR == MSE).
    data: shape (n_subjects, k_raters)
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_subjects x k_raters).")
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 subjects and 2 columns (raters/instruments).")

    mean_per_subject = np.mean(data, axis=1, keepdims=True)  # (n,1)
    mean_per_rater = np.mean(data, axis=0, keepdims=True)    # (1,k)
    grand_mean = np.mean(data)

    SSR = k * np.sum((mean_per_subject - grand_mean) ** 2)  # subjects
    SSC = n * np.sum((mean_per_rater - grand_mean) ** 2)    # raters
    SSE = np.sum((data - mean_per_subject - mean_per_rater + grand_mean) ** 2)  # residual

    dfR = n - 1
    dfC = k - 1
    dfE = (n - 1) * (k - 1)

    MSR = SSR / dfR
    MSC = SSC / dfC
    MSE = SSE / dfE

    # ICC(2,1) and ICC(2,k)
    denom_21 = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    icc21 = (MSR - MSE) / denom_21 if denom_21 != 0 else np.nan

    denom_2k = MSR + ((MSC - MSE) / n)
    icc2k = (MSR - MSE) / denom_2k if denom_2k != 0 else np.nan

    # F test for H0: MSR == MSE  (equivalently ICC=0 in this setting)
    if MSE <= 0:
        F = np.inf
        p_value = 0.0
    else:
        F = MSR / MSE
        p_value = float(stats.f.sf(F, dfR, dfE))  # survival function = 1 - cdf

    return {
        "n_subjects": n,
        "k_raters": k,
        "MSR": float(MSR),
        "MSC": float(MSC),
        "MSE": float(MSE),
        "dfR": int(dfR),
        "dfE": int(dfE),
        "F": float(F),
        "p_value": float(p_value),
        "ICC(2,1)": float(icc21),
        "ICC(2,k)": float(icc2k),
    }


def bootstrap_icc2k_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int | None = 0,
) -> dict:
    """
    IC do ICC(2,k) via bootstrap por sujeitos (reamostra linhas).
    Retorna percentis (2.5%, 97.5%) por padrão.

    Observação: para n muito pequeno, o IC pode ficar instável.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_subjects x k_raters).")
    n, k = data.shape
    if n < 3:
        return {"low": np.nan, "high": np.nan, "n_ok": 0, "n_boot": n_boot}

    rng = np.random.default_rng(random_state)
    iccs = []

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)  # reamostra sujeitos
        sample = data[idx, :]
        try:
            out = icc_2_k(sample)
            v = out["ICC(2,k)"]
            if np.isfinite(v):
                iccs.append(v)
        except Exception:
            pass

    if len(iccs) < max(30, 0.1 * n_boot):
        return {"low": np.nan, "high": np.nan, "n_ok": len(iccs), "n_boot": n_boot}

    alpha = 1.0 - ci
    low = float(np.percentile(iccs, 100 * (alpha / 2)))
    high = float(np.percentile(iccs, 100 * (1 - alpha / 2)))
    return {"low": low, "high": high, "n_ok": len(iccs), "n_boot": n_boot}


def pearson_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-sided p-value."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader("📁 Envie o arquivo (CSV/TXT)", type=["csv", "txt"])

left, right = st.columns([1, 1])

if not uploaded:
    st.info("Envie um arquivo para começar.")
    st.stop()

with st.spinner("Lendo arquivo..."):
    df = pd.read_csv(uploaded, sep=None, engine="python")

st.subheader("Pré-visualização")
st.dataframe(df.head(20), use_container_width=True)

if df.shape[1] < 2:
    st.error("O arquivo precisa ter pelo menos 2 colunas.")
    st.stop()

st.subheader("Mapeamento das colunas")
c1, c2 = st.columns(2)
with c1:
    col_cin = st.selectbox("Coluna da Cinemática", options=list(df.columns), index=0)
with c2:
    default_idx = 1 if df.shape[1] > 1 else 0
    col_smart = st.selectbox("Coluna do Smartphone", options=list(df.columns), index=default_idx)

x = coerce_numeric(df[col_cin])
y = coerce_numeric(df[col_smart])

valid = x.notna() & y.notna()
n_removed = int((~valid).sum())
x = x[valid].to_numpy()
y = y[valid].to_numpy()

if len(x) < 3:
    st.error("Após remover valores inválidos, restaram poucos pares (precisa de pelo menos 3).")
    st.stop()

if n_removed > 0:
    st.warning(f"Foram removidas {n_removed} linhas por valores ausentes/não numéricos.")

# ICC data: n_subjects x k_raters (2 colunas)
data_icc = np.column_stack([x, y])
icc_out = icc_2_k(data_icc)
icc2k = icc_out["ICC(2,k)"]

# -------------------------
# ICC CI settings (bootstrap)
# -------------------------
st.subheader("Configurações do IC 95% do ICC(2,k)")
b1, b2 = st.columns([1, 1])
with b1:
    n_boot = st.slider("Nº de reamostragens (bootstrap)", min_value=200, max_value=5000, value=2000, step=200)
with b2:
    seed = st.number_input("Seed (reprodutibilidade)", value=0, step=1)

with st.spinner("Calculando IC 95% do ICC(2,k) (bootstrap)..."):
    icc_ci = bootstrap_icc2k_ci(data_icc, n_boot=int(n_boot), ci=0.95, random_state=int(seed))

icc_ci_low = icc_ci["low"]
icc_ci_high = icc_ci["high"]

# SEM / MDC settings
st.subheader("Configurações de SEM/MDC")
sem_basis = st.radio(
    "Base do SD para SEM",
    options=[
        "SD do valor médio ( (cinemática+smartphone)/2 )",
        "SD pooled (todas as medições das duas colunas empilhadas)",
    ],
    index=0,
    horizontal=True,
)

mean_pair = (x + y) / 2.0

if "médio" in sem_basis:
    sd_for_sem = float(np.std(mean_pair, ddof=1))
    sd_label = "SD da média dos instrumentos"
else:
    stacked = np.concatenate([x, y])
    sd_for_sem = float(np.std(stacked, ddof=1))
    sd_label = "SD pooled (colunas empilhadas)"

sem = sd_for_sem * np.sqrt(max(0.0, 1.0 - icc2k)) if np.isfinite(icc2k) else np.nan
mdc95 = 1.96 * np.sqrt(2) * sem if np.isfinite(sem) else np.nan

# Pearson
r, p_r = pearson_with_p(x, y)

# Bland–Altman
diff = y - x  # smartphone - cinemática
bias = float(np.mean(diff))
sd_diff = float(np.std(diff, ddof=1))
loa_low = bias - 1.96 * sd_diff
loa_high = bias + 1.96 * sd_diff

# -------------------------
# Outputs
# -------------------------
with left:
    st.subheader("Resultados (numéricos)")

    icc_ci_str = (
        f"[{icc_ci_low:.4f}, {icc_ci_high:.4f}]"
        if np.isfinite(icc_ci_low) and np.isfinite(icc_ci_high)
        else "NA"
    )

    results = pd.DataFrame(
        {
            "Métrica": [
                "N (pares válidos)",
                "ICC(2,k)",
                "ICC(2,k) IC95% (bootstrap)",
                "p do ICC (H0: ICC=0)",
                "F do ICC (MSR/MSE)",
                "ICC(2,1) (extra)",
                sd_label,
                "SEM",
                "MDC95",
                "Pearson r",
                "p do Pearson (2 caudas)",
                "Bland–Altman viés (smart - cin)",
                "LoA inferior",
                "LoA superior",
            ],
            "Valor": [
                icc_out["n_subjects"],
                icc2k,
                icc_ci_str,
                icc_out["p_value"],
                icc_out["F"],
                icc_out["ICC(2,1)"],
                sd_for_sem,
                sem,
                mdc95,
                r,
                p_r,
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
                    "Termo": ["MSR (sujeitos)", "MSC (instrumentos)", "MSE (erro)", "dfR", "dfE"],
                    "Valor": [icc_out["MSR"], icc_out["MSC"], icc_out["MSE"], icc_out["dfR"], icc_out["dfE"]],
                }
            )
        )
        st.caption("ICC(2,k): two-way random, absolute agreement, average measures (k = nº de colunas).")
        st.caption(f"Bootstrap: n_boot={icc_ci['n_boot']}, válidos={icc_ci['n_ok']}")

with right:
    st.subheader("Gráficos")

    # Scatter + identidade
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(x, y)
    mn = float(min(np.min(x), np.min(y)))
    mx = float(max(np.max(x), np.max(y)))
    ax1.plot([mn, mx], [mn, mx])
    ax1.set_xlabel("Cinemática")
    ax1.set_ylabel("Smartphone")
    ax1.set_title("Dispersão (linha identidade)")
    st.pyplot(fig1, clear_figure=True)

    # Bland–Altman
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

    # Histograma das diferenças
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
