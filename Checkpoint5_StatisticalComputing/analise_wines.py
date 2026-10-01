"""Checkpoint 5 — Statistical Computing with R and Python (Parte 1).

Grupo:
    Luigi Francisco — RM 571204
    Breno Filipe — RM 571824
    Igor Cypriano — RM 570353
    Diogo Lopes — RM 570732

Execute com: python analise_wines.py
O CSV original está em https://gist.github.com/tijptjik/9408623
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Salva os gráficos mesmo sem uma janela gráfica.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent
DADOS = BASE / "wine.csv"
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

df = pd.read_csv(DADOS)
alcool = df["Alcohol"].dropna()
acido = df["Malic.acid"].dropna()

print(f"Total de vinhos: {len(df)}")
print(
    f"Valores ausentes: álcool={df['Alcohol'].isna().sum()}, "
    f"ácido málico={df['Malic.acid'].isna().sum()}"
)
if len(df) != 178 or len(alcool) != 178 or len(acido) != 178:
    raise ValueError("O CSV esperado tem 178 vinhos e não tem nulos nas duas variáveis.")


# a) VISUALIZAÇÃO: tabela de frequências e gráfico para cada variável.
# A regra de Sturges dá ceil(1 + 3,322 log10(178)) = 9 classes.
def tabela_frequencias(serie: pd.Series) -> tuple[pd.DataFrame, int]:
    n = len(serie)
    k = int(np.ceil(1 + 3.322 * np.log10(n)))
    intervalos = pd.cut(serie, bins=k)
    fi = intervalos.value_counts().sort_index()
    tabela = pd.DataFrame({"fi": fi})
    tabela["fr (%)"] = (tabela["fi"] / n * 100).round(2)
    tabela["Fi"] = tabela["fi"].cumsum()
    # Calculamos a acumulada a partir das contagens para evitar somar arredondamentos.
    tabela["Fr (%)"] = (tabela["Fi"] / n * 100).round(2)
    tabela.index.name = "Intervalo"
    return tabela, k


def salvar_histograma(serie: pd.Series, titulo: str, eixo: str, cor: str, nome: str, k: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(serie, bins=k, color=cor, edgecolor="white")
    ax.set(title=titulo, xlabel=eixo, ylabel="Frequência absoluta")
    fig.tight_layout()
    fig.savefig(RESULTADOS / nome, dpi=180)
    plt.close(fig)


tab_alcool, k_alcool = tabela_frequencias(alcool)
tab_acido, k_acido = tabela_frequencias(acido)
tab_alcool.to_csv(RESULTADOS / "frequencia_alcool.csv", encoding="utf-8-sig")
tab_acido.to_csv(RESULTADOS / "frequencia_acido_malico.csv", encoding="utf-8-sig")
print("\nTabela de frequências — álcool (9 classes):\n", tab_alcool.to_string())
print("\nTabela de frequências — ácido málico (9 classes):\n", tab_acido.to_string())
salvar_histograma(alcool, "Distribuição do álcool", "Álcool", "#377eb8", "01_histograma_alcool.png", k_alcool)
salvar_histograma(acido, "Distribuição do ácido málico", "Ácido málico", "#e69f00", "02_histograma_acido_malico.png", k_acido)

# Interpretação: o álcool vai de 11,03 a 14,83 e é aproximadamente simétrico;
# a primeira e a última classe têm poucas observações (3 e 2 vinhos).
# O ácido málico vai de 0,74 a 5,80; a segunda classe concentra 68 vinhos,
# e a cauda longa à direita indica assimetria positiva.


# b) ANÁLISE DESCRITIVA: tendência central, dispersão e quartis.
def descrever(serie: pd.Series) -> pd.Series:
    q1, q2, q3 = serie.quantile([0.25, 0.50, 0.75])
    return pd.Series({
        "Média": serie.mean(),
        "Mediana": q2,
        "Moda": ", ".join(f"{valor:.2f}" for valor in serie.mode()),
        "Mínimo": serie.min(),
        "Máximo": serie.max(),
        "Amplitude": serie.max() - serie.min(),
        "Variância amostral": serie.var(),
        "Desvio padrão amostral": serie.std(),
        "Coeficiente de variação (%)": serie.std() / serie.mean() * 100,
        "Q1 (25%)": q1,
        "Q2 (50%)": q2,
        "Q3 (75%)": q3,
        "Amplitude interquartil": q3 - q1,
        "Assimetria": serie.skew(),
    })


resumo = pd.DataFrame({"Álcool": descrever(alcool), "Ácido málico": descrever(acido)})
resumo.to_csv(RESULTADOS / "medidas_descritivas.csv", encoding="utf-8-sig")
print("\nMedidas descritivas:\n", resumo.to_string())
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([alcool, acido], tick_labels=["Álcool", "Ácido málico"], patch_artist=True)
for caixa, cor in zip(bp["boxes"], ["#377eb8", "#e69f00"]):
    caixa.set_facecolor(cor)
ax.set(title="Boxplots das variáveis", ylabel="Valor observado")
fig.tight_layout()
fig.savefig(RESULTADOS / "03_boxplots.png", dpi=180)
plt.close(fig)

# Interpretação: no álcool, média (~13,00) e mediana (13,05) são próximas;
# o coeficiente de variação é ~6,2%, sem outliers pelo critério de 1,5 IQR.
# No ácido málico, média (~2,34) supera a mediana (~1,87), indicando cauda
# à direita. O coeficiente de variação é ~47,8%; há 3 outliers acima de ~5,30.
# Os quartis do álcool são ~12,36, 13,05 e 13,68; os do ácido são ~1,60,
# 1,87 e 3,08. A metade central do ácido é, portanto, mais espalhada.


# c) ANÁLISE PROBABILÍSTICA: distribuição e dois cálculos por variável.
# Para o álcool, avaliamos uma distribuição Normal ajustada à média e ao
# desvio padrão amostral. O Shapiro-Wilk testa a normalidade estrita.
media_alc, dp_alc = alcool.mean(), alcool.std()
shapiro_alc = stats.shapiro(alcool)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(alcool, bins=k_alcool, density=True, color="#377eb8", alpha=0.75, edgecolor="white")
x = np.linspace(alcool.min() - 0.5, alcool.max() + 0.5, 300)
ax[0].plot(x, stats.norm.pdf(x, media_alc, dp_alc), color="#b22222", lw=2, label="Normal ajustada")
ax[0].set(title="Álcool: densidade e Normal", xlabel="Álcool", ylabel="Densidade")
ax[0].legend()
stats.probplot(alcool, dist="norm", plot=ax[1])
ax[1].set_title("Álcool: gráfico Q-Q")
fig.tight_layout()
fig.savefig(RESULTADOS / "04_ajuste_normal_alcool.png", dpi=180)
plt.close(fig)

p_alc_maior14 = stats.norm.sf(14, media_alc, dp_alc)
p_alc_12a14 = stats.norm.cdf(14, media_alc, dp_alc) - stats.norm.cdf(12, media_alc, dp_alc)
f_alc_maior14 = (alcool > 14).mean()
f_alc_12a14 = ((alcool > 12) & (alcool < 14)).mean()
print(f"\nShapiro-Wilk álcool: p={shapiro_alc.pvalue:.6f}")
print(f"P(álcool > 14): Normal={p_alc_maior14:.2%}; observada={f_alc_maior14:.2%}")
print(f"P(12 < álcool < 14): Normal={p_alc_12a14:.2%}; observada={f_alc_12a14:.2%}")
# Interpretação: a 5%, o Shapiro-Wilk rejeita a normalidade estrita.
# Ainda assim, a Normal aproxima bem estas duas probabilidades específicas:
# cerca de 10,9% contra 12,4%, e 78,2% contra 75,3% observados.

# Para o ácido málico, o histograma e o Q-Q plot mostram assimetria à direita,
# e o Shapiro-Wilk rejeita a Normal. Usamos a distribuição empírica (ECDF):
# P(X <= x) é a fração observada de vinhos com valor até x.
shapiro_ac = stats.shapiro(acido)
valores = np.sort(acido.to_numpy())
ecdf = np.arange(1, len(valores) + 1) / len(valores)
fig, ax = plt.subplots(figsize=(9, 5))
ax.step(valores, ecdf, where="post", color="#e69f00", lw=2)
ax.axvline(2, color="#666666", linestyle="--", label="2")
ax.axvline(3, color="#b22222", linestyle="--", label="3")
ax.set(title="Ácido málico: distribuição acumulada empírica", xlabel="Ácido málico",
       ylabel="Probabilidade acumulada", ylim=(0, 1.02))
ax.legend(title="Limiares")
fig.tight_layout()
fig.savefig(RESULTADOS / "05_distribuicao_empirica_acido_malico.png", dpi=180)
plt.close(fig)

p_ac_maior3 = (acido > 3).mean()
p_ac_menor2 = (acido < 2).mean()
print(f"Shapiro-Wilk ácido málico: p={shapiro_ac.pvalue:.3e}")
print(f"P(ácido málico > 3) empírica = {p_ac_maior3:.2%}")
print(f"P(ácido málico < 2) empírica = {p_ac_menor2:.2%}")
# Interpretação: 47 de 178 vinhos (26,4%) estão acima de 3; 100 de 178
# (56,2%) estão abaixo de 2. A massa dos valores baixos e a cauda longa
# tornam a distribuição empírica mais adequada aqui do que a Normal.

probabilidades = pd.DataFrame([
    {"Evento": "Álcool > 14", "Modelo": "Normal ajustada", "Probabilidade": p_alc_maior14,
     "Frequência observada": f_alc_maior14},
    {"Evento": "12 < Álcool < 14", "Modelo": "Normal ajustada", "Probabilidade": p_alc_12a14,
     "Frequência observada": f_alc_12a14},
    {"Evento": "Ácido málico > 3", "Modelo": "Empírica", "Probabilidade": p_ac_maior3,
     "Frequência observada": p_ac_maior3},
    {"Evento": "Ácido málico < 2", "Modelo": "Empírica", "Probabilidade": p_ac_menor2,
     "Frequência observada": p_ac_menor2},
])
probabilidades.to_csv(RESULTADOS / "probabilidades.csv", index=False, encoding="utf-8-sig")
print(f"\nTabelas e gráficos salvos em: {RESULTADOS}")
