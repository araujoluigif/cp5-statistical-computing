# Checkpoint 5 — Statistical Computing with R and Python

Entrega da Parte 1, utilizando Python e o conjunto de dados Wines.

## Integrantes

| Nome | RM |
| --- | --- |
| Luigi Francisco | 571204 |
| Breno Filipe | 571824 |
| Igor Cypriano | 570353 |
| Diogo Lopes | 570732 |

## Arquivos

- `relatorio_parte1.pdf`: identificação e síntese da análise.
- `analise_wines.py`: código comentado das alíneas a, b e c.
- `wine.csv`: cópia dos dados usados, para execução sem acesso à internet.
- `resultados/`: tabelas calculadas e gráficos gerados pelo código.

## Execução

```bash
python -m pip install -r requirements.txt
python analise_wines.py
```

O programa lê o `wine.csv` da mesma pasta e atualiza as tabelas e os gráficos em `resultados/`.

Fonte do conjunto de dados: [Wine dataset no GitHub Gist](https://gist.github.com/tijptjik/9408623).

## Observação estatística

O teste de Shapiro-Wilk rejeita normalidade estrita para o álcool (p = 0,0200), embora a Normal ajustada aproxime as duas probabilidades analisadas. Para o ácido málico, a análise de probabilidades usa a distribuição empírica.
