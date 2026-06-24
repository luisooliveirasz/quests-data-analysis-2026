import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

np.random.seed(42)

# Parâmetros da distribuição
media = 100
desvio_padrao = 15
n_estrelas = 1000

velocidades_radiais = np.random.normal(media, desvio_padrao, n_estrelas)

N = len(velocidades_radiais)

media_dados = np.mean(velocidades_radiais)
mediana = np.median(velocidades_radiais)

desvio_dados = np.std(velocidades_radiais, ddof=1)
erro_media = desvio_dados / np.sqrt(N)

minimo = np.min(velocidades_radiais)
maximo = np.max(velocidades_radiais)

assimetria = skew(velocidades_radiais)
curtose = kurtosis(velocidades_radiais)

plt.figure(figsize=(10, 6))

n, bins, patches = plt.hist(
    velocidades_radiais,
    bins=20,
    density=False,
    alpha=0.7,
    color='steelblue',
    edgecolor='black',
    label='Dados simulados'
)

x = np.linspace(minimo, maximo, 500)

largura_bin = bins[1] - bins[0]

y = (
    norm.pdf(x, media_dados, desvio_dados)
    * N
    * largura_bin
)

plt.plot(
    x, y,
    'r-',
    linewidth=2,
    label='Normal ajustada'
)

contagens, bins = np.histogram(velocidades_radiais, bins=20)

pontos_medios = (bins[:-1] + bins[1:]) / 2

media_hist = np.sum(contagens * pontos_medios) / np.sum(contagens)

variancia_hist = (
    np.sum(contagens * (pontos_medios - media_hist)**2)
    / np.sum(contagens)
)

desvio_hist = np.sqrt(variancia_hist)

d = np.abs(velocidades_radiais - media_dados)

frac_1sigma = np.sum(d <= desvio_dados) / N * 100
frac_2sigma = np.sum(d <= 2*desvio_dados) / N * 100
frac_3sigma = np.sum(d <= 3*desvio_dados) / N * 100

print("\n" + "="*60)
print("ESTATÍSTICAS DOS DADOS")
print("="*60)

print(f"Número de estrelas        = {N}")
print(f"Média                     = {media_dados:.4f} km/s")
print(f"Erro da média             = {erro_media:.4f} km/s")
print(f"Mediana                   = {mediana:.4f} km/s")
print(f"Desvio padrão             = {desvio_dados:.4f} km/s")
print(f"Mínimo                    = {minimo:.4f} km/s")
print(f"Máximo                    = {maximo:.4f} km/s")

print("\nFORMA DA DISTRIBUIÇÃO")
print("-"*60)
print(f"Assimetria (skewness)     = {assimetria:.4f}")
print(f"Curtose excedente         = {curtose:.4f}")

print("\nESTATÍSTICAS DO HISTOGRAMA")
print("-"*60)
print(f"Média (histograma)        = {media_hist:.4f} km/s")
print(f"Desvio padrão (hist.)     = {desvio_hist:.4f} km/s")

print("\nDIFERENÇAS")
print("-"*60)
print(f"Δ média                  = {media_hist - media_dados:.4f}")
print(f"Δ desvio padrão          = {desvio_hist - desvio_dados:.4f}")

print("\nFRAÇÃO DOS DADOS")
print("-"*60)
print(f"Dentro de 1σ             = {frac_1sigma:.2f}%")
print(f"Dentro de 2σ             = {frac_2sigma:.2f}%")
print(f"Dentro de 3σ             = {frac_3sigma:.2f}%")

print("="*60)

# ==================================================
# Gráfico
# ==================================================

plt.title(
    'Histograma das Velocidades Radiais - Aglomerado Globular',
    fontsize=14
)

plt.xlabel('Velocidade radial (km/s)', fontsize=12)
plt.ylabel('Número de estrelas', fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig('histograma.png', dpi=300)
plt.show()