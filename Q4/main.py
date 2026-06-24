import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import linregress

c = 299793  # km/s

df = pd.read_csv("supernovas.csv")

z = df["z"].to_numpy()
m_b = df["m_B"].to_numpy()

I_SN = (10**(0.2 * m_b)) / (z * c)

resultado = linregress(z, I_SN)

a = resultado.slope
b = resultado.intercept

sigma_a = resultado.stderr
sigma_b = resultado.intercept_stderr

r = resultado.rvalue
R2 = r**2

I_fit = a * z + b

residuos = I_SN - I_fit

rmse = np.sqrt(np.mean(residuos**2))

N = len(z)

print("\n" + "="*60)
print("ANÁLISE DE I_SN × z")
print("="*60)

print(f"Número de supernovas = {N}")

print("\nAJUSTE LINEAR")
print("-"*60)

print(f"a = {a:.6e} ± {sigma_a:.6e}")
print(f"b = {b:.6e} ± {sigma_b:.6e}")

print(f"r  = {r:.6f}")
print(f"R² = {R2:.6f}")

print("\nQUALIDADE DO AJUSTE")
print("-"*60)

print(f"RMSE = {rmse:.6e}")

print("\nESTATÍSTICAS DE I_SN")
print("-"*60)

print(f"Média   = {np.mean(I_SN):.6e}")
print(f"Mediana = {np.median(I_SN):.6e}")
print(f"Desvio  = {np.std(I_SN, ddof=1):.6e}")

print("\nDISTÂNCIA MODULAR ASSOCIADA A b")
print("-"*60)

mu = 5 * np.log10(b)

sigma_mu = 5/(np.log(10)*b) * sigma_b

print(f"μ = {mu:.5f} ± {sigma_mu:.5f}")

print("="*60)

resultado_df = pd.DataFrame({
    "SN": df["SN"],
    "z": z,
    "I_SN": I_SN,
    "I_ajuste": I_fit,
    "residuo": residuos
})

print("\nPrimeiras linhas:")
print(resultado_df.head())

resultado_df.to_csv(
    "resultado_supernovas.csv",
    index=False
)

plt.figure(figsize=(8,6))

plt.scatter(
    z,
    I_SN,
    label="Supernovas"
)

plt.plot(
    np.sort(z),
    a*np.sort(z)+b,
    linewidth=2,
    label="Ajuste linear"
)

plt.xlabel("z")
plt.ylabel("I_SN")
plt.title("I_SN versus z")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_ajuste.png", dpi=300)

plt.figure(figsize=(8,6))

plt.scatter(
    z,
    residuos
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("z")
plt.ylabel("Resíduo")
plt.title("Resíduos do ajuste")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("residuos.png", dpi=300)

plt.show()