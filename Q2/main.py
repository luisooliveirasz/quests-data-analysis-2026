import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

SATELLITES_RAW = [
    ("Large Magellanic Cloud",  "05h23m35s", "-69d45m22s",  49.97),
    ("Small Magellanic Cloud",  "00h52m45s", "-72d49m43s",  62.44),
    ("Sagittarius dSph",        "18h55m19s", "-30d32m43s",  26.30),
    ("Fornax dSph",             "02h39m59s", "-34d26m57s", 147.00),
    ("Leo I dSph",              "10h08m28s", "+12d18m23s", 254.00),
    ("Sculptor dSph",           "01h00m09s", "-33d42m33s",  85.90),
    ("Leo II dSph",             "11h13m29s", "+22d09m06s", 233.00),
    ("Sextans dSph",            "10h13m03s", "-01d36m53s",  95.50),
]

# Referencial galactocêntrico
GALCEN_FRAME = Galactocentric(
    galcen_distance=8.178 * u.kpc,
    z_sun=17.0 * u.pc,
)

SEPARATOR = "─" * 100

def build_dataframe(raw: list) -> pd.DataFrame:
    records = []
    for name, ra_hms, dec_dms, d_helio in raw:
        coord = SkyCoord(
            ra=ra_hms,
            dec=dec_dms,
            distance=d_helio * u.kpc,
            frame="icrs",
        )

        gal = coord.galactic
        gc  = coord.galactocentric

        X = gc.x.to(u.kpc).value
        Y = gc.y.to(u.kpc).value
        Z = gc.z.to(u.kpc).value

        records.append({
            "name": name,
            "l_deg": round(gal.l.deg, 4),
            "b_deg": round(gal.b.deg, 4),
            "X_kpc": round(X, 3),
            "Y_kpc": round(Y, 3),
            "Z_kpc": round(Z, 3),
            "d_helio_kpc": round(d_helio, 3),
            "r_gc_kpc": round(np.sqrt(X**2 + Y**2 + Z**2), 3),
        })

    df = pd.DataFrame(records).set_index("name")
    return df


def print_main_table(df: pd.DataFrame) -> None:
    print(f"\n{'GALÁXIA':<30} {'l (°)':>10} {'b (°)':>10} "
          f"{'X (kpc)':>10} {'Y (kpc)':>10} {'Z (kpc)':>10} "
          f"{'d_helio(kpc)':>13} {'r_gc(kpc)':>11}")
    print(SEPARATOR)

    for name, row in df.iterrows():
        print(
            f"{name:<30} {row['l_deg']:>10.4f} {row['b_deg']:>10.4f} "
            f"{row['X_kpc']:>10.2f} {row['Y_kpc']:>10.2f} {row['Z_kpc']:>10.2f} "
            f"{row['d_helio_kpc']:>13.2f} {row['r_gc_kpc']:>11.2f}"
        )
    print()


def print_statistics(df: pd.DataFrame) -> None:
    print("----- ESTATÍSTICAS (distância galactocêntrica r_gc) -----")

    desc = df["r_gc_kpc"].describe()

    idx_min = df["r_gc_kpc"].idxmin()
    idx_max = df["r_gc_kpc"].idxmax()

    print(f"Mais próximo ao GC : {idx_min:30s}  r_gc = {df.loc[idx_min, 'r_gc_kpc']:.2f} kpc")
    print(f"Mais distante do GC: {idx_max:30s}  r_gc = {df.loc[idx_max, 'r_gc_kpc']:.2f} kpc")
    print(f"r_gc médio         : {desc['mean']:.2f} kpc")
    print(f"r_gc mediano       : {desc['50%']:.2f} kpc")
    print(f"Desvio-padrão      : {df['r_gc_kpc'].std(ddof=0):.2f} kpc")
    print(f"|Z| médio (altura acima/abaixo do plano): {df['Z_kpc'].abs().mean():.2f} kpc\n")

    print("----- DESCRIBE COMPLETO (r_gc_kpc) -----")
    print(desc.to_string())
    print()


def compute_centroid_and_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o centróide (Xc, Yc, Zc) e, para cada satélite, o vetor relativo
    ao centróide e o ângulo theta entre o vetor centróide e o vetor relativo.
    Retorna um novo DataFrame com essas colunas extras.
    """
    centroid = df[["X_kpc", "Y_kpc", "Z_kpc"]].mean()
    Xc, Yc, Zc = centroid["X_kpc"], centroid["Y_kpc"], centroid["Z_kpc"]
    norm_centroid = np.linalg.norm([Xc, Yc, Zc])
    centroid_vec  = np.array([Xc, Yc, Zc])

    dX = df["X_kpc"] - Xc
    dY = df["Y_kpc"] - Yc
    dZ = df["Z_kpc"] - Zc

    norm_rel = np.sqrt(dX**2 + dY**2 + dZ**2)

    # Produto escalar centróide · vetor relativo
    dot = dX * Xc + dY * Yc + dZ * Zc

    cos_theta = (dot / (norm_centroid * norm_rel)).clip(-1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    result = df.copy()
    result["dX_kpc"]    = dX.round(3)
    result["dY_kpc"]    = dY.round(3)
    result["dZ_kpc"]    = dZ.round(3)
    result["norm_rel"]  = norm_rel.round(3)
    result["theta_deg"] = theta_deg.round(3)

    return result, centroid, norm_centroid


def print_centroid_and_angles(result: pd.DataFrame, centroid: pd.Series, norm_centroid: float) -> None:
    print("\n----- CENTRÓIDE DAS COORDENADAS GALACTOCÊNTRICAS -----")
    print(f"Xc = {centroid['X_kpc']:.3f} kpc")
    print(f"Yc = {centroid['Y_kpc']:.3f} kpc")
    print(f"Zc = {centroid['Z_kpc']:.3f} kpc")
    print(f"Norma do centróide = {norm_centroid:.3f} kpc")

    print("\n----- VETORES RELATIVOS (ΔX, ΔY, ΔZ) e suas NORMAS -----")
    print(f"{'GALÁXIA':<35} {'ΔX (kpc)':>10} {'ΔY (kpc)':>10} {'ΔZ (kpc)':>10} {'|Δ| (kpc)':>12}")
    print(SEPARATOR)

    for name, row in result.iterrows():
        print(
            f"{name:<35} {row['dX_kpc']:>10.3f} {row['dY_kpc']:>10.3f} "
            f"{row['dZ_kpc']:>10.3f} {row['norm_rel']:>12.3f}"
        )
    print(SEPARATOR)

    angles = result["theta_deg"].dropna()
    print("\n----- ESTATÍSTICAS DOS ÂNGULOS THETA -----")
    print(f"Média: {angles.mean():.1f}°")
    print(f"Mediana: {angles.median():.1f}°")
    print(f"Desvio padrão: {angles.std(ddof=0):.1f}°")
    print(f"Mínimo: {angles.min():.1f}° ({angles.idxmin()})")
    print(f"Máximo: {angles.max():.1f}° ({angles.idxmax()})")


# Ajuste por mínimos quadrados
def fit_plane(df: pd.DataFrame) -> dict:
    x = df["X_kpc"].values
    y = df["Y_kpc"].values
    z = df["Z_kpc"].values

    # Desvios
    dx = x - x.mean()
    dy = y - y.mean()
    dz = z - z.mean()

    # Soma dos desvios
    Sxx = (dx * dx).sum()
    Syy = (dy * dy).sum()
    Sxy = (dx * dy).sum()
    Sxz = (dx * dz).sum()
    Syz = (dy * dz).sum()

    # Coeficientes do plano
    D = Sxx * Syy - Sxy**2
    a = (Sxz * Syy - Syz * Sxy) / D
    b = (Syz * Sxx - Sxz * Sxy) / D
    c = z.mean() - a * x.mean() - b * y.mean()

    inclinacao_deg = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))

    z_pred = a * x + b * y + c
    residuos = z - z_pred
    rms = np.sqrt((residuos**2).mean())

    # DataFrame de resíduos para inspeção
    residuos_df = pd.Series(residuos, index=df.index, name="residuo_kpc").round(4)

    return dict(
        a=a, b=b, c=c,
        inclinacao_deg=inclinacao_deg,
        Sxx=Sxx, Syy=Syy, Sxy=Sxy, Sxz=Sxz, Syz=Syz,
        rms=rms,
        residuos=residuos_df,
    )


def print_plane_fit(df: pd.DataFrame) -> None:
    fit = fit_plane(df)

    print("\n----- AJUSTE DE UM PLANO z = a·x + b·y + c PELOS MÍNIMOS QUADRADOS -----")
    print("Utilizando as coordenadas galactocêntricas (X, Y, Z) dos satélites.")
    print(f"Número de pontos: {len(df)}")

    print("\n--- Somatórios dos desvios ---")
    print(f"Sxx = Σ (x_i - x̄)² = {fit['Sxx']:.3f} kpc²")
    print(f"Syy = Σ (y_i - ȳ)² = {fit['Syy']:.3f} kpc²")
    print(f"Sxy = Σ (x_i - x̄)(y_i - ȳ) = {fit['Sxy']:.3f} kpc²")
    print(f"Sxz = Σ (x_i - x̄)(z_i - z̄) = {fit['Sxz']:.3f} kpc²")
    print(f"Syz = Σ (y_i - ȳ)(z_i - z̄) = {fit['Syz']:.3f} kpc²")

    print("\n--- Coeficientes do plano ---")
    print(f"a (coeficiente de x) = {fit['a']:.6f}")
    print(f"b (coeficiente de y) = {fit['b']:.6f}")
    print(f"c (intercepto) = {fit['c']:.3f} kpc")

    print("\n--- Inclinação do plano em relação ao plano xy ---")
    print(f"i = arctan(√(a²+b²)) = {fit['inclinacao_deg']:.2f}°")
    print(f"RMS dos resíduos verticais: {fit['rms']:.3f} kpc")

    print("\n--- Resíduos por satélite ---")
    print(fit["residuos"].to_string())


# Export em csv
def export_csv(df: pd.DataFrame, path: str = "satellites_gc.csv") -> None:
    df.to_csv(path)
    print(f"\nDados exportados para: {path}")


if __name__ == "__main__":
    df = build_dataframe(SATELLITES_RAW)

    print_main_table(df)

    print_statistics(df)

    result, centroid, norm_centroid = compute_centroid_and_angles(df)
    print_centroid_and_angles(result, centroid, norm_centroid)

    print_plane_fit(df)

    export_csv(df, "satellites_gc.csv")