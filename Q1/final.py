import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from skyfield.api import load
from datetime import datetime, timedelta, timezone

# Carregar efemérides
ts = load.timescale()
eph = load('de440s.bsp')

earth = eph['earth']
sun = eph['sun']
moon = eph['moon']

# 13 pontos de 15 em 15 minutos
start_dt = datetime(2027, 8, 2, 8, 30, 0, tzinfo=timezone.utc)
num_points = 13
step_hours = 1 / 4

gst_deg_list = []
vecs_sun  = np.empty((num_points, 3))
vecs_moon = np.empty((num_points, 3))
separation_angles = []
time_labels = []
datetime_strs = []

print(
    f"{'Tempo (UTC)':<25} |
    {'GST (deg)':>10} |
    {'Objeto':<5} |
    {'X (km)':>14} |
    {'Y (km)':>14} |
    {'Z (km)':>14}"
)
print("-" * 97)

for i in range(num_points):
    current_dt = start_dt + timedelta(hours=i * step_hours)
    t = ts.from_datetime(current_dt)

    gst_deg = t.gast * 15.0
    gst_deg_list.append(gst_deg)

    astro_sun = earth.at(t).observe(sun)
    astro_moon = earth.at(t).observe(moon)

    ra_sun, dec_sun, dist_sun = astro_sun.radec()
    ra_moon, dec_moon, dist_moon = astro_moon.radec()

    sep_deg = astro_sun.separation_from(astro_moon).degrees
    separation_angles.append(sep_deg)
    time_labels.append(current_dt.strftime('%H:%M'))
    datetime_strs.append(current_dt.strftime('%Y-%m-%d %H:%M:%S+00:00'))

    # Coordenadas cartesianas
    ra_rad_sun = ra_sun.hours  * (np.pi / 12.0)
    dec_rad_sun = dec_sun.radians
    ra_rad_moon = ra_moon.hours * (np.pi / 12.0)
    dec_rad_moon = dec_moon.radians

    vecs_sun[i] =
    [
        np.cos(dec_rad_sun) * np.cos(ra_rad_sun) * dist_sun.km,
        np.cos(dec_rad_sun) * np.sin(ra_rad_sun) * dist_sun.km,
        np.sin(dec_rad_sun) * dist_sun.km
    ]

    vecs_moon[i] =
    [
        np.cos(dec_rad_moon) * np.cos(ra_rad_moon) * dist_moon.km,
        np.cos(dec_rad_moon) * np.sin(ra_rad_moon) * dist_moon.km,
        np.sin(dec_rad_moon) * dist_moon.km
    ]

    print(f"{current_dt} | {gst_deg:10.4f} | {'Sol':<5} | "
          f"{vecs_sun[i][0]:14.4e} | {vecs_sun[i][1]:14.4e} | {vecs_sun[i][2]:14.4e}")
    print(f"{current_dt} | {gst_deg:10.4f} | {'Lua':<5} | "
          f"{vecs_moon[i][0]:14.4e} | {vecs_moon[i][1]:14.4e} | {vecs_moon[i][2]:14.4e}")


diff_vectors = vecs_moon - vecs_sun
print("\n" + "="*97)
print("TABELA 1: Diferença Lua - Sol (componentes cartesianas geocêntricas em km)")
print(
    f"{'Tempo (UTC)':<25} |
    {'x_L - x_S (km)':>18} |
    {'y_L - y_S (km)':>18} |
    {'z_L - z_S (km)':>18}"
)
print("-" * 97)
for i in range(num_points):
    print(f"{datetime_strs[i]:<25} |
    {diff_vectors[i,0]:18.4e} |
    {diff_vectors[i,1]:18.4e} |
    {diff_vectors[i,2]:18.4e}"
)


us = diff_vectors.copy()
norms = np.linalg.norm(us, axis=1)
mask = norms > 1e-12
us[mask] /= norms[mask][:, None]
us[~mask] = np.nan

print("\n" + "="*97)
print("TABELA 2: Vetor unitário u = (r_L - r_S) / |r_L - r_S| (adimensional)")
print(f"{'Tempo (UTC)':<25} | {'u_x':>18} | {'u_y':>18} | {'u_z':>18}")
print("-" * 97)

for i in range(num_points):
    if np.any(np.isnan(us[i])):
        print(f"{datetime_strs[i]:<25} | {'NaN':>18} | {'NaN':>18} | {'NaN':>18}")
    else:
        print(f"{datetime_strs[i]:<25} | {us[i,0]:18.6f} | {us[i,1]:18.6f} | {us[i,2]:18.6f}")

# Separação angular via produto escalar
sun_unit = vecs_sun / np.linalg.norm(vecs_sun, axis=1, keepdims=True)
moon_unit = vecs_moon / np.linalg.norm(vecs_moon, axis=1, keepdims=True)
dot_prod = np.sum(sun_unit * moon_unit, axis=1)
dot_prod = np.clip(dot_prod, -1.0, 1.0)
sep_dot = np.arccos(dot_prod) * 180.0 / np.pi

print("\n" + "="*97)
print("TABELA 3: Separação angular Sol–Lua (geocêntrica) via produto escalar")
print(f"{'Tempo (UTC)':<25} | {'Separação angular (graus)':>25}")
print("-" * 97)
for i in range(num_points):
    print(f"{datetime_strs[i]:<25} | {sep_dot[i]:25.6f}")

# Regressão quadrática
time_hours = np.arange(num_points) * step_hours   # [0, 0.25, 0.5, ..., 3.0]
coeffs = np.polyfit(time_hours, sep_dot, deg=2)
A, B, C = coeffs

# Vértice
t0_hours = -B / (2 * A)                # horas desde 08:30
theta_min_fit = C - B**2 / (4 * A)     # graus

t0_seconds = t0_hours * 3600
t0_abs = start_dt + timedelta(seconds=t0_seconds)
t0_abs_str = t0_abs.strftime('%Y-%m-%d %H:%M:%S UTC')

min_idx = np.argmin(sep_dot)
t_min_discreto_hours = time_hours[min_idx]
theta_min_discreto = sep_dot[min_idx]
t_min_discreto_abs = start_dt + timedelta(hours=t_min_discreto_hours)
t_min_discreto_str = t_min_discreto_abs.strftime('%Y-%m-%d %H:%M:%S UTC')

print("\n" + "="*97)
print("AJUSTE PARABÓLICO PARA DETERMINAÇÃO DO MÁXIMO ALINHAMENTO")
print(f"Modelo: θ(t) = a (t - t0)² + θ_min")
print(f"Coeficientes da parábola genérica (θ = A t² + B t + C):")
print(f"A = {A:.6f}  graus/hora²")
print(f"B = {B:.6f}  graus/hora")
print(f"C = {C:.6f}  graus")
print(f"\nInstante de máximo eclipse (vértice):")
print(f"t0 = {t0_hours:.4f} horas após 08:30 UTC  →  {t0_abs_str}")
print(f"θ_min (ajustado) = {theta_min_fit:.6f} graus")
print(f"\nMínimo discreto (ponto da tabela):")
print(f"t_min = {t_min_discreto_hours:.2f} horas após 08:30 UTC  →  {t_min_discreto_str}")
print(f"θ_min (tabela) = {theta_min_discreto:.6f} graus")
print(f"\nDiferença entre os métodos:")
print(f"Δt = {abs(t0_hours - t_min_discreto_hours) * 60:.2f} minutos")
print(f"Δθ = {abs(theta_min_fit - theta_min_discreto):.6f} graus")

# Coeficiente R²
theta_fit = np.polyval(coeffs, time_hours)
ss_res = np.sum((sep_dot - theta_fit)**2)
ss_tot = np.sum((sep_dot - np.mean(sep_dot))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nQualidade do ajuste: R² = {r2:.6f}")

# Regressão quadrática local
time_hours = np.arange(num_points) * step_hours

n_local = 5
indices_local = np.argsort(sep_dot)[:n_local]
indices_local = np.sort(indices_local)
t_local = time_hours[indices_local]
theta_local = sep_dot[indices_local]

coeffs_local = np.polyfit(t_local, theta_local, deg=2)
A_local, B_local, C_local = coeffs_local

# Vértice do ajuste local
t0_local = -B_local / (2 * A_local)
theta_min_local = C_local - B_local**2 / (4 * A_local)

# Converter para UTC
t0_local_abs = start_dt + timedelta(seconds=t0_local * 3600)
t0_local_str = t0_local_abs.strftime('%Y-%m-%d %H:%M:%S UTC')

min_idx_global = np.argmin(sep_dot)
t_min_global = time_hours[min_idx_global]
theta_min_global = sep_dot[min_idx_global]

print("\n" + "="*97)
print("AJUSTE PARABÓLICO LOCAL (5 PONTOS EM TORNO DO MÍNIMO)")
print(f"Pontos utilizados (t em horas após 08:30 UTC):")
for ti, thi in zip(t_local, theta_local):
    print(f"  t = {ti:.2f} h  →  θ = {thi:.6f}°")
print(f"\nInstante de máximo eclipse (vértice local):")
print(f"t0 = {t0_local:.4f} h  →  {t0_local_str}")
print(f"θ_min (ajustado) = {theta_min_local:.6f}°")
print(f"\nMínimo discreto (todos os pontos):")
print(f"t_min = {t_min_global:.2f} h  →  {start_dt + timedelta(hours=t_min_global)}")
print(f"θ_min (tabela) = {theta_min_global:.6f}°")

coeffs_global = np.polyfit(time_hours, sep_dot, deg=2)
A_global, B_global, C_global = coeffs_global
t0_global = -B_global / (2 * A_global)
theta_min_global_fit = C_global - B_global**2 / (4 * A_global)

print(f"\nComparação com ajuste global (13 pontos):")
print(f"t0_global = {t0_global:.4f} h")
print(f"θ_min_global_fit = {theta_min_global_fit:.6f}°")

# Coeficiente k para todos os instantes de tempo
def solve_k(M, u, R):
    if np.any(np.isnan(M)) or np.any(np.isnan(u)):
        return np.nan
    
    a = np.dot(u, u)
    b = 2 * np.dot(M, u)
    c = np.dot(M, M) - R**2
    delta = b**2 - 4*a*c

    if delta < 0:
        return np.nan

    k1 = (-b + np.sqrt(delta)) / (2*a)
    k2 = (-b - np.sqrt(delta)) / (2*a)
    ks = [k for k in (k1, k2) if k > 0]

    return min(ks) if ks else np.nan

R_earth = 6378.0
ks = np.empty(num_points)
for i in range(num_points):
    ks[i] = solve_k(vecs_moon[i], us[i], R_earth)

print("\n" + "="*97)
print("TABELA 4: Parâmetros k (distância da Lua ao ponto de interseção com a Terra, km)")
print(f"{'Tempo (UTC)':<25} | {'k (km)':>18}")
print("-" * 97)
for i in range(num_points):
    if np.isnan(ks[i]):
        print(f"{datetime_strs[i]:<25} | {'NaN':>18}")
    else:
        print(f"{datetime_strs[i]:<25} | {ks[i]:18.4e}")


vecs_earth = vecs_moon + ks[:, None] * us

print("\n" + "="*97)
print("TABELA 5: Coordenadas cartesianas geocêntricas dos pontos de interseção (umbra central)")
print(f"{'Tempo (UTC)':<25} | {'X (km)':>18} | {'Y (km)':>18} | {'Z (km)':>18}")
print("-" * 97)

for i in range(num_points):
    if np.isnan(vecs_earth[i,0]):
        print(f"{datetime_strs[i]:<25} | {'NaN':>18} | {'NaN':>18} | {'NaN':>18}")
    else:
        print(f"{datetime_strs[i]:<25} | {vecs_earth[i,0]:18.4e} | {vecs_earth[i,1]:18.4e} | {vecs_earth[i,2]:18.4e}")

lats, lons = [], []
for i, (x, y, z) in enumerate(vecs_earth):
    if np.isnan(x):
        continue
    lat_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon_rad = np.arctan2(y, x)
    theta_rad = np.deg2rad(gst_deg_list[i])
    lon_rad -= theta_rad
    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)
    lon_deg = (lon_deg + 180) % 360 - 180
    lats.append(lat_deg)
    lons.append(lon_deg)

print("\n" + "="*97)
print("TABELA 6: Coordenadas geográficas dos pontos de interseção (umbra central)")
print(f"{'Tempo (UTC)':<25} | {'Latitude (deg)':>15} | {'Longitude (deg)':>15}")
print("-" * 97)
for i in range(len(lats)):
    print(f"{datetime_strs[i]:<25} | {lats[i]:15.6f} | {lons[i]:15.6f}")

# Mapa do eclipse (finalmente)
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines(linewidth=0.8, color='black')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
             linewidth=0.5, color='gray', alpha=0.7)
ax.scatter(lons, lats, s=40, c='red', edgecolors='black', zorder=5,
           label='Pontos da umbra (central)')
ax.plot(lons, lats, 'b--', linewidth=1.5, label='Trajetória da sombra')
plt.title("Trajetória da sombra do eclipse total\n2 de agosto de 2027", fontsize=14)
plt.legend(loc='lower left')
plt.savefig("mapa_eclipse_2027.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot do ajuste quadrático
plt.figure(figsize=(10, 5))
plt.plot(time_hours, sep_dot, 'o-', color='blue', linewidth=1.5, markersize=6,
         label='Dados calculados')
t_cont = np.linspace(0, max(time_hours), 200)
theta_cont = A * (t_cont - t0_hours)**2 + theta_min_fit
plt.plot(t_cont, theta_cont, 'r--', linewidth=2,
         label=f'Ajuste parabólico: θ = {A:.4f} (t - {t0_hours:.4f})² + {theta_min_fit:.4f}')

plt.xlabel('Tempo decorrido desde 08:30 UTC (horas)')
plt.ylabel('Separação angular (graus)')
plt.title('Distância angular Sol–Lua – Ajuste parabólico para o máximo alinhamento')
plt.grid(True, linestyle='--', alpha=0.6)
plt.plot(t0_hours, theta_min_fit, 'ro', markersize=10,
         label=f'Máximo alinhamento (ajuste):\nt = {t0_hours:.3f} h → {t0_abs_str}\nθ = {theta_min_fit:.4f}°')
plt.legend()
plt.tight_layout()
plt.savefig("separacao_angular_ajuste.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_hours, sep_dot, 'o-', color='blue', linewidth=1.5, markersize=6,
         label='Dados calculados (13 pontos)')
plt.plot(t_local, theta_local, 'ro', markersize=8, label='Pontos do ajuste local (5 menores)')

theta_cont_local = A_local * (t_cont - t0_local)**2 + theta_min_local
plt.plot(t_cont, theta_cont_local, 'r-', linewidth=2,
         label=f'Ajuste local')


plt.xlabel('Tempo decorrido desde 08:30 UTC (horas)')
plt.ylabel('Separação angular (graus)')
plt.title('Ajustes parabólicos para o máximo alinhamento Sol–Lua')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("separacao_angular_ajustes.png", dpi=300, bbox_inches='tight')
plt.show()