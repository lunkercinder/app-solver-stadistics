import math
import os
import json
import statistics
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import openpyxl
try:
    from scipy.stats import norm, t
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Solver Estadístico Unificado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px;}
    .sub-title {font-size: 18px; color: #9CA3AF; margin-bottom: 22px;}
    .metric-card {
        background-color: rgba(120, 120, 120, 0.10);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(150, 150, 150, 0.25);
        margin-bottom: 12px;
        min-height: 135px;
    }
    .report-box {
        background-color: rgba(34, 197, 94, 0.10);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-top: 15px;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 22px;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .section-chip {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.20);
        color: #dbeafe;
        font-size: 0.85rem;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FUNCIONES AUXILIARES GENERALES
# ============================================================

def parse_num(valor):
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)
    s = str(valor).strip().replace(" ", "")
    if s == "":
        raise ValueError("Entrada vacía")
    if s.endswith("%"):
        s = s[:-1]
        s = s.replace(".", "").replace(",", ".")
        return float(s) / 100
    if s.count(".") >= 2 and "," not in s:
        s = s.replace(".", "")
        return float(s)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def phi(z):
    if SCIPY_OK:
        return norm.cdf(z)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def inv_norm(p):
    if SCIPY_OK:
        return norm.ppf(p)
    low, high = -5, 5
    for _ in range(100):
        mid = (low + high) / 2
        if phi(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def fmt_prob(p):
    if p < 0.000001:
        return "< 0.000001 (<0.0001%)"
    return f"{p:.6f} ({p*100:.2f}%)"


def interpretar_correlacion(r):
    if r > 0:
        tipo = "positiva"
    elif r < 0:
        tipo = "negativa"
    else:
        tipo = "nula"

    fuerza = abs(r)
    if fuerza == 1:
        intensidad = "perfecta"
    elif fuerza >= 0.80:
        intensidad = "muy fuerte"
    elif fuerza >= 0.60:
        intensidad = "fuerte"
    elif fuerza >= 0.40:
        intensidad = "moderada"
    elif fuerza >= 0.20:
        intensidad = "débil"
    else:
        intensidad = "muy débil o casi inexistente"
    return tipo, intensidad


def p_valor_normal(z, tipo):
    if tipo == "Bilateral":
        return 2 * (1 - phi(abs(z)))
    if tipo == "Cola izquierda":
        return phi(z)
    return 1 - phi(z)


def p_valor_t(t_calc, gl, tipo):
    if SCIPY_OK:
        if tipo == "Bilateral":
            return 2 * (1 - t.cdf(abs(t_calc), gl))
        if tipo == "Cola izquierda":
            return t.cdf(t_calc, gl)
        return 1 - t.cdf(t_calc, gl)
    return p_valor_normal(t_calc, tipo)


def critico_z(alpha, tipo):
    if tipo == "Bilateral":
        return abs(inv_norm(1 - alpha / 2))
    return abs(inv_norm(1 - alpha))


def critico_t(alpha, gl, tipo):
    if SCIPY_OK:
        if tipo == "Bilateral":
            return t.ppf(1 - alpha / 2, gl)
        return t.ppf(1 - alpha, gl)
    return critico_z(alpha, tipo)


def conclusion_hipotesis(rechaza, alpha, contexto, h1_texto):
    nivel = (1 - alpha) * 100
    if rechaza:
        return (
            f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, "
            f"se rechaza la hipótesis nula. Por tanto, existe evidencia estadística suficiente para afirmar que "
            f"{h1_texto}. En el contexto del problema, esto indica que {contexto}"
        )
    return (
        f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, "
        f"no se rechaza la hipótesis nula. Por tanto, no existe evidencia estadística suficiente para afirmar que "
        f"{h1_texto}. En el contexto del problema, esto indica que {contexto}"
    )


def show_report(texto):
    st.markdown("### Conclusión para informe")
    st.markdown(f"<div class='report-box'>{texto}</div>", unsafe_allow_html=True)
    st.download_button(
        "Descargar conclusión en TXT",
        data=texto,
        file_name="conclusion_estadistica.txt",
        mime="text/plain"
    )


def leer_csv_excel(archivo):
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        try:
            return pd.read_csv(archivo, sep=";")
        except Exception:
            archivo.seek(0)
            try:
                return pd.read_csv(archivo, sep=",")
            except Exception:
                archivo.seek(0)
                return pd.read_csv(archivo)
    if nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        return pd.read_excel(archivo)
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def plot_heatmap(matriz):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matriz, aspect="auto")
    ax.set_xticks(range(len(matriz.columns)))
    ax.set_yticks(range(len(matriz.columns)))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right")
    ax.set_yticklabels(matriz.columns)
    for i in range(len(matriz.columns)):
        for j in range(len(matriz.columns)):
            ax.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de correlación de Pearson")
    plt.tight_layout()
    st.pyplot(fig)


def render_section_header(title, subtitle, icon="📊"):
    st.markdown(
        f"""
        <div class="section-chip">Módulo activo</div>
        <div style="display: flex; align-items: center; gap: 18px; margin-bottom: 12px;">
            <div style="font-size: 2rem; background: rgba(56, 189, 248, 0.10); padding: 10px 14px; border-radius: 16px;">{icon}</div>
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">{title}</h1>
                <p style="margin: 0; color: #64748b;">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_topic_selector(options, key):
    st.markdown("### Tema")
    return st.selectbox("Selecciona el tema a trabajar", options, key=key, label_visibility="collapsed")


def render_three_inputs(labels_defaults):
    cols = st.columns(3)
    values = []
    for col, (label, default) in zip(cols, labels_defaults):
        with col:
            values.append(parse_num(st.text_input(label, str(default))))
    return values


def render_four_inputs(labels_defaults):
    cols = st.columns(4)
    values = []
    for col, (label, default) in zip(cols, labels_defaults):
        with col:
            values.append(parse_num(st.text_input(label, str(default))))
    return values


def render_result_box(title, value):
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.16), rgba(56, 189, 248, 0.04));
                    border: 1px solid rgba(56, 189, 248, 0.25); padding: 18px; border-radius: 16px; margin-top: 12px; margin-bottom: 16px;">
            <div style="font-size: 0.95rem; color: #93c5fd; font-weight: 700; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 1.45rem; font-weight: 800;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FUNCIONES ACADÉMICAS ADICIONALES
# ============================================================

def calcular_moda(datos):
    conteos = Counter(datos)
    max_frecuencia = max(conteos.values())
    modas = [k for k, v in conteos.items() if v == max_frecuencia]
    return modas


def calcular_percentil(datos_ordenados, p):
    n = len(datos_ordenados)
    if n == 0:
        return 0.0
    if n == 1:
        return datos_ordenados[0]
    idx = (p / 100) * (n - 1)
    idx_inf = int(math.floor(idx))
    idx_sup = int(math.ceil(idx))
    peso = idx - idx_inf
    if idx_inf == idx_sup:
        return datos_ordenados[idx_inf]
    return datos_ordenados[idx_inf] * (1 - peso) + datos_ordenados[idx_sup] * peso


def permutacion_lineal_sin_rep(n):
    return math.factorial(n)


def permutacion_con_rep_identicos(n, frecuencias):
    denominador = 1
    for f in frecuencias:
        denominador *= math.factorial(f)
    return math.factorial(n) // denominador


def variacion_sin_rep(n, r):
    return math.factorial(n) // math.factorial(n - r) if r <= n else 0


def variacion_con_rep(n, r):
    return n ** r


def combinacion_sin_rep(n, r):
    return math.comb(n, r) if r <= n else 0


def combinacion_con_rep(n, r):
    return math.comb(n + r - 1, r)


def funcion_error_aprox(x):
    signo = 1 if x >= 0 else -1
    x = abs(x)
    t_aux = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t_aux - 1.453152027) * t_aux) + 1.421413741) * t_aux - 0.284496736) * t_aux + 0.254829592) * t_aux * math.exp(-x * x)
    return signo * y


def cdf_normal_estandar(z):
    return 0.5 * (1.0 + funcion_error_aprox(z / math.sqrt(2)))


def pdf_normal(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def probabilidad_binomial(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def probabilidad_poisson(k, lam):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def get_z_value(confianza):
    alfa = 1.0 - confianza
    p = 1 - alfa / 2
    t_aux = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t_aux - ((c2*t_aux + c1)*t_aux + c0) / (((d3*t_aux + d2)*t_aux + d1)*t_aux + 1.0)
    return z


def get_t_value(confianza, df):
    z = get_z_value(confianza)
    if df >= 30:
        return z
    return z + (z**3 + z) / (4.0 * df)


def registrar_calculo(modulo, inputs_dict):
    try:
        log_entry = {
            "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modulo": modulo,
            "inputs": inputs_dict
        }
        with open("historial_calculos.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        return True
    except Exception:
        return False

# ============================================================
# NAVEGACIÓN
# ============================================================

MENU = [
    "Inicio",
    "Indicadores Estadísticos",
    "Reglas de Conteo",
    "Teorema de Bayes",
    "Variable Aleatoria",
    "Distribuciones Probabilísticas",
    "Distribuciones de muestreo",
    "Intervalos de confianza",
    "Intervalos de Confianza_2",
    "Pruebas de hipótesis",
    "Correlación",
    "Base de datos",  
    "Fórmulas",
    "Historial de Cálculos"
]

st.sidebar.title("📊 Menú principal")
categoria = st.sidebar.radio("Sección", MENU)

st.markdown("<div class='main-title'>Solver Estadístico Unificado</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Módulos académicos, distribuciones de muestreo, pruebas de hipótesis, evaluación y análisis de bases de datos.</div>",
    unsafe_allow_html=True
)

# ============================================================
# INICIO
# ============================================================

if categoria == "Inicio":
    render_section_header(
        "Solver Estadístico Unificado",
        "Plataforma unificada para análisis estadístico, muestreo, hipótesis, conteo, correlación y exploración de datos.",
        "🏠"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='metric-card'><h3>📘 Módulos principales</h3><p>Distribuciones de muestreo, intervalos de confianza, pruebas de hipótesis, correlación, base de datos y fórmulas.</p></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            "<div class='metric-card'><h3>🧠 Módulos académicos adicionales</h3><p>Indicadores estadísticos, reglas de conteo, teorema de Bayes, variable aleatoria, distribuciones probabilísticas e historial.</p></div>",
            unsafe_allow_html=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            "<div class='metric-card'><h3>📂 Trabajo con archivos</h3><p>Puedes subir archivos CSV o Excel para análisis, correlaciones, exploración básica e indicadores con gráficos.</p></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            "<div class='metric-card'><h3>📝 Salidas para informe</h3><p>El sistema genera conclusiones descargables y resultados listos para usar en trabajos académicos.</p></div>",
            unsafe_allow_html=True
        )

    st.info("Usa el menú lateral para navegar por todos los módulos del sistema.")

# ============================================================
# DISTRIBUCIONES DE MUESTREO
# ============================================================

elif categoria == "Distribuciones de muestreo":
    render_section_header(
        "Distribuciones de muestreo",
        "Media muestral y proporción muestral con una estructura uniforme de entrada y salida.",
        "📌"
    )
    tema = render_topic_selector(["Media muestral", "Proporción muestral"], "tema_dist_muest")

    if tema == "Media muestral":
        st.markdown("### Datos de entrada")
        mu, sigma, n = render_three_inputs([
            ("μ media poblacional", 50),
            ("σ desviación poblacional", 10),
            ("n tamaño de muestra", 36)
        ])
        n = int(n)

        tipo = st.selectbox("Tipo de cálculo", ["P(X̄ < c)", "P(X̄ > c)", "P(a < X̄ < b)"], key="dm_media_tipo")
        se = sigma / math.sqrt(n)
        st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
        render_result_box("Error estándar", f"{se:.6f}")

        if tipo in ["P(X̄ < c)", "P(X̄ > c)"]:
            c = parse_num(st.text_input("Valor c", "52", key="dm_media_c"))
            if st.button("Calcular", key="dm_media_btn_1"):
                z = (c - mu) / se
                p_menor = phi(z)
                st.latex(r"z = \frac{c-\mu}{SE}")
                st.write(f"z = {z:.6f}")
                if tipo == "P(X̄ < c)":
                    st.success(f"P(X̄ < {c}) = {fmt_prob(p_menor)}")
                    show_report(f"La probabilidad de que la media muestral sea menor que {c} es aproximadamente {fmt_prob(p_menor)}.")
                else:
                    prob = 1 - p_menor
                    st.success(f"P(X̄ > {c}) = {fmt_prob(prob)}")
                    show_report(f"La probabilidad de que la media muestral sea mayor que {c} es aproximadamente {fmt_prob(prob)}.")
        else:
            a, b = render_three_inputs([
                ("Límite inferior a", 45),
                ("Límite superior b", 55),
                ("Campo auxiliar", 0)
            ])[:2]
            if st.button("Calcular", key="dm_media_btn_2"):
                z_a = (a - mu) / se
                z_b = (b - mu) / se
                prob = phi(z_b) - phi(z_a)
                st.write(f"z(a) = {z_a:.6f}")
                st.write(f"z(b) = {z_b:.6f}")
                st.success(f"P({a} < X̄ < {b}) = {fmt_prob(prob)}")
                show_report(f"La probabilidad de que la media muestral esté entre {a} y {b} es aproximadamente {fmt_prob(prob)}.")

    else:
        st.markdown("### Datos de entrada")
        p, n = render_three_inputs([
            ("p proporción poblacional", "17%"),
            ("n tamaño de muestra", 400),
            ("Campo auxiliar", 0)
        ])[:2]
        n = int(n)

        tipo = st.selectbox("Tipo de cálculo", ["P(p̂ < c)", "P(p̂ > c)", "P(a < p̂ < b)"], key="dm_prop_tipo")
        se = math.sqrt(p * (1 - p) / n)
        st.latex(r"SE = \sqrt{\frac{p(1-p)}{n}}")
        render_result_box("Error estándar", f"{se:.6f}")
        st.write(f"n = {n}")
        st.write(f"n·p = {n*p:.4f}")
        if n >= 30 and n * p >= 5:
            st.success("Se cumple la aproximación normal usada normalmente en clase.")
        else:
            st.warning("No se cumple completamente la regla n ≥ 30 y n·p ≥ 5.")

        if tipo in ["P(p̂ < c)", "P(p̂ > c)"]:
            c = parse_num(st.text_input("Valor c", "20%", key="dm_prop_c"))
            if st.button("Calcular", key="dm_prop_btn_1"):
                z = (c - p) / se
                p_menor = phi(z)
                st.write(f"z = {z:.6f}")
                if tipo == "P(p̂ < c)":
                    st.success(f"P(p̂ < {c:.4f}) = {fmt_prob(p_menor)}")
                    show_report(f"La probabilidad de que la proporción muestral sea menor que {c:.4f} es aproximadamente {fmt_prob(p_menor)}.")
                else:
                    prob = 1 - p_menor
                    st.success(f"P(p̂ > {c:.4f}) = {fmt_prob(prob)}")
                    show_report(f"La probabilidad de que la proporción muestral sea mayor que {c:.4f} es aproximadamente {fmt_prob(prob)}.")
        else:
            a, b = render_three_inputs([
                ("Límite inferior a", "10%"),
                ("Límite superior b", "20%"),
                ("Campo auxiliar", 0)
            ])[:2]
            if st.button("Calcular", key="dm_prop_btn_2"):
                z_a = (a - p) / se
                z_b = (b - p) / se
                prob = phi(z_b) - phi(z_a)
                st.write(f"z(a) = {z_a:.6f}")
                st.write(f"z(b) = {z_b:.6f}")
                st.success(f"P({a:.4f} < p̂ < {b:.4f}) = {fmt_prob(prob)}")
                show_report(f"La probabilidad de que la proporción muestral esté entre {a:.4f} y {b:.4f} es aproximadamente {fmt_prob(prob)}.")

# ============================================================
# INTERVALOS DE CONFIANZA
# ============================================================

elif categoria == "Intervalos de confianza":
    render_section_header(
        "Intervalos de confianza",
        "Misma lógica visual para media con Z, media con t y proporción con Z.",
        "📏"
    )
    tema = render_topic_selector(["Media con Z", "Media con t", "Proporción con Z"], "tema_ic")

    if tema in ["Media con Z", "Media con t"]:
        nc, xbar, desv, n = render_four_inputs([
            ("Nivel de confianza", "95%"),
            ("x̄", 50),
            (("σ poblacional" if tema == "Media con Z" else "s muestral"), 10),
            ("n", 36 if tema == "Media con Z" else 25)
        ])
        n = int(n)
        if nc > 1:
            nc /= 100

        if st.button("Calcular intervalo", key=f"ic_btn_{tema}"):
            alpha = 1 - nc
            if tema == "Media con Z":
                crit = abs(inv_norm(1 - alpha / 2))
                gl_txt = "No aplica"
            else:
                crit = critico_t(alpha, n - 1, "Bilateral")
                gl_txt = str(n - 1)
            se = desv / math.sqrt(n)
            margen = crit * se
            li, ls = xbar - margen, xbar + margen
            st.write(f"gl = {gl_txt}")
            st.write(f"Valor crítico = {crit:.4f}")
            st.write(f"SE = {se:.6f}")
            st.write(f"Margen = {margen:.6f}")
            st.success(f"IC {nc*100:.2f}% para μ = ({li:.6f}, {ls:.6f})")
            show_report(f"Con un nivel de confianza del {nc*100:.2f}%, el intervalo de confianza para la media poblacional es ({li:.6f}, {ls:.6f}).")

    else:
        nc = parse_num(st.text_input("Nivel de confianza", "95%", key="ic_prop_nc"))
        if nc > 1:
            nc /= 100

        modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"], horizontal=True, key="ic_prop_modo")

        if modo == "Ya tengo p̂":
            phat, n = render_three_inputs([
                ("p̂", 0.50),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
        else:
            x, n = render_three_inputs([
                ("x éxitos", 200),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
            phat = x / n
        n = int(n)

        if st.button("Calcular intervalo", key="ic_prop_z"):
            alpha = 1 - nc
            z = abs(inv_norm(1 - alpha / 2))
            se = math.sqrt(phat * (1 - phat) / n)
            margen = z * se
            li, ls = phat - margen, phat + margen
            st.write(f"p̂ = {phat:.6f}")
            st.write(f"z* = {z:.4f}")
            st.write(f"SE = {se:.6f}")
            st.success(f"IC {nc*100:.2f}% para p = ({li:.6f}, {ls:.6f})")
            show_report(f"Con un nivel de confianza del {nc*100:.2f}%, el intervalo de confianza para la proporción poblacional es ({li:.6f}, {ls:.6f}).")

# ============================================================
# PRUEBAS DE HIPÓTESIS
# ============================================================

elif categoria == "Pruebas de hipótesis":
    render_section_header(
        "Pruebas de hipótesis",
        "Todas las pruebas comparten la misma estructura visual: tema, datos, tipo de prueba, cálculo y conclusión.",
        "🧪"
    )
    tema = render_topic_selector(["Media con Z", "Media con t", "Proporción con Z", "Diferencia de medias", "Diferencia de proporciones"], "tema_ph")

    if tema in ["Media con Z", "Media con t"]:
        mu0, xbar, desv, n = render_four_inputs([
            ("μ₀", 50),
            ("x̄", 53),
            (("σ poblacional" if tema == "Media con Z" else "s muestral"), 12),
            ("n", 36 if tema == "Media con Z" else 25)
        ])
        n = int(n)
        alpha = parse_num(st.text_input("α", "0.05", key=f"ph_alpha_{tema}"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key=f"ph_tipo_{tema}")
        contexto = st.text_area("Contexto para la conclusión", "la media observada debe analizarse frente al valor de referencia planteado.", key=f"ph_ctx_{tema}")

        if st.button("Resolver prueba", key=f"ph_btn_{tema}"):
            se = desv / math.sqrt(n)
            estad = (xbar - mu0) / se
            if tema == "Media con Z":
                p_value = p_valor_normal(estad, tipo)
                crit = critico_z(alpha, tipo)
                nombre = "z"
                gl_txt = "No aplica"
            else:
                gl = n - 1
                p_value = p_valor_t(estad, gl, tipo)
                crit = critico_t(alpha, gl, tipo)
                nombre = "t"
                gl_txt = str(gl)
            rechaza = p_value < alpha
            st.write(f"gl = {gl_txt}")
            st.write(f"SE = {se:.6f}")
            st.write(f"{nombre} calculado = {estad:.6f}")
            st.write(f"Valor crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            st.write(f"α = {alpha:.4f}")
            st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
            h1 = (
                "la media poblacional es diferente al valor de referencia"
                if tipo == "Bilateral"
                else ("la media poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la media poblacional es mayor al valor de referencia")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    elif tema == "Proporción con Z":
        p0 = parse_num(st.text_input("p₀", "17%", key="ph_prop_p0"))
        alpha = parse_num(st.text_input("α", "0.05", key="ph_prop_alpha"))
        modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"], horizontal=True, key="ph_prop_modo")
        if modo == "Ya tengo p̂":
            phat, n = render_three_inputs([
                ("p̂", 0.20),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
        else:
            x, n = render_three_inputs([
                ("x éxitos", 80),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
            phat = x / n
        n = int(n)
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="ph_prop_tipo")
        contexto = st.text_area("Contexto para la conclusión", "la proporción observada debe compararse con el porcentaje establecido como referencia.", key="ph_prop_ctx")

        if st.button("Resolver prueba", key="ph_prop_btn"):
            se = math.sqrt(p0 * (1 - p0) / n)
            z = (phat - p0) / se
            p_value = p_valor_normal(z, tipo)
            crit = critico_z(alpha, tipo)
            rechaza = p_value < alpha
            st.write(f"p̂ = {phat:.6f}")
            st.write(f"SE = {se:.6f}")
            st.write(f"z calculado = {z:.6f}")
            st.write(f"Valor crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
            h1 = (
                "la proporción poblacional es diferente al valor de referencia"
                if tipo == "Bilateral"
                else ("la proporción poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la proporción poblacional es mayor al valor de referencia")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    elif tema == "Diferencia de medias":
        metodo = st.selectbox("Método", ["Muestras grandes con Z", "t pooled varianzas iguales", "t Welch varianzas diferentes", "Muestras pareadas"], key="ph_difm_met")
        alpha = parse_num(st.text_input("α", "0.05", key="difm_alpha"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="difm_tipo")
        d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0", key="difm_d0"))
        contexto = st.text_area("Contexto para la conclusión", "los promedios de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.", key="difm_ctx")

        if metodo != "Muestras pareadas":
            st.markdown("### Datos de entrada")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Grupo 1")
                xbar1 = parse_num(st.text_input("x̄₁", "42.48", key="difm_x1"))
                s1 = parse_num(st.text_input("s₁", "7.19", key="difm_s1"))
                n1 = int(parse_num(st.text_input("n₁", "205", key="difm_n1")))
            with c2:
                st.markdown("#### Grupo 2")
                xbar2 = parse_num(st.text_input("x̄₂", "43.04", key="difm_x2"))
                s2 = parse_num(st.text_input("s₂", "7.47", key="difm_s2"))
                n2 = int(parse_num(st.text_input("n₂", "195", key="difm_n2")))

            if st.button("Resolver prueba", key="dif_medias"):
                if metodo == "Muestras grandes con Z":
                    se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
                    estad = ((xbar1 - xbar2) - d0) / se
                    p_value = p_valor_normal(estad, tipo)
                    crit = critico_z(alpha, tipo)
                    nombre = "z"
                    gl_txt = "No aplica"
                elif metodo == "t pooled varianzas iguales":
                    gl = n1 + n2 - 2
                    sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / gl
                    sp = math.sqrt(sp2)
                    se = sp * math.sqrt((1/n1) + (1/n2))
                    estad = ((xbar1 - xbar2) - d0) / se
                    p_value = p_valor_t(estad, gl, tipo)
                    crit = critico_t(alpha, gl, tipo)
                    nombre = "t"
                    gl_txt = str(gl)
                else:
                    se = math.sqrt((s1**2/n1) + (s2**2/n2))
                    estad = ((xbar1 - xbar2) - d0) / se
                    num = ((s1**2/n1) + (s2**2/n2))**2
                    den = ((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1)
                    gl = num / den
                    p_value = p_valor_t(estad, gl, tipo)
                    crit = critico_t(alpha, gl, tipo)
                    nombre = "t"
                    gl_txt = f"{gl:.4f}"
                rechaza = p_value < alpha
                st.write(f"SE = {se:.6f}")
                st.write(f"{nombre} calculado = {estad:.6f}")
                st.write(f"gl = {gl_txt}")
                st.write(f"Valor crítico = {crit:.4f}")
                st.write(f"p-valor = {p_value:.6f}")
                st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
                h1 = (
                    "existe diferencia entre las medias poblacionales"
                    if tipo == "Bilateral"
                    else ("la media del grupo 1 es menor que la media del grupo 2" if tipo == "Cola izquierda" else "la media del grupo 1 es mayor que la media del grupo 2")
                )
                show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))
        else:
            st.write("Ingresa datos separados por coma. La diferencia se calcula como después - antes.")
            antes_txt = st.text_area("Antes", "10,12,15,13,11", key="difm_antes")
            despues_txt = st.text_area("Después", "12,15,16,15,13", key="difm_despues")
            if st.button("Resolver prueba pareada", key="dif_pareada"):
                antes = np.array([parse_num(v) for v in antes_txt.split(",") if v.strip() != ""])
                despues = np.array([parse_num(v) for v in despues_txt.split(",") if v.strip() != ""])
                if len(antes) != len(despues):
                    st.error("Las dos listas deben tener la misma cantidad de datos.")
                elif len(antes) < 2:
                    st.error("Se necesitan mínimo dos pares de datos.")
                else:
                    d = despues - antes
                    n = len(d)
                    dbar = np.mean(d)
                    sd = np.std(d, ddof=1)
                    se = sd / math.sqrt(n)
                    gl = n - 1
                    estad = (dbar - d0) / se
                    p_value = p_valor_t(estad, gl, tipo)
                    crit = critico_t(alpha, gl, tipo)
                    rechaza = p_value < alpha
                    st.write(f"n = {n}")
                    st.write(f"Promedio de diferencias = {dbar:.6f}")
                    st.write(f"s_d = {sd:.6f}")
                    st.write(f"SE = {se:.6f}")
                    st.write(f"t calculado = {estad:.6f}")
                    st.write(f"gl = {gl}")
                    st.write(f"Valor crítico = {crit:.4f}")
                    st.write(f"p-valor = {p_value:.6f}")
                    st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
                    h1 = (
                        "existe diferencia significativa entre las mediciones pareadas"
                        if tipo == "Bilateral"
                        else ("la diferencia promedio es menor que el valor hipotético" if tipo == "Cola izquierda" else "la diferencia promedio es mayor que el valor hipotético")
                    )
                    show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    else:
        alpha = parse_num(st.text_input("α", "0.05", key="difp_alpha"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="difp_tipo")
        d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0", key="difp_d0"))
        contexto = st.text_area("Contexto para la conclusión", "las proporciones de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.", key="difp_ctx")

        st.markdown("### Datos de entrada")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Grupo 1")
            x1 = int(parse_num(st.text_input("x₁ éxitos", "80", key="difp_x1")))
            n1 = int(parse_num(st.text_input("n₁", "205", key="difp_n1")))
        with c2:
            st.markdown("#### Grupo 2")
            x2 = int(parse_num(st.text_input("x₂ éxitos", "90", key="difp_x2")))
            n2 = int(parse_num(st.text_input("n₂", "195", key="difp_n2")))

        if st.button("Resolver prueba", key="dif_prop"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            q_pool = 1 - p_pool
            se = math.sqrt(p_pool * q_pool * ((1/n1) + (1/n2)))
            z = ((p1 - p2) - d0) / se
            p_value = p_valor_normal(z, tipo)
            crit = critico_z(alpha, tipo)
            rechaza = p_value < alpha
            st.write(f"p̂₁ = {p1:.6f}")
            st.write(f"p̂₂ = {p2:.6f}")
            st.write(f"p combinada = {p_pool:.6f}")
            st.write(f"SE = {se:.6f}")
            st.write(f"z calculado = {z:.6f}")
            st.write(f"Valor crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
            h1 = (
                "existe diferencia entre las proporciones poblacionales"
                if tipo == "Bilateral"
                else ("la proporción del grupo 1 es menor que la proporción del grupo 2" if tipo == "Cola izquierda" else "la proporción del grupo 1 es mayor que la proporción del grupo 2")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

# ============================================================
# CORRELACIÓN
# ============================================================

elif categoria == "Correlación":
    render_section_header(
        "Correlación de Pearson",
        "Misma estructura para trabajo manual o con archivo.",
        "📈"
    )
    tema = render_topic_selector(["Datos manuales", "Archivo CSV o Excel"], "tema_corr")

    if tema == "Datos manuales":
        col1, col2 = st.columns(2)
        with col1:
            nombre_x = st.text_input("Nombre de X", "X")
            datos_x = st.text_area("Valores de X separados por coma", "1,2,3,4,5")
        with col2:
            nombre_y = st.text_input("Nombre de Y", "Y")
            datos_y = st.text_area("Valores de Y separados por coma", "2,4,6,8,10")

        if st.button("Calcular correlación", key="corr_manual"):
            try:
                x = np.array([parse_num(v) for v in datos_x.split(",") if v.strip() != ""])
                y = np.array([parse_num(v) for v in datos_y.split(",") if v.strip() != ""])
                if len(x) != len(y):
                    st.error("X y Y deben tener la misma cantidad de datos.")
                elif len(x) < 2:
                    st.error("Se necesitan mínimo 2 datos.")
                else:
                    r = np.corrcoef(x, y)[0, 1]
                    tipo_corr, intensidad = interpretar_correlacion(r)
                    st.success(f"Coeficiente de correlación r = {r:.4f}")
                    st.write(f"Correlación {tipo_corr} {intensidad} entre {nombre_x} y {nombre_y}.")
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(x, y)
                    ax.set_xlabel(nombre_x)
                    ax.set_ylabel(nombre_y)
                    ax.set_title(f"Diagrama de dispersión: {nombre_x} vs {nombre_y}")
                    st.pyplot(fig)
                    if tipo_corr == "positiva":
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo_corr} {intensidad}. A medida que aumenta {nombre_x}, también tiende a aumentar {nombre_y}."
                    elif tipo_corr == "negativa":
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo_corr} {intensidad}. A medida que aumenta {nombre_x}, {nombre_y} tiende a disminuir."
                    else:
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, por lo que no se observa una relación lineal clara entre {nombre_x} y {nombre_y}."
                    show_report(reporte)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="corr_file")
        if archivo is not None:
            try:
                df = leer_csv_excel(archivo)
                st.success("Base cargada correctamente.")
                st.write(f"Filas: {df.shape[0]}")
                st.write(f"Columnas: {df.shape[1]}")
                st.dataframe(df.head())
                numericas = df.select_dtypes(include=["int64", "float64"])
                for col in ["ID", "Id", "id", "Registro", "registro"]:
                    if col in numericas.columns:
                        numericas = numericas.drop(columns=[col])
                if numericas.shape[1] < 2:
                    st.error("La base debe tener mínimo dos columnas numéricas.")
                else:
                    st.write("Variables numéricas utilizadas:")
                    st.write(list(numericas.columns))
                    matriz = numericas.corr(method="pearson")
                    st.subheader("Matriz de correlación")
                    st.dataframe(matriz.round(4))
                    plot_heatmap(matriz)
                    st.subheader("Interpretación automática")
                    textos = []
                    for i in range(len(matriz.columns)):
                        for j in range(i + 1, len(matriz.columns)):
                            var1, var2 = matriz.columns[i], matriz.columns[j]
                            r = matriz.iloc[i, j]
                            tipo_corr, intensidad = interpretar_correlacion(r)
                            linea = f"{var1} y {var2}: r = {r:.4f} → correlación {tipo_corr} {intensidad}."
                            st.write(linea)
                            textos.append(linea)
                    st.download_button(
                        "Descargar interpretación",
                        data="\n".join(textos),
                        file_name="interpretacion_correlaciones.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

# ============================================================
# BASE DE DATOS
# ============================================================

elif categoria == "Base de datos":
    render_section_header(
        "Base de datos",
        "Exploración básica o comparación por grupo con estructura uniforme de tema.",
        "📂"
    )
    tema = render_topic_selector(["Exploración básica", "Comparación por grupo"], "tema_bd")

    if tema == "Exploración básica":
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="bd_explo")
        if archivo is not None:
            try:
                df = leer_csv_excel(archivo)
                st.success("Base cargada correctamente.")
                st.write(f"Filas: {df.shape[0]}")
                st.write(f"Columnas: {df.shape[1]}")
                st.markdown("### Vista previa")
                st.dataframe(df.head())
                st.markdown("### Tipos de datos")
                st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipo"]))
                st.markdown("### Valores faltantes")
                faltantes = df.isna().sum().reset_index()
                faltantes.columns = ["Variable", "Faltantes"]
                st.dataframe(faltantes)
                numericas = df.select_dtypes(include=["int64", "float64"])
                if numericas.shape[1] > 0:
                    st.markdown("### Estadística descriptiva")
                    st.dataframe(numericas.describe().T)
                    variable = st.selectbox("Variable para histograma", numericas.columns)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(numericas[variable].dropna(), bins=20)
                    ax.set_title(f"Histograma de {variable}")
                    ax.set_xlabel(variable)
                    ax.set_ylabel("Frecuencia")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="bd_grupo")
        if archivo is not None:
            try:
                df = leer_csv_excel(archivo)
                st.success("Base cargada correctamente.")
                st.dataframe(df.head())
                columnas = list(df.columns)
                col_grupo = st.selectbox("Variable de grupo", columnas)
                numericas = df.select_dtypes(include=["int64", "float64"]).columns
                col_num = st.selectbox("Variable numérica", numericas)
                if st.button("Comparar", key="comparar_bd"):
                    resumen = df.groupby(col_grupo)[col_num].agg(["count", "mean", "std", "min", "max"])
                    st.dataframe(resumen)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    grupos = [grupo[col_num].dropna().values for _, grupo in df.groupby(col_grupo)]
                    labels = [str(g) for g in df.groupby(col_grupo).groups.keys()]
                    ax.boxplot(grupos, tick_labels=labels)
                    ax.set_title(f"{col_num} por {col_grupo}")
                    ax.set_xlabel(col_grupo)
                    ax.set_ylabel(col_num)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

# ============================================================
# FÓRMULAS
# ============================================================

elif categoria == "Fórmulas":
    render_section_header(
        "Formulario completo del programa",
        "Resumen de las fórmulas usadas en todos los cálculos del sistema.",
        "📚"
    )

    st.markdown("## 1. Distribuciones de muestreo")
    st.markdown("### Media muestral")
    st.latex(r"\mu_{\bar{x}} = \mu")
    st.latex(r"SE_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")
    st.latex(r"z = \frac{\bar{x}-\mu}{\sigma/\sqrt{n}}")
    st.latex(r"P(\bar{X}<c), \quad P(\bar{X}>c), \quad P(a<\bar{X}<b)")

    st.markdown("### Proporción muestral")
    st.latex(r"\mu_{\hat{p}} = p")
    st.latex(r"SE_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
    st.latex(r"z = \frac{\hat{p}-p}{\sqrt{p(1-p)/n}}")
    st.latex(r"P(\hat{p}<c), \quad P(\hat{p}>c), \quad P(a<\hat{p}<b)")

    st.markdown("## 2. Intervalos de confianza")
    st.markdown("### Media con Z")
    st.latex(r"IC_\mu = \bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}")
    st.markdown("### Media con t")
    st.latex(r"IC_\mu = \bar{x} \pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt{n}}")
    st.markdown("### Proporción con Z")
    st.latex(r"IC_p = \hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")

    st.markdown("## 3. Pruebas de hipótesis")
    st.markdown("### Media con Z")
    st.latex(r"z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}")
    st.markdown("### Media con t")
    st.latex(r"t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}}")
    st.latex(r"gl = n-1")
    st.markdown("### Proporción con Z")
    st.latex(r"z = \frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}")
    st.markdown("### Diferencia de medias con Z")
    st.latex(r"z = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}")
    st.markdown("### Diferencia de medias con t (Welch)")
    st.latex(r"t = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}")
    st.latex(r"gl \approx \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1-1}+\frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2-1}}")
    st.markdown("### Diferencia de medias con t pooled")
    st.latex(r"s_p^2 = \frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}")
    st.latex(r"t = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}")
    st.latex(r"gl = n_1+n_2-2")
    st.markdown("### Muestras pareadas")
    st.latex(r"d_i = X_{después,i} - X_{antes,i}")
    st.latex(r"\bar{d} = \frac{\sum d_i}{n}")
    st.latex(r"t = \frac{\bar{d}-d_0}{s_d/\sqrt{n}}")
    st.latex(r"gl = n-1")
    st.markdown("### Diferencia de proporciones")
    st.latex(r"\hat{p}_1 = \frac{x_1}{n_1}, \quad \hat{p}_2 = \frac{x_2}{n_2}")
    st.latex(r"\hat{p} = \frac{x_1+x_2}{n_1+n_2}")
    st.latex(r"\hat{q} = 1-\hat{p}")
    st.latex(r"z = \frac{(\hat{p}_1-\hat{p}_2)-d_0}{\sqrt{\hat{p}\hat{q}\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")

    st.markdown("## 4. Correlación de Pearson")
    st.latex(r"r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}")

    st.markdown("## 5. Indicadores estadísticos")
    st.latex(r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i")
    st.latex(r"\sigma^2 = \frac{\sum (x_i-\mu)^2}{N}")
    st.latex(r"s^2 = \frac{\sum (x_i-\bar{x})^2}{n-1}")
    st.latex(r"\sigma = \sqrt{\sigma^2}, \qquad s = \sqrt{s^2}")
    st.latex(r"CV = \frac{s}{\bar{x}} \times 100\%")
    st.latex(r"As = \frac{3(\bar{x}-Mo)}{s}")

    st.markdown("## 6. Reglas de conteo")
    st.latex(r"P(n)=n!")
    st.latex(r"P = \frac{n!}{n_1!n_2!\cdots n_k!}")
    st.latex(r"V(n,r)=\frac{n!}{(n-r)!}")
    st.latex(r"VR(n,r)=n^r")
    st.latex(r"C(n,r)=\binom{n}{r}=\frac{n!}{r!(n-r)!}")
    st.latex(r"CR(n,r)=\binom{n+r-1}{r}")

    st.markdown("## 7. Teorema de Bayes")
    st.latex(r"P(A_i|B)=\frac{P(A_i)P(B|A_i)}{\sum_{j=1}^{n}P(A_j)P(B|A_j)}")

    st.markdown("## 8. Variable aleatoria discreta")
    st.latex(r"E(X)=\sum x_i P(x_i)")
    st.latex(r"E(X^2)=\sum x_i^2 P(x_i)")
    st.latex(r"Var(X)=E(X^2)-[E(X)]^2")
    st.latex(r"\sigma_X = \sqrt{Var(X)}")

    st.markdown("## 9. Distribuciones probabilísticas")
    st.markdown("### Normal")
    st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")
    st.markdown("### Binomial")
    st.latex(r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
    st.markdown("### Poisson")
    st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}")
    st.markdown("### Exponencial")
    st.latex(r"f(x)=\lambda e^{-\lambda x}, \quad x\geq 0")
    st.latex(r"P(X\le x)=1-e^{-\lambda x}")
    st.latex(r"P(X>x)=e^{-\lambda x}")

# ============================================================
# MÓDULOS ACADÉMICOS ADICIONALES
# ============================================================

elif categoria == "Indicadores Estadísticos":
    render_section_header("Indicadores Estadísticos", "Medidas de tendencia, dispersión, forma y posición, con datos manuales o archivo.", "🧮")
    tema = render_topic_selector(["Tendencia Central", "Dispersión", "Forma (Pearson)", "Posición"], "ind_tema")
    fuente = st.radio("Fuente de datos", ["Datos manuales", "Archivo CSV o Excel"], horizontal=True)

    if fuente == "Datos manuales":
        if tema in ["Tendencia Central", "Dispersión", "Posición"]:
            d_str = st.text_input("Ingrese datos separados por coma", "12, 14, 10, 15, 14, 18")
            try:
                datos = sorted([float(x) for x in d_str.replace(",", " ").split() if x.strip()])
                if len(datos) >= 2:
                    if tema == "Tendencia Central":
                        m = statistics.mean(datos)
                        med = statistics.median(datos)
                        modas = calcular_moda(datos)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Media", f"{m:.4f}")
                        c2.metric("Mediana", f"{med:.4f}")
                        c3.metric("Moda(s)", ", ".join(f"{x:g}" for x in modas))
                    elif tema == "Dispersión":
                        t_d = st.radio("Contexto", ["Muestra", "Población"], horizontal=True)
                        m = statistics.mean(datos)
                        dev = statistics.pstdev(datos) if t_d == "Población" else statistics.stdev(datos)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Varianza", f"{dev**2:.4f}")
                        c2.metric("Desv. Est.", f"{dev:.4f}")
                        c3.metric("CV", f"{(dev/m*100):.2f}%")
                    elif tema == "Posición":
                        pk = st.number_input("Percentil (k)", 1, 99, 90)
                        st.metric(f"P{pk}", f"{calcular_percentil(datos, pk):.4f}")
            except Exception as e:
                st.warning(f"Revisa los datos ingresados. {e}")
        else:
            c1, c2, c3 = st.columns(3)
            m = c1.number_input("Media", value=15.0)
            mo = c2.number_input("Moda", value=14.0)
            s = c3.number_input("Desviación", min_value=0.1, value=2.0)
            asimetria = (3 * (m - mo)) / s
            st.metric("Asimetría", f"{asimetria:.4f}")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="indicadores_file")
        if archivo is not None:
            try:
                df = leer_csv_excel(archivo)
                st.success("Archivo cargado correctamente.")
                st.dataframe(df.head())
                numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categoricas = df.columns.tolist()
                if tema in ["Tendencia Central", "Dispersión", "Posición"]:
                    if len(numericas) == 0:
                        st.error("No hay columnas numéricas para analizar.")
                    else:
                        col_num = st.selectbox("Selecciona una variable numérica", numericas)
                        datos = df[col_num].dropna().astype(float).tolist()
                        datos_ordenados = sorted(datos)
                        if tema == "Tendencia Central":
                            m = statistics.mean(datos)
                            med = statistics.median(datos)
                            modas = calcular_moda(datos)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Media", f"{m:.4f}")
                            c2.metric("Mediana", f"{med:.4f}")
                            c3.metric("Moda(s)", ", ".join(f"{x:g}" for x in modas[:10]))
                        elif tema == "Dispersión":
                            t_d = st.radio("Contexto", ["Muestra", "Población"], horizontal=True, key="disp_arch")
                            m = statistics.mean(datos)
                            dev = statistics.pstdev(datos) if t_d == "Población" else statistics.stdev(datos)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Varianza", f"{dev**2:.4f}")
                            c2.metric("Desv. Est.", f"{dev:.4f}")
                            c3.metric("CV", f"{(dev/m*100):.2f}%")
                        elif tema == "Posición":
                            pk = st.number_input("Percentil (k)", 1, 99, 90, key="pos_arch")
                            st.metric(f"P{pk}", f"{calcular_percentil(datos_ordenados, pk):.4f}")
                else:
                    if len(numericas) == 0:
                        st.error("No hay columnas numéricas para analizar.")
                    else:
                        col_num = st.selectbox("Selecciona una variable numérica", numericas, key="forma_arch")
                        datos = df[col_num].dropna().astype(float).tolist()
                        media = statistics.mean(datos)
                        modas = calcular_moda(datos)
                        moda = modas[0]
                        desv = statistics.stdev(datos) if len(datos) > 1 else 0
                        if desv == 0:
                            st.warning("La desviación es 0, no se puede calcular asimetría.")
                        else:
                            asimetria = (3 * (media - moda)) / desv
                            st.metric("Asimetría de Pearson", f"{asimetria:.4f}")
                st.markdown("### Gráficos")
                tipo_graf = st.selectbox("Tipo de gráfico", ["Gráfico de barras", "Gráfico de torta"])
                col_cat = st.selectbox("Variable categórica para graficar", categoricas)
                freq = df[col_cat].astype(str).value_counts().reset_index()
                freq.columns = [col_cat, "Frecuencia"]
                if tipo_graf == "Gráfico de barras":
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(freq[col_cat], freq["Frecuencia"])
                    ax.set_title(f"Frecuencia de {col_cat}")
                    ax.set_xlabel(col_cat)
                    ax.set_ylabel("Frecuencia")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.pie(freq["Frecuencia"], labels=freq[col_cat], autopct="%1.1f%%", startangle=90)
                    ax.set_title(f"Distribución de {col_cat}")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")

elif categoria == "Reglas de Conteo":
    render_section_header("Reglas de Conteo", "Permutaciones, variaciones y combinaciones.", "🔢")
    tema = render_topic_selector([
        "Permutación (Sin repetición)",
        "Permutación con repetición",
        "Combinación (Sin repetición)",
        "Combinación con repetición",
        "Variación (Sin repetición)",
        "Variación con repetición"
    ], "conteo_tema")

    if "Permutación (Sin repetición)" in tema:
        st.latex(r"P(n,r) = \frac{n!}{(n-r)!}")
    elif "Permutación con repetición" in tema:
        st.latex(r"P'_n(r) = n^r")
    elif "Combinación (Sin repetición)" in tema:
        st.latex(r"C_n^r = \binom{n}{r} = \frac{n!}{r!(n-r)!}")
    elif "Combinación con repetición" in tema:
        st.latex(r"CR_n^r = \binom{n+r-1}{r}")
    elif "Variación (Sin repetición)" in tema:
        st.latex(r"V_n^r = \frac{n!}{(n-r)!}")
    elif "Variación con repetición" in tema:
        st.latex(r"VR_n^r = n^r")

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n", 1, value=10)
    with c2:
        max_r = n if "(Sin repetición)" in tema else None
        r = st.number_input("r", 0, max_value=max_r, value=min(3, n))

    if "Permutación (Sin repetición)" in tema:
        res = variacion_sin_rep(n, r)
    elif "Permutación con repetición" in tema:
        res = variacion_con_rep(n, r)
    elif "Combinación con repetición" in tema:
        res = combinacion_con_rep(n, r)
    elif "Combinación (Sin repetición)" in tema:
        res = combinacion_sin_rep(n, r)
    elif "Variación con repetición" in tema:
        res = variacion_con_rep(n, r)
    else:
        res = variacion_sin_rep(n, r)

    st.metric("Resultado", f"{res:,}")

elif categoria == "Teorema de Bayes":
    render_section_header("Teorema de Bayes", "Probabilidades a posteriori.", "🧠")
    nh = st.number_input("Número de hipótesis", 2, 5, 2)
    p1, p2 = [], []
    for i in range(nh):
        c1, c2 = st.columns(2)
        p1.append(c1.number_input(f"P(A{i+1})", 0.0, 1.0, 1.0/nh, key=f"bayes_a_{i}"))
        p2.append(c2.number_input(f"P(B|A{i+1})", 0.0, 1.0, 0.5, key=f"bayes_b_{i}"))
    if abs(sum(p1) - 1.0) < 1e-4:
        pb = sum(a * b for a, b in zip(p1, p2))
        st.metric("P(B)", f"{pb:.4f}")
    else:
        st.warning("Las probabilidades P(Ai) deben sumar 1.")

elif categoria == "Variable Aleatoria":
    render_section_header("Variable Aleatoria", "Momentos y distribución discreta.", "🎲")
    nv = st.number_input("Número de valores", 1, 20, 3)
    vx, vp = [], []
    for i in range(nv):
        c1, c2 = st.columns(2)
        vx.append(c1.number_input(f"Valor x_{i+1}", value=float(i), key=f"va_x_{i}"))
        default_p = 0.0
        if i == nv - 1:
            suma_previa = sum(vp)
            default_p = max(0.0, 1.0 - suma_previa)
        else:
            default_p = 1.0 / nv
        vp.append(c2.number_input(f"Probabilidad P(x_{i+1})", 0.0, 1.0, default_p, format="%.4f", key=f"va_p_{i}"))

    ex = sum(x * p for x, p in zip(vx, vp))
    ex2 = sum((x**2) * p for x, p in zip(vx, vp))
    var_x = max(0.0, ex2 - ex**2)
    sigma_x = math.sqrt(var_x)

    indices = list(range(1, nv + 1))
    ei = sum(i * p for i, p in zip(indices, vp))
    ei2 = sum((i**2) * p for i, p in zip(indices, vp))
    var_i = max(0.0, ei2 - ei**2)
    sigma_i = math.sqrt(var_i)

    df_data = {
        "i": indices,
        "x": vx,
        "P(x)": vp,
        "x·P(x)": [x * p for x, p in zip(vx, vp)],
        "i·P(x)": [i * p for i, p in zip(indices, vp)],
        "x²·P(x)": [(x**2) * p for x, p in zip(vx, vp)],
        "i²·P(x)": [(i**2) * p for i, p in zip(indices, vp)]
    }
    st.dataframe(pd.DataFrame(df_data))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Esperanza E(X)", f"{ex:.6f}")
        st.metric("Varianza Var(X)", f"{var_x:.6f}")
        st.metric("Desv. Estándar σ", f"{sigma_x:.6f}")
    with c2:
        st.metric("Esperanza E(i)", f"{ei:.6f}")
        st.metric("Varianza Var(i)", f"{var_i:.6f}")
        st.metric("Desv. Estándar σ(i)", f"{sigma_i:.6f}")

elif categoria == "Distribuciones Probabilísticas":
    render_section_header("Distribuciones Probabilísticas", "Normal, Binomial, Poisson y Exponencial.", "📈")
    tema = render_topic_selector(["Normal", "Binomial", "Poisson", "Exponencial"], "dist_prob_tema")
    tipo_calc = st.selectbox("Tipo de probabilidad", ["Puntual / Densidad", "Acumulada Inferior (P ≤ x)", "Acumulada Superior (P > x)", "Intervalo (P [a, b])"], key="dist_prob_tipo")

    res = 0.0
    res_label = "Resultado"
    c1, c2 = st.columns(2)

    if tema == "Normal":
        mu = c1.number_input("Media (μ)", value=0.0)
        sigma = c2.number_input("Desv. Estándar (σ)", 0.001, value=1.0)
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Límite inferior (a)", value=-1.0)
            b = c2.number_input("Límite superior (b)", value=1.0)
            res = cdf_normal_estandar((b - mu) / sigma) - cdf_normal_estandar((a - mu) / sigma)
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            x = st.number_input("Valor (x)", value=0.0)
            if "Puntual" in tipo_calc:
                res = pdf_normal(x, mu, sigma)
                res_label = f"f({x}) [Densidad]"
            elif "Inferior" in tipo_calc:
                res = cdf_normal_estandar((x - mu) / sigma)
                res_label = f"P(X ≤ {x})"
            else:
                res = 1 - cdf_normal_estandar((x - mu) / sigma)
                res_label = f"P(X > {x})"

    elif tema == "Binomial":
        n = c1.number_input("Ensayos (n)", 1, value=10)
        p = c2.number_input("Prob. éxito (p)", 0.0, 1.0, 0.5)
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Mínimo éxitos (a)", 0, n, 0)
            b = c2.number_input("Máximo éxitos (b)", 0, n, n)
            res = sum(probabilidad_binomial(i, n, p) for i in range(int(a), int(b) + 1))
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            k = st.number_input("Éxitos (k)", 0, n, 0)
            if "Puntual" in tipo_calc:
                res = probabilidad_binomial(int(k), n, p)
                res_label = f"P(X = {k})"
            elif "Inferior" in tipo_calc:
                res = sum(probabilidad_binomial(i, n, p) for i in range(int(k) + 1))
                res_label = f"P(X ≤ {k})"
            else:
                res = 1 - sum(probabilidad_binomial(i, n, p) for i in range(int(k) + 1))
                res_label = f"P(X > {k})"

    elif tema == "Poisson":
        lam = c1.number_input("Tasa promedio (λ)", 0.001, value=5.0)
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Mínimo (a)", 0)
            b = c2.number_input("Máximo (b)", 0)
            res = sum(probabilidad_poisson(i, lam) for i in range(int(a), int(b) + 1))
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            k = st.number_input("Ocurrencias (k)", 0)
            if "Puntual" in tipo_calc:
                res = probabilidad_poisson(int(k), lam)
                res_label = f"P(X = {k})"
            elif "Inferior" in tipo_calc:
                res = sum(probabilidad_poisson(i, lam) for i in range(int(k) + 1))
                res_label = f"P(X ≤ {k})"
            else:
                res = 1 - sum(probabilidad_poisson(i, lam) for i in range(int(k) + 1))
                res_label = f"P(X > {k})"

    else:
        lam = c1.number_input("Tasa (λ)", 0.001, value=1.0)
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Inicio (a)", 0.0)
            b = c2.number_input("Fin (b)", 0.0)
            res = math.exp(-lam * a) - math.exp(-lam * b)
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            x = st.number_input("Valor (x)", 0.0)
            if "Puntual" in tipo_calc:
                res = lam * math.exp(-lam * x)
                res_label = f"f({x}) [Densidad]"
            elif "Inferior" in tipo_calc:
                res = 1 - math.exp(-lam * x)
                res_label = f"P(X ≤ {x})"
            else:
                res = math.exp(-lam * x)
                res_label = f"P(X > {x})"

    render_result_box(res_label, f"{res:.10f}")

elif categoria == "Intervalos de Confianza_2":
    render_section_header("Intervalos de Confianza_2", "Lógica Z o t según la información disponible.", "📏")
    n, m, c = render_three_inputs([
        ("Tamaño de muestra (n)", 30),
        ("Media muestral (x̄)", 100),
        ("Nivel de confianza", 0.95)
    ])
    n = int(n)
    sigma_conocida = st.radio("¿Conoce σ?", ["Sí, es conocida (σ)", "No, usar muestral (s)"], horizontal=True)
    label_s = "Desviación estándar (σ)" if "Sí" in sigma_conocida else "Desviación estándar (s)"
    s = parse_num(st.text_input(label_s, "15"))

    es_sigma = "Sí" in sigma_conocida
    if es_sigma:
        v_critico = get_z_value(c)
        razon = "Se utiliza Z porque la desviación estándar poblacional es conocida."
    else:
        if n >= 30:
            v_critico = get_z_value(c)
            razon = f"Se utiliza Z como aproximación porque n={n} y n ≥ 30."
        else:
            v_critico = get_t_value(c, n - 1)
            razon = f"Se utiliza t de Student porque σ es desconocida y n={n} es pequeña."
    err = v_critico * (s / math.sqrt(n))
    li, ls = m - err, m + err
    st.info(razon)
    c1, c2, c3 = st.columns(3)
    c1.metric("Valor crítico", f"{v_critico:.4f}")
    c2.metric("Margen de error", f"{err:.4f}")
    c3.metric("Error estándar", f"{(s/math.sqrt(n)):.4f}")
    st.success(f"IC = [{li:.4f}, {ls:.4f}]")

elif categoria == "Historial de Cálculos":
    render_section_header("Historial de Cálculos", "Registro local de cálculos realizados.", "📜")
    path = "historial_calculos.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            registros = [json.loads(line) for line in f]
        for reg in reversed(registros):
            with st.expander(f"🕒 {reg['fecha']} - {reg['modulo']}"):
                st.json(reg["inputs"])
        if st.button("🗑️ Borrar historial"):
            os.remove(path)
            st.rerun()
    else:
        st.info("No hay historial guardado.")


