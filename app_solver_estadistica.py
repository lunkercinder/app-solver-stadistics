
# app.py
# ============================================================
# SOLVER DE ESTADÍSTICA - STREAMLIT
# Probabilidades | Intervalos | Pruebas de hipótesis | Correlación
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Solver de Estadística",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Solver de Estadística")
st.caption("Probabilidades, intervalos de confianza, pruebas de hipótesis y correlación de Pearson")

# ============================================================
# FUNCIONES BASE
# ============================================================

def parse_num(valor):
    if isinstance(valor, (int, float)):
        return float(valor)

    s = str(valor).strip().replace(" ", "")

    if s.endswith("%"):
        s = s[:-1]
        s = s.replace(".", "").replace(",", ".")
        return float(s) / 100

    if "," in s and "." not in s:
        s = s.replace(",", ".")

    return float(s)


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def z_critico(alpha, tipo):
    if tipo == "Bilateral":
        return abs(inversa_normal(1 - alpha / 2))
    return abs(inversa_normal(1 - alpha))


def inversa_normal(p, tol=1e-6):
    """
    Aproximación por búsqueda binaria de la inversa normal estándar.
    """
    low, high = -5, 5
    while high - low > tol:
        mid = (low + high) / 2
        if phi(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


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


def decision_region_critica(estadistico, critico, tipo_prueba):
    if tipo_prueba == "Bilateral":
        rechaza = abs(estadistico) > critico
        region = f"|estadístico| > {critico:.4f}"
    elif tipo_prueba == "Cola izquierda":
        rechaza = estadistico < -critico
        region = f"estadístico < {-critico:.4f}"
    else:
        rechaza = estadistico > critico
        region = f"estadístico > {critico:.4f}"

    return rechaza, region


def mostrar_decision(rechaza):
    if rechaza:
        st.success("✅ Decisión: Se rechaza H₀.")
    else:
        st.warning("⚠️ Decisión: No se rechaza H₀.")


def leer_csv_flexible(archivo):
    try:
        return pd.read_csv(archivo, sep=";")
    except Exception:
        archivo.seek(0)
        try:
            return pd.read_csv(archivo, sep=",")
        except Exception:
            archivo.seek(0)
            return pd.read_csv(archivo)


# ============================================================
# MENÚ LATERAL
# ============================================================

menu = st.sidebar.selectbox(
    "Menú principal",
    [
        "Inicio",
        "Probabilidad - Media muestral",
        "Probabilidad - Proporción muestral",
        "Intervalo de confianza - Media con Z",
        "Intervalo de confianza - Media con t",
        "Intervalo de confianza - Proporción",
        "Prueba de hipótesis - Media con Z",
        "Prueba de hipótesis - Media con t",
        "Prueba de hipótesis - Proporción",
        "Prueba de hipótesis - Diferencia de medias",
        "Prueba de hipótesis - Diferencia de proporciones",
        "Correlación de Pearson"
    ]
)

# ============================================================
# INICIO
# ============================================================

if menu == "Inicio":
    st.subheader("Bienvenido")
    st.write(
        """
        Esta aplicación permite resolver ejercicios básicos de estadística de forma interactiva.

        Usa el menú lateral para escoger el tema.
        """
    )

    st.info(
        """
        Recomendación:
        - Usa punto decimal o coma decimal.
        - Puedes escribir proporciones como 0.17 o 17%.
        - En pruebas de hipótesis, si el enunciado no da alfa, normalmente se usa α = 0.05.
        """
    )

# ============================================================
# PROBABILIDAD MEDIA MUESTRAL
# ============================================================

elif menu == "Probabilidad - Media muestral":
    st.subheader("Probabilidad con media muestral X̄")

    col1, col2, col3 = st.columns(3)

    with col1:
        mu = parse_num(st.text_input("μ media poblacional", "50"))
    with col2:
        sigma = parse_num(st.text_input("σ desviación poblacional", "10"))
    with col3:
        n = int(parse_num(st.text_input("n tamaño de muestra", "36")))

    tipo = st.selectbox("Tipo de probabilidad", ["P(X̄ < c)", "P(X̄ > c)", "P(a < X̄ < b)"])

    se = sigma / math.sqrt(n)
    st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
    st.write(f"SE = {se:.6f}")

    if tipo in ["P(X̄ < c)", "P(X̄ > c)"]:
        c = parse_num(st.text_input("Valor c", "52"))

        if st.button("Calcular"):
            z = (c - mu) / se
            prob_menor = phi(z)

            st.latex(r"z = \frac{c-\mu}{SE}")
            st.write(f"z = {z:.6f}")

            if tipo == "P(X̄ < c)":
                st.success(f"P(X̄ < {c}) = {prob_menor:.6f} = {prob_menor*100:.2f}%")
            else:
                prob = 1 - prob_menor
                st.success(f"P(X̄ > {c}) = {prob:.6f} = {prob*100:.2f}%")

    else:
        col1, col2 = st.columns(2)
        with col1:
            a = parse_num(st.text_input("Límite inferior a", "45"))
        with col2:
            b = parse_num(st.text_input("Límite superior b", "55"))

        if st.button("Calcular"):
            z_a = (a - mu) / se
            z_b = (b - mu) / se
            prob = phi(z_b) - phi(z_a)

            st.write(f"z(a) = {z_a:.6f}")
            st.write(f"z(b) = {z_b:.6f}")
            st.success(f"P({a} < X̄ < {b}) = {prob:.6f} = {prob*100:.2f}%")

# ============================================================
# PROBABILIDAD PROPORCIÓN MUESTRAL
# ============================================================

elif menu == "Probabilidad - Proporción muestral":
    st.subheader("Probabilidad con proporción muestral p̂")

    col1, col2 = st.columns(2)

    with col1:
        p = parse_num(st.text_input("p proporción poblacional", "17%"))
    with col2:
        n = int(parse_num(st.text_input("n tamaño de muestra", "400")))

    st.write(f"Verificación: n = {n}, n·p = {n*p:.4f}")

    if n >= 30 and n * p >= 5:
        st.success("✅ Se cumple la aproximación normal usada por el profe.")
    else:
        st.warning("⚠️ No se cumple completamente la regla n ≥ 30 y n·p ≥ 5.")

    tipo = st.selectbox("Tipo de probabilidad", ["P(p̂ < c)", "P(p̂ > c)", "P(a < p̂ < b)"])

    se = math.sqrt(p * (1 - p) / n)
    st.latex(r"SE = \sqrt{\frac{p(1-p)}{n}}")
    st.write(f"SE = {se:.6f}")

    if tipo in ["P(p̂ < c)", "P(p̂ > c)"]:
        c = parse_num(st.text_input("Valor c", "20%"))

        if st.button("Calcular"):
            z = (c - p) / se
            prob_menor = phi(z)

            st.write(f"z = {z:.6f}")

            if tipo == "P(p̂ < c)":
                st.success(f"P(p̂ < {c:.4f}) = {prob_menor:.6f} = {prob_menor*100:.2f}%")
            else:
                prob = 1 - prob_menor
                st.success(f"P(p̂ > {c:.4f}) = {prob:.6f} = {prob*100:.2f}%")

    else:
        col1, col2 = st.columns(2)
        with col1:
            a = parse_num(st.text_input("Límite inferior a", "10%"))
        with col2:
            b = parse_num(st.text_input("Límite superior b", "20%"))

        if st.button("Calcular"):
            z_a = (a - p) / se
            z_b = (b - p) / se
            prob = phi(z_b) - phi(z_a)

            st.write(f"z(a) = {z_a:.6f}")
            st.write(f"z(b) = {z_b:.6f}")
            st.success(f"P({a:.4f} < p̂ < {b:.4f}) = {prob:.6f} = {prob*100:.2f}%")

# ============================================================
# INTERVALO MEDIA Z
# ============================================================

elif menu == "Intervalo de confianza - Media con Z":
    st.subheader("Intervalo de confianza para μ con Z")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nc = parse_num(st.text_input("Nivel de confianza", "95%"))
    with col2:
        xbar = parse_num(st.text_input("x̄ media muestral", "50"))
    with col3:
        sigma = parse_num(st.text_input("σ poblacional", "10"))
    with col4:
        n = int(parse_num(st.text_input("n", "36")))

    if nc > 1:
        nc = nc / 100

    if st.button("Calcular intervalo"):
        alpha = 1 - nc
        z = abs(inversa_normal(1 - alpha / 2))
        se = sigma / math.sqrt(n)
        margen = z * se
        li = xbar - margen
        ls = xbar + margen

        st.write(f"α = {alpha:.4f}")
        st.write(f"z* = {z:.4f}")
        st.write(f"SE = {se:.6f}")
        st.write(f"Margen de error = {margen:.6f}")

        st.success(f"IC {nc*100:.2f}% para μ = ({li:.6f}, {ls:.6f})")

# ============================================================
# INTERVALO MEDIA T
# ============================================================

elif menu == "Intervalo de confianza - Media con t":
    st.subheader("Intervalo de confianza para μ con t-Student")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nc = parse_num(st.text_input("Nivel de confianza", "95%"))
    with col2:
        xbar = parse_num(st.text_input("x̄ media muestral", "50"))
    with col3:
        s = parse_num(st.text_input("s muestral", "10"))
    with col4:
        n = int(parse_num(st.text_input("n", "25")))

    if nc > 1:
        nc = nc / 100

    if st.button("Calcular intervalo"):
        alpha = 1 - nc
        df = n - 1

        # Aproximación usando normal inversa para df grandes o como apoyo simple.
        # Si quieres t exacta, instala scipy y reemplaza por scipy.stats.t.ppf.
        try:
            from scipy.stats import t
            t_star = t.ppf(1 - alpha / 2, df)
        except Exception:
            t_star = abs(inversa_normal(1 - alpha / 2))

        se = s / math.sqrt(n)
        margen = t_star * se
        li = xbar - margen
        ls = xbar + margen

        st.write(f"gl = n - 1 = {df}")
        st.write(f"t* ≈ {t_star:.4f}")
        st.write(f"SE = {se:.6f}")
        st.write(f"Margen de error = {margen:.6f}")

        st.success(f"IC {nc*100:.2f}% para μ = ({li:.6f}, {ls:.6f})")

# ============================================================
# INTERVALO PROPORCIÓN
# ============================================================

elif menu == "Intervalo de confianza - Proporción":
    st.subheader("Intervalo de confianza para proporción p")

    nc = parse_num(st.text_input("Nivel de confianza", "95%"))

    if nc > 1:
        nc = nc / 100

    modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"])

    if modo == "Ya tengo p̂":
        col1, col2 = st.columns(2)
        with col1:
            phat = parse_num(st.text_input("p̂", "0.50"))
        with col2:
            n = int(parse_num(st.text_input("n", "400")))
    else:
        col1, col2 = st.columns(2)
        with col1:
            x = int(parse_num(st.text_input("x éxitos", "200")))
        with col2:
            n = int(parse_num(st.text_input("n", "400")))
        phat = x / n if n != 0 else 0

    if st.button("Calcular intervalo"):
        alpha = 1 - nc
        z = abs(inversa_normal(1 - alpha / 2))
        se = math.sqrt(phat * (1 - phat) / n)
        margen = z * se
        li = phat - margen
        ls = phat + margen

        st.write(f"p̂ = {phat:.6f}")
        st.write(f"z* = {z:.4f}")
        st.write(f"SE = {se:.6f}")
        st.write(f"Margen de error = {margen:.6f}")

        st.success(f"IC {nc*100:.2f}% para p = ({li:.6f}, {ls:.6f})")

# ============================================================
# PRUEBA MEDIA Z
# ============================================================

elif menu == "Prueba de hipótesis - Media con Z":
    st.subheader("Prueba de hipótesis para una media con Z")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mu0 = parse_num(st.text_input("μ₀", "50"))
    with col2:
        xbar = parse_num(st.text_input("x̄", "53"))
    with col3:
        sigma = parse_num(st.text_input("σ poblacional", "12"))
    with col4:
        n = int(parse_num(st.text_input("n", "36")))

    alpha = parse_num(st.text_input("α nivel de significancia", "0.05"))
    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

    if st.button("Resolver prueba"):
        se = sigma / math.sqrt(n)
        z = (xbar - mu0) / se
        critico = z_critico(alpha, tipo_prueba)
        rechaza, region = decision_region_critica(z, critico, tipo_prueba)

        st.latex(r"z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}")
        st.write(f"SE = {se:.6f}")
        st.write(f"z calculado = {z:.6f}")
        st.write(f"Región crítica: {region}")
        mostrar_decision(rechaza)

# ============================================================
# PRUEBA MEDIA T
# ============================================================

elif menu == "Prueba de hipótesis - Media con t":
    st.subheader("Prueba de hipótesis para una media con t-Student")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mu0 = parse_num(st.text_input("μ₀", "50"))
    with col2:
        xbar = parse_num(st.text_input("x̄", "53"))
    with col3:
        s = parse_num(st.text_input("s muestral", "12"))
    with col4:
        n = int(parse_num(st.text_input("n", "25")))

    alpha = parse_num(st.text_input("α nivel de significancia", "0.05"))
    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

    if st.button("Resolver prueba"):
        df = n - 1
        se = s / math.sqrt(n)
        t_calc = (xbar - mu0) / se

        try:
            from scipy.stats import t
            if tipo_prueba == "Bilateral":
                critico = t.ppf(1 - alpha / 2, df)
            else:
                critico = t.ppf(1 - alpha, df)
        except Exception:
            critico = z_critico(alpha, tipo_prueba)

        rechaza, region = decision_region_critica(t_calc, critico, tipo_prueba)

        st.latex(r"t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}}")
        st.write(f"gl = {df}")
        st.write(f"SE = {se:.6f}")
        st.write(f"t calculado = {t_calc:.6f}")
        st.write(f"Región crítica: {region}")
        mostrar_decision(rechaza)

# ============================================================
# PRUEBA PROPORCIÓN
# ============================================================

elif menu == "Prueba de hipótesis - Proporción":
    st.subheader("Prueba de hipótesis para una proporción con Z")

    p0 = parse_num(st.text_input("p₀", "17%"))
    alpha = parse_num(st.text_input("α nivel de significancia", "0.05"))

    modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"])

    if modo == "Ya tengo p̂":
        col1, col2 = st.columns(2)
        with col1:
            phat = parse_num(st.text_input("p̂", "0.20"))
        with col2:
            n = int(parse_num(st.text_input("n", "400")))
    else:
        col1, col2 = st.columns(2)
        with col1:
            x = int(parse_num(st.text_input("x éxitos", "80")))
        with col2:
            n = int(parse_num(st.text_input("n", "400")))
        phat = x / n

    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

    if st.button("Resolver prueba"):
        se = math.sqrt(p0 * (1 - p0) / n)
        z = (phat - p0) / se
        critico = z_critico(alpha, tipo_prueba)
        rechaza, region = decision_region_critica(z, critico, tipo_prueba)

        st.latex(r"z = \frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}")
        st.write(f"p̂ = {phat:.6f}")
        st.write(f"SE = {se:.6f}")
        st.write(f"z calculado = {z:.6f}")
        st.write(f"Región crítica: {region}")
        mostrar_decision(rechaza)

# ============================================================
# DIFERENCIA DE MEDIAS
# ============================================================

elif menu == "Prueba de hipótesis - Diferencia de medias":
    st.subheader("Prueba de hipótesis para diferencia de medias")

    metodo = st.radio(
        "Método",
        [
            "Muestras grandes con Z",
            "Muestras independientes con t pooled",
            "Muestras independientes con t Welch",
            "Muestras pareadas"
        ]
    )

    alpha = parse_num(st.text_input("α nivel de significancia", "0.05"))
    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
    d0 = parse_num(st.text_input("Diferencia hipotética d₀ = μ₁ - μ₂", "0"))

    if metodo in ["Muestras grandes con Z", "Muestras independientes con t pooled", "Muestras independientes con t Welch"]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Muestra 1")
            xbar1 = parse_num(st.text_input("x̄₁", "42.48"))
            s1 = parse_num(st.text_input("s₁", "7.19"))
            n1 = int(parse_num(st.text_input("n₁", "205")))

        with col2:
            st.markdown("### Muestra 2")
            xbar2 = parse_num(st.text_input("x̄₂", "43.04"))
            s2 = parse_num(st.text_input("s₂", "7.47"))
            n2 = int(parse_num(st.text_input("n₂", "195")))

        if st.button("Resolver prueba"):
            if metodo == "Muestras grandes con Z":
                se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
                estadistico = ((xbar1 - xbar2) - d0) / se
                critico = z_critico(alpha, tipo_prueba)
                nombre = "z"

            elif metodo == "Muestras independientes con t pooled":
                sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2)
                sp = math.sqrt(sp2)
                se = sp * math.sqrt((1/n1) + (1/n2))
                estadistico = ((xbar1 - xbar2) - d0) / se
                df = n1 + n2 - 2

                try:
                    from scipy.stats import t
                    if tipo_prueba == "Bilateral":
                        critico = t.ppf(1 - alpha / 2, df)
                    else:
                        critico = t.ppf(1 - alpha, df)
                except Exception:
                    critico = z_critico(alpha, tipo_prueba)

                nombre = "t"
                st.write(f"gl = n₁ + n₂ - 2 = {df}")

            else:
                se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
                estadistico = ((xbar1 - xbar2) - d0) / se

                num = ((s1**2 / n1) + (s2**2 / n2)) ** 2
                den = ((s1**2 / n1) ** 2) / (n1 - 1) + ((s2**2 / n2) ** 2) / (n2 - 1)
                df = num / den

                try:
                    from scipy.stats import t
                    if tipo_prueba == "Bilateral":
                        critico = t.ppf(1 - alpha / 2, df)
                    else:
                        critico = t.ppf(1 - alpha, df)
                except Exception:
                    critico = z_critico(alpha, tipo_prueba)

                nombre = "t"
                st.write(f"gl aproximados Welch = {df:.4f}")

            rechaza, region = decision_region_critica(estadistico, critico, tipo_prueba)

            st.write(f"SE = {se:.6f}")
            st.write(f"{nombre} calculado = {estadistico:.6f}")
            st.write(f"Región crítica: {region}")
            mostrar_decision(rechaza)

    else:
        st.markdown("### Datos pareados")
        st.write("Ingresa los datos separados por coma.")

        antes_txt = st.text_area("Datos antes", "10,12,15,13,11")
        despues_txt = st.text_area("Datos después", "12,15,16,15,13")

        if st.button("Resolver prueba pareada"):
            antes = np.array([parse_num(v) for v in antes_txt.split(",") if v.strip() != ""])
            despues = np.array([parse_num(v) for v in despues_txt.split(",") if v.strip() != ""])

            if len(antes) != len(despues):
                st.error("Ambas listas deben tener la misma cantidad de datos.")
            elif len(antes) < 2:
                st.error("Se necesitan mínimo 2 pares de datos.")
            else:
                d = despues - antes
                n = len(d)
                dbar = np.mean(d)
                sd = np.std(d, ddof=1)
                se = sd / math.sqrt(n)
                t_calc = (dbar - d0) / se
                df = n - 1

                try:
                    from scipy.stats import t
                    if tipo_prueba == "Bilateral":
                        critico = t.ppf(1 - alpha / 2, df)
                    else:
                        critico = t.ppf(1 - alpha, df)
                except Exception:
                    critico = z_critico(alpha, tipo_prueba)

                rechaza, region = decision_region_critica(t_calc, critico, tipo_prueba)

                st.write(f"Diferencias = después - antes")
                st.write(f"n = {n}")
                st.write(f"Promedio de diferencias = {dbar:.6f}")
                st.write(f"s_d = {sd:.6f}")
                st.write(f"SE = {se:.6f}")
                st.write(f"t calculado = {t_calc:.6f}")
                st.write(f"gl = {df}")
                st.write(f"Región crítica: {region}")
                mostrar_decision(rechaza)

# ============================================================
# DIFERENCIA DE PROPORCIONES
# ============================================================

elif menu == "Prueba de hipótesis - Diferencia de proporciones":
    st.subheader("Prueba de hipótesis para diferencia de proporciones")

    alpha = parse_num(st.text_input("α nivel de significancia", "0.05"))
    tipo_prueba = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
    d0 = parse_num(st.text_input("Diferencia hipotética d₀ = p₁ - p₂", "0"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Muestra 1")
        x1 = int(parse_num(st.text_input("x₁ éxitos", "80")))
        n1 = int(parse_num(st.text_input("n₁", "205")))

    with col2:
        st.markdown("### Muestra 2")
        x2 = int(parse_num(st.text_input("x₂ éxitos", "90")))
        n2 = int(parse_num(st.text_input("n₂", "195")))

    if st.button("Resolver prueba"):
        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        q_pool = 1 - p_pool

        se = math.sqrt(p_pool * q_pool * ((1/n1) + (1/n2)))
        z = ((p1 - p2) - d0) / se
        critico = z_critico(alpha, tipo_prueba)
        rechaza, region = decision_region_critica(z, critico, tipo_prueba)

        st.write(f"p̂₁ = {p1:.6f}")
        st.write(f"p̂₂ = {p2:.6f}")
        st.write(f"p combinada = {p_pool:.6f}")
        st.write(f"SE = {se:.6f}")
        st.write(f"z calculado = {z:.6f}")
        st.write(f"Región crítica: {region}")
        mostrar_decision(rechaza)

# ============================================================
# CORRELACIÓN DE PEARSON
# ============================================================

elif menu == "Correlación de Pearson":
    st.subheader("Correlación de Pearson")

    modo = st.radio(
        "¿Cómo quieres ingresar la información?",
        ["Datos manuales", "Archivo CSV"]
    )

    if modo == "Datos manuales":
        col1, col2 = st.columns(2)

        with col1:
            nombre_x = st.text_input("Nombre de X", "X")
            datos_x = st.text_area("Valores de X separados por coma", "1,2,3,4,5")

        with col2:
            nombre_y = st.text_input("Nombre de Y", "Y")
            datos_y = st.text_area("Valores de Y separados por coma", "2,4,6,8,10")

        if st.button("Calcular correlación"):
            try:
                x = np.array([parse_num(v) for v in datos_x.split(",") if v.strip() != ""])
                y = np.array([parse_num(v) for v in datos_y.split(",") if v.strip() != ""])

                if len(x) != len(y):
                    st.error("X y Y deben tener la misma cantidad de datos.")
                elif len(x) < 2:
                    st.error("Se necesitan mínimo 2 datos.")
                else:
                    r = np.corrcoef(x, y)[0, 1]
                    tipo, intensidad = interpretar_correlacion(r)

                    st.success(f"Coeficiente de correlación r = {r:.4f}")
                    st.write(f"Existe una correlación **{tipo} {intensidad}** entre {nombre_x} y {nombre_y}.")

                    if tipo == "positiva":
                        st.write(f"A medida que aumenta {nombre_x}, también tiende a aumentar {nombre_y}.")
                    elif tipo == "negativa":
                        st.write(f"A medida que aumenta {nombre_x}, {nombre_y} tiende a disminuir.")
                    else:
                        st.write(f"No se observa una relación lineal clara entre {nombre_x} y {nombre_y}.")

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(x, y)
                    ax.set_xlabel(nombre_x)
                    ax.set_ylabel(nombre_y)
                    ax.set_title(f"Diagrama de dispersión: {nombre_x} vs {nombre_y}")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

        if archivo is not None:
            try:
                df = leer_csv_flexible(archivo)

                st.success("Base cargada correctamente.")
                st.write(f"Filas: {df.shape[0]}")
                st.write(f"Columnas: {df.shape[1]}")

                st.write("Vista previa:")
                st.dataframe(df.head())

                numericas = df.select_dtypes(include=["int64", "float64"])

                posibles_id = ["ID", "Id", "id", "Registro", "registro"]
                for col in posibles_id:
                    if col in numericas.columns:
                        numericas = numericas.drop(columns=[col])

                if numericas.shape[1] < 2:
                    st.error("La base debe tener mínimo dos columnas numéricas.")
                else:
                    st.write("Variables numéricas detectadas:")
                    st.write(list(numericas.columns))

                    matriz = numericas.corr(method="pearson")

                    st.subheader("Matriz de correlación")
                    st.dataframe(matriz.round(4))

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

                    st.subheader("Interpretación automática")

                    for i in range(len(matriz.columns)):
                        for j in range(i + 1, len(matriz.columns)):
                            var1 = matriz.columns[i]
                            var2 = matriz.columns[j]
                            r = matriz.iloc[i, j]
                            tipo, intensidad = interpretar_correlacion(r)

                            st.write(f"**{var1} y {var2}:** r = {r:.4f} → correlación {tipo} {intensidad}")

            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
