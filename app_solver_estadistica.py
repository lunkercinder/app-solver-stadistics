import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from scipy.stats import norm, t
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(
    page_title="Solver Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px;}
    .sub-title {font-size: 18px; color: #9CA3AF; margin-bottom: 25px;}
    .metric-card {background-color: rgba(120, 120, 120, 0.10); padding: 18px; border-radius: 16px; border: 1px solid rgba(150, 150, 150, 0.25); margin-bottom: 12px;}
    .report-box {background-color: rgba(34, 197, 94, 0.10); padding: 18px; border-radius: 16px; border: 1px solid rgba(34, 197, 94, 0.35); margin-top: 15px;}
    </style>
    """,
    unsafe_allow_html=True
)

def parse_num(valor):
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip().replace(" ", "")
    if s == "":
        raise ValueError("Entrada vacía")
    if s.endswith("%"):
        s = s[:-1]
        s = s.replace(".", "").replace(",", ".")
        return float(s) / 100
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
        return f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, se rechaza la hipótesis nula. Por tanto, existe evidencia estadística suficiente para afirmar que {h1_texto}. En el contexto del problema, esto indica que {contexto}"
    return f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, no se rechaza la hipótesis nula. Por tanto, no existe evidencia estadística suficiente para afirmar que {h1_texto}. En el contexto del problema, esto indica que {contexto}"

def show_report(texto):
    st.markdown("### Texto listo para informe")
    st.markdown(f"<div class='report-box'>{texto}</div>", unsafe_allow_html=True)
    st.download_button("Descargar conclusión en TXT", data=texto, file_name="conclusion_estadistica.txt", mime="text/plain")

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

st.sidebar.title("📊 Menú principal")
categoria = st.sidebar.radio("Sección", ["Inicio", "Probabilidades", "Intervalos de confianza", "Pruebas de hipótesis", "Correlación", "Base de datos", "Fórmulas"])
opcion = None
if categoria == "Probabilidades":
    opcion = st.sidebar.selectbox("Tema", ["Media muestral", "Proporción muestral"])
elif categoria == "Intervalos de confianza":
    opcion = st.sidebar.selectbox("Tema", ["Media con Z", "Media con t", "Proporción con Z"])
elif categoria == "Pruebas de hipótesis":
    opcion = st.sidebar.selectbox("Tema", ["Media con Z", "Media con t", "Proporción con Z", "Diferencia de medias", "Diferencia de proporciones"])
elif categoria == "Correlación":
    opcion = st.sidebar.selectbox("Tema", ["Datos manuales", "Archivo CSV o Excel"])
elif categoria == "Base de datos":
    opcion = st.sidebar.selectbox("Tema", ["Exploración básica", "Comparación por grupo"])

st.markdown("<div class='main-title'>Solver Estadístico</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Calculadora interactiva para estadística aplicada, ingeniería y análisis de datos.</div>", unsafe_allow_html=True)

if categoria == "Inicio":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><h3>📌 Probabilidades</h3><p>Media muestral y proporción muestral con distribución normal.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h3>🧪 Hipótesis</h3><p>Pruebas Z, t, proporciones, diferencias y p-valores automáticos.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h3>📈 Correlación</h3><p>Ingreso manual o carga de archivos CSV/Excel.</p></div>", unsafe_allow_html=True)
    st.info("Usa el menú lateral para escoger el tema. Puedes escribir porcentajes como 17% o proporciones en decimales tal que así → 0.17.")
    st.markdown("### Recomendación para el informe")
    st.write("Cuando resuelvas tu prueba de hipótesis, copia la conclusión generada automáticamente por el programa para tu informe. Está escrita en formato académico, lista para pegar. ")

elif categoria == "Probabilidades" and opcion == "Media muestral":
    st.subheader("Probabilidad con media muestral")
    col1, col2, col3 = st.columns(3)
    with col1: mu = parse_num(st.text_input("μ media poblacional", "50"))
    with col2: sigma = parse_num(st.text_input("σ desviación poblacional", "10"))
    with col3: n = int(parse_num(st.text_input("n tamaño de muestra", "36")))
    tipo = st.selectbox("Tipo de probabilidad", ["P(X̄ < c)", "P(X̄ > c)", "P(a < X̄ < b)"])
    se = sigma / math.sqrt(n)
    st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
    st.write(f"SE = {se:.6f}")
    if tipo in ["P(X̄ < c)", "P(X̄ > c)"]:
        c = parse_num(st.text_input("Valor c", "52"))
        if st.button("Calcular"):
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
        col1, col2 = st.columns(2)
        with col1: a = parse_num(st.text_input("Límite inferior a", "45"))
        with col2: b = parse_num(st.text_input("Límite superior b", "55"))
        if st.button("Calcular"):
            z_a = (a - mu) / se
            z_b = (b - mu) / se
            prob = phi(z_b) - phi(z_a)
            st.write(f"z(a) = {z_a:.6f}")
            st.write(f"z(b) = {z_b:.6f}")
            st.success(f"P({a} < X̄ < {b}) = {fmt_prob(prob)}")
            show_report(f"La probabilidad de que la media muestral esté entre {a} y {b} es aproximadamente {fmt_prob(prob)}.")

elif categoria == "Probabilidades" and opcion == "Proporción muestral":
    st.subheader("Probabilidad con proporción muestral")
    col1, col2 = st.columns(2)
    with col1: p = parse_num(st.text_input("p proporción poblacional", "17%"))
    with col2: n = int(parse_num(st.text_input("n tamaño de muestra", "400")))
    st.write(f"n = {n}")
    st.write(f"n·p = {n*p:.4f}")
    if n >= 30 and n * p >= 5: st.success("Se cumple la aproximación normal usada normalmente en clase.")
    else: st.warning("No se cumple completamente la regla n ≥ 30 y n·p ≥ 5.")
    tipo = st.selectbox("Tipo de probabilidad", ["P(p̂ < c)", "P(p̂ > c)", "P(a < p̂ < b)"])
    se = math.sqrt(p * (1 - p) / n)
    st.latex(r"SE = \sqrt{\frac{p(1-p)}{n}}")
    st.write(f"SE = {se:.6f}")
    if tipo in ["P(p̂ < c)", "P(p̂ > c)"]:
        c = parse_num(st.text_input("Valor c", "20%"))
        if st.button("Calcular"):
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
        col1, col2 = st.columns(2)
        with col1: a = parse_num(st.text_input("Límite inferior a", "10%"))
        with col2: b = parse_num(st.text_input("Límite superior b", "20%"))
        if st.button("Calcular"):
            z_a = (a - p) / se
            z_b = (b - p) / se
            prob = phi(z_b) - phi(z_a)
            st.write(f"z(a) = {z_a:.6f}")
            st.write(f"z(b) = {z_b:.6f}")
            st.success(f"P({a:.4f} < p̂ < {b:.4f}) = {fmt_prob(prob)}")
            show_report(f"La probabilidad de que la proporción muestral esté entre {a:.4f} y {b:.4f} es aproximadamente {fmt_prob(prob)}.")

elif categoria == "Intervalos de confianza":
    st.subheader(f"Intervalo de confianza - {opcion}")
    if opcion in ["Media con Z", "Media con t"]:
        col1, col2, col3, col4 = st.columns(4)
        with col1: nc = parse_num(st.text_input("Nivel de confianza", "95%"))
        with col2: xbar = parse_num(st.text_input("x̄", "50"))
        with col3:
            desv = parse_num(st.text_input("σ poblacional" if opcion == "Media con Z" else "s muestral", "10"))
        with col4: n = int(parse_num(st.text_input("n", "36" if opcion == "Media con Z" else "25")))
        if nc > 1: nc /= 100
        if st.button("Calcular intervalo"):
            alpha = 1 - nc
            if opcion == "Media con Z": crit = abs(inv_norm(1 - alpha / 2)); gl_txt = "No aplica"
            else: crit = critico_t(alpha, n-1, "Bilateral"); gl_txt = str(n-1)
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
        nc = parse_num(st.text_input("Nivel de confianza", "95%"))
        if nc > 1: nc /= 100
        modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"])
        if modo == "Ya tengo p̂":
            col1, col2 = st.columns(2)
            with col1: phat = parse_num(st.text_input("p̂", "0.50"))
            with col2: n = int(parse_num(st.text_input("n", "400")))
        else:
            col1, col2 = st.columns(2)
            with col1: x = int(parse_num(st.text_input("x éxitos", "200")))
            with col2: n = int(parse_num(st.text_input("n", "400")))
            phat = x / n
        if st.button("Calcular intervalo"):
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

elif categoria == "Pruebas de hipótesis" and opcion in ["Media con Z", "Media con t"]:
    st.subheader(f"Prueba de hipótesis - {opcion}")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mu0 = parse_num(st.text_input("μ₀", "50"))
    with col2: xbar = parse_num(st.text_input("x̄", "53"))
    with col3: desv = parse_num(st.text_input("σ poblacional" if opcion == "Media con Z" else "s muestral", "12"))
    with col4: n = int(parse_num(st.text_input("n", "36" if opcion == "Media con Z" else "25")))
    alpha = parse_num(st.text_input("α", "0.05"))
    tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
    contexto = st.text_area("Contexto para la conclusión", "la media observada debe analizarse frente al valor de referencia planteado.")
    if st.button("Resolver prueba"):
        se = desv / math.sqrt(n)
        estad = (xbar - mu0) / se
        if opcion == "Media con Z":
            p_value = p_valor_normal(estad, tipo); crit = critico_z(alpha, tipo); nombre = "z"; gl_txt = "No aplica"
        else:
            gl = n - 1; p_value = p_valor_t(estad, gl, tipo); crit = critico_t(alpha, gl, tipo); nombre = "t"; gl_txt = str(gl)
        rechaza = p_value < alpha
        st.write(f"gl = {gl_txt}")
        st.write(f"SE = {se:.6f}")
        st.write(f"{nombre} calculado = {estad:.6f}")
        st.write(f"Valor crítico = {crit:.4f}")
        st.write(f"p-valor = {p_value:.6f}")
        st.write(f"α = {alpha:.4f}")
        st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
        h1 = "la media poblacional es diferente al valor de referencia" if tipo == "Bilateral" else ("la media poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la media poblacional es mayor al valor de referencia")
        show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

elif categoria == "Pruebas de hipótesis" and opcion == "Proporción con Z":
    st.subheader("Prueba de hipótesis para proporción con Z")
    p0 = parse_num(st.text_input("p₀", "17%"))
    alpha = parse_num(st.text_input("α", "0.05"))
    modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"])
    if modo == "Ya tengo p̂":
        col1, col2 = st.columns(2)
        with col1: phat = parse_num(st.text_input("p̂", "0.20"))
        with col2: n = int(parse_num(st.text_input("n", "400")))
    else:
        col1, col2 = st.columns(2)
        with col1: x = int(parse_num(st.text_input("x éxitos", "80")))
        with col2: n = int(parse_num(st.text_input("n", "400")))
        phat = x / n
    tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
    contexto = st.text_area("Contexto para la conclusión", "la proporción observada debe compararse con el porcentaje establecido como referencia.")
    if st.button("Resolver prueba"):
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
        h1 = "la proporción poblacional es diferente al valor de referencia" if tipo == "Bilateral" else ("la proporción poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la proporción poblacional es mayor al valor de referencia")
        show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

elif categoria == "Pruebas de hipótesis" and opcion == "Diferencia de medias":
    st.subheader("Prueba de hipótesis para diferencia de medias")
    metodo = st.radio("Método", ["Muestras grandes con Z", "t pooled varianzas iguales", "t Welch varianzas diferentes", "Muestras pareadas"])
    alpha = parse_num(st.text_input("α", "0.05"))
    tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
    d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0"))
    contexto = st.text_area("Contexto para la conclusión", "los promedios de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.")
    if metodo != "Muestras pareadas":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Grupo 1")
            xbar1 = parse_num(st.text_input("x̄₁", "42.48")); s1 = parse_num(st.text_input("s₁", "7.19")); n1 = int(parse_num(st.text_input("n₁", "205")))
        with col2:
            st.markdown("### Grupo 2")
            xbar2 = parse_num(st.text_input("x̄₂", "43.04")); s2 = parse_num(st.text_input("s₂", "7.47")); n2 = int(parse_num(st.text_input("n₂", "195")))
        if st.button("Resolver prueba"):
            if metodo == "Muestras grandes con Z":
                se = math.sqrt((s1**2 / n1) + (s2**2 / n2)); estad = ((xbar1 - xbar2) - d0) / se; p_value = p_valor_normal(estad, tipo); crit = critico_z(alpha, tipo); nombre = "z"; gl_txt = "No aplica"
            elif metodo == "t pooled varianzas iguales":
                gl = n1 + n2 - 2; sp2 = (((n1-1)*s1**2)+((n2-1)*s2**2))/gl; sp = math.sqrt(sp2); se = sp*math.sqrt((1/n1)+(1/n2)); estad = ((xbar1-xbar2)-d0)/se; p_value = p_valor_t(estad, gl, tipo); crit = critico_t(alpha, gl, tipo); nombre = "t"; gl_txt = str(gl)
            else:
                se = math.sqrt((s1**2/n1)+(s2**2/n2)); estad = ((xbar1-xbar2)-d0)/se; num = ((s1**2/n1)+(s2**2/n2))**2; den = ((s1**2/n1)**2)/(n1-1)+((s2**2/n2)**2)/(n2-1); gl = num/den; p_value = p_valor_t(estad, gl, tipo); crit = critico_t(alpha, gl, tipo); nombre="t"; gl_txt=f"{gl:.4f}"
            rechaza = p_value < alpha
            st.write(f"SE = {se:.6f}"); st.write(f"{nombre} calculado = {estad:.6f}"); st.write(f"gl = {gl_txt}"); st.write(f"Valor crítico = {crit:.4f}"); st.write(f"p-valor = {p_value:.6f}")
            st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
            h1 = "existe diferencia entre las medias poblacionales" if tipo == "Bilateral" else ("la media del grupo 1 es menor que la media del grupo 2" if tipo == "Cola izquierda" else "la media del grupo 1 es mayor que la media del grupo 2")
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))
    else:
        st.write("Ingresa datos separados por coma. La diferencia se calcula como después - antes.")
        antes_txt = st.text_area("Antes", "10,12,15,13,11")
        despues_txt = st.text_area("Después", "12,15,16,15,13")
        if st.button("Resolver prueba pareada"):
            antes = np.array([parse_num(v) for v in antes_txt.split(",") if v.strip() != ""]); despues = np.array([parse_num(v) for v in despues_txt.split(",") if v.strip() != ""])
            if len(antes) != len(despues): st.error("Las dos listas deben tener la misma cantidad de datos.")
            elif len(antes) < 2: st.error("Se necesitan mínimo dos pares de datos.")
            else:
                d = despues - antes; n = len(d); dbar = np.mean(d); sd = np.std(d, ddof=1); se = sd/math.sqrt(n); gl = n-1; estad = (dbar-d0)/se; p_value = p_valor_t(estad, gl, tipo); crit = critico_t(alpha, gl, tipo); rechaza = p_value < alpha
                st.write(f"n = {n}"); st.write(f"Promedio de diferencias = {dbar:.6f}"); st.write(f"s_d = {sd:.6f}"); st.write(f"SE = {se:.6f}"); st.write(f"t calculado = {estad:.6f}"); st.write(f"gl = {gl}"); st.write(f"Valor crítico = {crit:.4f}"); st.write(f"p-valor = {p_value:.6f}")
                st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
                h1 = "existe diferencia significativa entre las mediciones pareadas" if tipo == "Bilateral" else ("la diferencia promedio es menor que el valor hipotético" if tipo == "Cola izquierda" else "la diferencia promedio es mayor que el valor hipotético")
                show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

elif categoria == "Pruebas de hipótesis" and opcion == "Diferencia de proporciones":
    st.subheader("Prueba de hipótesis para diferencia de proporciones")
    alpha = parse_num(st.text_input("α", "0.05")); tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"]); d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0"))
    contexto = st.text_area("Contexto para la conclusión", "las proporciones de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.")
    col1, col2 = st.columns(2)
    with col1: st.markdown("### Grupo 1"); x1 = int(parse_num(st.text_input("x₁ éxitos", "80"))); n1 = int(parse_num(st.text_input("n₁", "205")))
    with col2: st.markdown("### Grupo 2"); x2 = int(parse_num(st.text_input("x₂ éxitos", "90"))); n2 = int(parse_num(st.text_input("n₂", "195")))
    if st.button("Resolver prueba"):
        p1 = x1/n1; p2 = x2/n2; p_pool = (x1+x2)/(n1+n2); q_pool = 1-p_pool; se = math.sqrt(p_pool*q_pool*((1/n1)+(1/n2))); z = ((p1-p2)-d0)/se; p_value = p_valor_normal(z, tipo); crit = critico_z(alpha, tipo); rechaza = p_value < alpha
        st.write(f"p̂₁ = {p1:.6f}"); st.write(f"p̂₂ = {p2:.6f}"); st.write(f"p combinada = {p_pool:.6f}"); st.write(f"SE = {se:.6f}"); st.write(f"z calculado = {z:.6f}"); st.write(f"Valor crítico = {crit:.4f}"); st.write(f"p-valor = {p_value:.6f}")
        st.success("Decisión: Se rechaza H₀.") if rechaza else st.warning("Decisión: No se rechaza H₀.")
        h1 = "existe diferencia entre las proporciones poblacionales" if tipo == "Bilateral" else ("la proporción del grupo 1 es menor que la proporción del grupo 2" if tipo == "Cola izquierda" else "la proporción del grupo 1 es mayor que la proporción del grupo 2")
        show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

elif categoria == "Correlación" and opcion == "Datos manuales":
    st.subheader("Correlación de Pearson con datos manuales")
    col1, col2 = st.columns(2)
    with col1: nombre_x = st.text_input("Nombre de X", "X"); datos_x = st.text_area("Valores de X separados por coma", "1,2,3,4,5")
    with col2: nombre_y = st.text_input("Nombre de Y", "Y"); datos_y = st.text_area("Valores de Y separados por coma", "2,4,6,8,10")
    if st.button("Calcular correlación"):
        try:
            x = np.array([parse_num(v) for v in datos_x.split(",") if v.strip() != ""]); y = np.array([parse_num(v) for v in datos_y.split(",") if v.strip() != ""])
            if len(x) != len(y): st.error("X y Y deben tener la misma cantidad de datos.")
            elif len(x) < 2: st.error("Se necesitan mínimo 2 datos.")
            else:
                r = np.corrcoef(x, y)[0,1]; tipo, intensidad = interpretar_correlacion(r); st.success(f"Coeficiente de correlación r = {r:.4f}"); st.write(f"Correlación {tipo} {intensidad} entre {nombre_x} y {nombre_y}.")
                fig, ax = plt.subplots(figsize=(7,5)); ax.scatter(x,y); ax.set_xlabel(nombre_x); ax.set_ylabel(nombre_y); ax.set_title(f"Diagrama de dispersión: {nombre_x} vs {nombre_y}"); st.pyplot(fig)
                if tipo == "positiva": reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo} {intensidad}. A medida que aumenta {nombre_x}, también tiende a aumentar {nombre_y}."
                elif tipo == "negativa": reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo} {intensidad}. A medida que aumenta {nombre_x}, {nombre_y} tiende a disminuir."
                else: reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, por lo que no se observa una relación lineal clara entre {nombre_x} y {nombre_y}."
                show_report(reporte)
        except Exception as e: st.error(f"Error: {e}")

elif categoria == "Correlación" and opcion == "Archivo CSV o Excel":
    st.subheader("Correlación de Pearson con archivo")
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    if archivo is not None:
        try:
            df = leer_csv_excel(archivo); st.success("Base cargada correctamente."); st.write(f"Filas: {df.shape[0]}"); st.write(f"Columnas: {df.shape[1]}"); st.dataframe(df.head())
            numericas = df.select_dtypes(include=["int64", "float64"])
            for col in ["ID", "Id", "id", "Registro", "registro"]:
                if col in numericas.columns: numericas = numericas.drop(columns=[col])
            if numericas.shape[1] < 2: st.error("La base debe tener mínimo dos columnas numéricas.")
            else:
                st.write("Variables numéricas utilizadas:"); st.write(list(numericas.columns)); matriz = numericas.corr(method="pearson"); st.subheader("Matriz de correlación"); st.dataframe(matriz.round(4)); plot_heatmap(matriz); st.subheader("Interpretación automática")
                textos = []
                for i in range(len(matriz.columns)):
                    for j in range(i+1, len(matriz.columns)):
                        var1, var2 = matriz.columns[i], matriz.columns[j]; r = matriz.iloc[i,j]; tipo, intensidad = interpretar_correlacion(r); linea = f"{var1} y {var2}: r = {r:.4f} → correlación {tipo} {intensidad}."; st.write(linea); textos.append(linea)
                st.download_button("Descargar interpretación", data="\n".join(textos), file_name="interpretacion_correlaciones.txt", mime="text/plain")
        except Exception as e: st.error(f"No se pudo leer el archivo: {e}")

elif categoria == "Base de datos" and opcion == "Exploración básica":
    st.subheader("Exploración básica de base de datos")
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    if archivo is not None:
        try:
            df = leer_csv_excel(archivo); st.success("Base cargada correctamente."); st.write(f"Filas: {df.shape[0]}"); st.write(f"Columnas: {df.shape[1]}"); st.markdown("### Vista previa"); st.dataframe(df.head()); st.markdown("### Tipos de datos"); st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipo"])); st.markdown("### Valores faltantes"); faltantes = df.isna().sum().reset_index(); faltantes.columns = ["Variable", "Faltantes"]; st.dataframe(faltantes)
            numericas = df.select_dtypes(include=["int64", "float64"])
            if numericas.shape[1] > 0:
                st.markdown("### Estadística descriptiva"); st.dataframe(numericas.describe().T); variable = st.selectbox("Variable para histograma", numericas.columns); fig, ax = plt.subplots(figsize=(8,5)); ax.hist(numericas[variable].dropna(), bins=20); ax.set_title(f"Histograma de {variable}"); ax.set_xlabel(variable); ax.set_ylabel("Frecuencia"); st.pyplot(fig)
        except Exception as e: st.error(f"No se pudo leer el archivo: {e}")

elif categoria == "Base de datos" and opcion == "Comparación por grupo":
    st.subheader("Comparación de una variable numérica por grupo")
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    if archivo is not None:
        try:
            df = leer_csv_excel(archivo); st.success("Base cargada correctamente."); st.dataframe(df.head()); columnas = list(df.columns); col_grupo = st.selectbox("Variable de grupo", columnas); numericas = df.select_dtypes(include=["int64", "float64"]).columns; col_num = st.selectbox("Variable numérica", numericas)
            if st.button("Comparar"):
                resumen = df.groupby(col_grupo)[col_num].agg(["count", "mean", "std", "min", "max"]); st.dataframe(resumen); fig, ax = plt.subplots(figsize=(8,5)); grupos = [grupo[col_num].dropna().values for _, grupo in df.groupby(col_grupo)]; labels = [str(g) for g in df.groupby(col_grupo).groups.keys()]; ax.boxplot(grupos, labels=labels); ax.set_title(f"{col_num} por {col_grupo}"); ax.set_xlabel(col_grupo); ax.set_ylabel(col_num); st.pyplot(fig)
        except Exception as e: st.error(f"No se pudo leer el archivo: {e}")

elif categoria == "Fórmulas":
    st.subheader("Formulario rápido")
    st.markdown("### Media muestral"); st.latex(r"SE = \frac{\sigma}{\sqrt{n}}"); st.latex(r"z = \frac{\bar{x}-\mu}{\sigma/\sqrt{n}}")
    st.markdown("### Proporción muestral"); st.latex(r"SE = \sqrt{\frac{p(1-p)}{n}}"); st.latex(r"z = \frac{\hat{p}-p}{\sqrt{p(1-p)/n}}")
    st.markdown("### Intervalos"); st.latex(r"IC_\mu = \bar{x} \pm z^* \frac{\sigma}{\sqrt{n}}"); st.latex(r"IC_p = \hat{p} \pm z^* \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
    st.markdown("### Diferencia de medias"); st.latex(r"SE = \sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}"); st.latex(r"z = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{SE}")
    st.markdown("### Muestras pareadas"); st.latex(r"t = \frac{\bar{d}-d_0}{s_d/\sqrt{n}}")


