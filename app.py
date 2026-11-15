
import io
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    acorr_ljungbox
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Análisis estadístico - Abadi y Lezcano",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

ALFA = 0.05

VARIABLES = [
    "Temperatura_Abrigo_150cm",
    "Humedad_Media",
    "Presion_Media",
    "Radiacion_Global"
]

DEPENDIENTE = "Radiacion_Global"
PREDICTOR_SIMPLE = "Temperatura_Abrigo_150cm"

PREDICTORES_MULTIPLES = [
    "Temperatura_Abrigo_150cm",
    "Humedad_Media",
    "Presion_Media"
]

# ============================================================
# CARGA Y VALIDACIÓN DE DATOS
# ============================================================

def cargar_datos(archivo):

    contenido = archivo.getvalue()

    if not contenido:
        raise ValueError("El archivo está vacío.")

    datos = pd.read_csv(
        io.BytesIO(contenido),
        sep=None,
        engine="python"
    )

    obligatorias = ["Fecha"] + VARIABLES

    faltantes = [
        columna for columna in obligatorias
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas: " + ", ".join(faltantes)
        )

    datos = datos[obligatorias].copy()

    # CORRECCIÓN: interpretación de fechas ISO del CSV
    # Ejemplo: 2023-08-10 00:00:00.0

    datos["Fecha"] = pd.to_datetime(
        datos["Fecha"],
        errors="coerce",
        format="mixed"
    )

    for columna in VARIABLES:

        if datos[columna].dtype == object:
            datos[columna] = (
                datos[columna]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    if datos["Fecha"].isna().any():
        raise ValueError(
            "Hay fechas vacías o con formato incorrecto."
        )

    if datos[VARIABLES].isna().any().any():
        raise ValueError(
            "Hay valores faltantes o no numéricos."
        )

    datos = datos.sort_values("Fecha").reset_index(drop=True)

    return datos


# ============================================================
# ESTADÍSTICA DESCRIPTIVA
# ============================================================

def tabla_descriptiva(datos):

    tabla = datos[VARIABLES].describe().T

    tabla["varianza"] = datos[VARIABLES].var()
    tabla["mediana"] = datos[VARIABLES].median()
    tabla["Q1"] = datos[VARIABLES].quantile(0.25)
    tabla["Q3"] = datos[VARIABLES].quantile(0.75)

    tabla["IQR"] = tabla["Q3"] - tabla["Q1"]

    tabla["CV_%"] = (
        datos[VARIABLES].std()
        / datos[VARIABLES].mean().abs()
    ) * 100

    tabla["asimetria"] = datos[VARIABLES].skew()
    tabla["curtosis"] = datos[VARIABLES].kurt()

    return tabla


# ============================================================
# MODELOS DE REGRESIÓN
# ============================================================

def ajustar_modelos(datos):

    y = datos[DEPENDIENTE]

    X_simple = sm.add_constant(
        datos[[PREDICTOR_SIMPLE]]
    )

    modelo_simple = sm.OLS(y, X_simple).fit()

    X_multiple = sm.add_constant(
        datos[PREDICTORES_MULTIPLES]
    )

    modelo_multiple = sm.OLS(y, X_multiple).fit()

    return modelo_simple, modelo_multiple


# ============================================================
# PRUEBAS ESTADÍSTICAS DE RESIDUOS
# ============================================================

def mostrar_pruebas_residuos(modelo, nombre):

    residuos = modelo.resid
    exog = modelo.model.exog

    shapiro = stats.shapiro(residuos)
    jb = stats.jarque_bera(residuos)

    bp = het_breuschpagan(residuos, exog)

    dw = durbin_watson(residuos)

    lb = acorr_ljungbox(
        residuos,
        lags=[10],
        return_df=True
    )

    resultados = pd.DataFrame([
        [
            "Shapiro-Wilk",
            shapiro.statistic,
            shapiro.pvalue
        ],
        [
            "Jarque-Bera",
            jb.statistic,
            jb.pvalue
        ],
        [
            "Breusch-Pagan",
            bp[0],
            bp[1]
        ],
        [
            "Durbin-Watson",
            dw,
            np.nan
        ],
        [
            "Ljung-Box (rezago 10)",
            lb["lb_stat"].iloc[0],
            lb["lb_pvalue"].iloc[0]
        ]
    ], columns=[
        "Prueba",
        "Estadístico",
        "p-valor"
    ])

    st.subheader(f"Pruebas de residuos: {nombre}")

    st.dataframe(
        resultados.round(5),
        use_container_width=True
    )


# ============================================================
# GRÁFICOS DE RESIDUOS
# ============================================================

def grafico_residuos(modelo, nombre):

    residuos = modelo.resid
    ajustados = modelo.fittedvalues

    fig, ejes = plt.subplots(
        2, 2,
        figsize=(12, 8)
    )

    ejes[0, 0].scatter(
        ajustados,
        residuos,
        alpha=0.55
    )

    ejes[0, 0].axhline(
        0,
        color="red",
        linestyle="--"
    )

    ejes[0, 0].set_title(
        "Residuos vs. valores ajustados"
    )

    sm.qqplot(
        residuos,
        line="45",
        ax=ejes[0, 1]
    )

    ejes[0, 1].set_title("Gráfico Q-Q")

    ejes[1, 0].plot(residuos.values)

    ejes[1, 0].axhline(
        0,
        color="red",
        linestyle="--"
    )

    ejes[1, 0].set_title(
        "Residuos por observación"
    )

    sns.histplot(
        residuos,
        kde=True,
        ax=ejes[1, 1]
    )

    ejes[1, 1].set_title(
        "Distribución de residuos"
    )

    fig.suptitle(nombre)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 1. ANÁLISIS ESTRUCTURAL
# ============================================================

def mostrar_datos(datos):

    st.title("1. Análisis estructural de los datos")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observaciones", len(datos))
    c2.metric("Variables", len(datos.columns))

    c3.metric(
        "Valores faltantes",
        int(datos.isna().sum().sum())
    )

    c4.metric(
        "Filas duplicadas",
        int(datos.duplicated().sum())
    )

    st.write(
        f"Período: {datos['Fecha'].min().date()} "
        f"al {datos['Fecha'].max().date()}"
    )

    st.subheader("Vista de los datos")

    st.dataframe(
        datos.head(20),
        use_container_width=True
    )


# ============================================================
# 2. DESCRIPCIÓN NUMÉRICA
# ============================================================

def mostrar_descriptiva(datos):

    st.title("2. Descripción numérica")

    tabla = tabla_descriptiva(datos)

    st.dataframe(
        tabla.round(4),
        use_container_width=True
    )

    st.subheader("Valores atípicos")

    filas = []

    for variable in VARIABLES:

        q1 = datos[variable].quantile(0.25)
        q3 = datos[variable].quantile(0.75)

        iqr = q3 - q1

        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr

        cantidad = (
            (datos[variable] < inferior)
            | (datos[variable] > superior)
        ).sum()

        filas.append([
            variable,
            inferior,
            superior,
            cantidad
        ])

    st.dataframe(
        pd.DataFrame(
            filas,
            columns=[
                "Variable",
                "Límite inferior",
                "Límite superior",
                "Atípicos"
            ]
        ),
        use_container_width=True
    )

    st.subheader("Normalidad de las variables")

    resultados = []

    for variable in VARIABLES:

        resultado = stats.shapiro(
            datos[variable]
        )

        resultados.append([
            variable,
            resultado.statistic,
            resultado.pvalue
        ])

    st.dataframe(
        pd.DataFrame(
            resultados,
            columns=[
                "Variable",
                "W",
                "p-valor"
            ]
        ),
        use_container_width=True
    )


# ============================================================
# 3. DESCRIPCIÓN GRÁFICA
# ============================================================

def mostrar_graficos(datos):

    st.title("3. Descripción gráfica")

    for variable in VARIABLES:

        with st.expander(variable):

            fig, ejes = plt.subplots(
                1, 2,
                figsize=(11, 4)
            )

            sns.histplot(
                datos[variable],
                kde=True,
                ax=ejes[0]
            )

            ejes[0].set_title("Histograma")

            sns.boxplot(
                x=datos[variable],
                ax=ejes[1]
            )

            ejes[1].set_title("Diagrama de caja")

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# 4. SERIES TEMPORALES
# ============================================================

def mostrar_temporales(datos):

    st.title("4. Series temporales")

    seleccion = st.multiselect(
        "Seleccionar variables",
        VARIABLES,
        default=VARIABLES
    )

    if not seleccion:
        st.warning("Seleccioná alguna variable.")
        return

    for variable in seleccion:

        fig, ax = plt.subplots(figsize=(12, 3))

        ax.plot(
            datos["Fecha"],
            datos[variable]
        )

        ax.set_title(variable)

        ax.set_xlabel("Fecha")

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# 5. CORRELACIONES
# ============================================================

def mostrar_correlaciones(datos):

    st.title("5. Matriz de correlación")

    matriz = datos[VARIABLES].corr()

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        matriz,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)

    st.subheader(
        "Correlaciones con radiación global"
    )

    correlaciones = (
        matriz[DEPENDIENTE]
        .drop(DEPENDIENTE)
        .sort_values()
    )

    st.dataframe(correlaciones)


# ============================================================
# 6. CONCLUSIÓN DESCRIPTIVA
# ============================================================

def mostrar_conclusion_descriptiva(datos):

    st.title("6. Conclusión descriptiva")

    tabla = tabla_descriptiva(datos)

    correlaciones = (
        datos[VARIABLES]
        .corr()[DEPENDIENTE]
        .drop(DEPENDIENTE)
    )

    st.write(
        f"La base contiene {len(datos)} observaciones."
    )

    st.write(
        f"La variable con mayor variabilidad relativa "
        f"es {tabla['CV_%'].idxmax()}."
    )

    st.write(
        f"La variable con menor variabilidad relativa "
        f"es {tabla['CV_%'].idxmin()}."
    )

    st.dataframe(
        correlaciones.sort_values().to_frame(
            "Correlación de Pearson"
        )
    )


# ============================================================
# 7. REGRESIÓN SIMPLE
# ============================================================

def mostrar_regresion_simple(modelo):

    st.title("7. Regresión lineal simple")

    b0 = modelo.params["const"]

    b1 = modelo.params[PREDICTOR_SIMPLE]

    c1, c2, c3 = st.columns(3)

    c1.metric("Intercepto", f"{b0:.4f}")

    c2.metric("Pendiente", f"{b1:.4f}")

    c3.metric("R²", f"{modelo.rsquared:.4f}")

    st.write(
        f"Radiacion_Global = {b0:.4f} + "
        f"({b1:.4f} × Temperatura_Abrigo_150cm)"
    )

    st.write(
        f"El modelo explica el "
        f"{modelo.rsquared * 100:.2f}% "
        f"de la variabilidad de la radiación global."
    )

    st.text(modelo.summary().as_text())

    fig, ax = plt.subplots(figsize=(8, 4))

    x = modelo.model.exog[:, 1]

    ax.scatter(
        x,
        modelo.model.endog,
        alpha=0.5
    )

    orden = np.argsort(x)

    ax.plot(
        x[orden],
        modelo.fittedvalues.iloc[orden],
        color="red"
    )

    ax.set_xlabel(PREDICTOR_SIMPLE)

    ax.set_ylabel(DEPENDIENTE)

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 8. RESIDUOS
# ============================================================

def mostrar_residuos_simple(modelo):

    st.title("8. Análisis de residuos")

    mostrar_pruebas_residuos(
        modelo,
        "Regresión simple"
    )

    grafico_residuos(
        modelo,
        "Regresión lineal simple"
    )


# ============================================================
# 9. TEST DE HIPÓTESIS
# ============================================================

def mostrar_test_pendiente(modelo):

    st.title("9. Test de hipótesis sobre la pendiente")

    beta = modelo.params[PREDICTOR_SIMPLE]

    error = modelo.bse[PREDICTOR_SIMPLE]

    t_obs = modelo.tvalues[PREDICTOR_SIMPLE]

    p_valor = modelo.pvalues[PREDICTOR_SIMPLE]

    gl = int(modelo.df_resid)

    t_critico = stats.t.ppf(
        1 - ALFA / 2,
        gl
    )

    ic = modelo.conf_int(alpha=ALFA).loc[
        PREDICTOR_SIMPLE
    ]

    st.markdown("""
    **Hipótesis nula:** H₀: β₁ = 0

    **Hipótesis alternativa:** H₁: β₁ ≠ 0

    **Nivel de significación:** α = 0,05
    """)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Pendiente", f"{beta:.5f}")

    c2.metric("Error estándar", f"{error:.5f}")

    c3.metric("t observado", f"{t_obs:.4f}")

    c4.metric("p-valor", f"{p_valor:.5g}")

    st.write(
        f"Valor crítico: ±{t_critico:.4f}"
    )

    st.write(
        f"Intervalo de confianza: "
        f"[{ic.iloc[0]:.4f}, {ic.iloc[1]:.4f}]"
    )

    if p_valor < ALFA:

        st.success(
            "Se rechaza H₀. Existe evidencia estadística "
            "de una relación lineal."
        )

    else:

        st.info(
            "No se rechaza H₀. No existe evidencia "
            "estadística suficiente."
        )


# ============================================================
# 10. ERRORES ROBUSTOS NEWEY-WEST
# ============================================================

def mostrar_newey_west(modelo):

    st.title("10. Errores robustos Newey-West")

    maxlags = min(
        10,
        int(modelo.nobs) - 1
    )

    robusto = modelo.get_robustcov_results(
        cov_type="HAC",
        maxlags=maxlags
    )

    indice = list(modelo.params.index).index(
        PREDICTOR_SIMPLE
    )

    st.write(
        f"Rezagos HAC utilizados: {maxlags}"
    )

    resultados = pd.DataFrame({
        "Indicador": [
            "Pendiente",
            "Error estándar robusto",
            "Estadístico t",
            "p-valor"
        ],
        "Resultado": [
            modelo.params[PREDICTOR_SIMPLE],
            robusto.bse[indice],
            robusto.tvalues[indice],
            robusto.pvalues[indice]
        ]
    })

    st.dataframe(resultados)

    if robusto.pvalues[indice] < ALFA:

        st.success(
            "La pendiente continúa siendo significativa "
            "con errores robustos."
        )

    else:

        st.warning(
            "La pendiente no resulta significativa "
            "con errores robustos."
        )


# ============================================================
# 11. REGRESIÓN MÚLTIPLE
# ============================================================

def mostrar_multiple(modelo_simple, modelo_multiple):

    st.title("11. Regresión lineal múltiple")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("R²", f"{modelo_multiple.rsquared:.4f}")

    c2.metric(
        "R² ajustado",
        f"{modelo_multiple.rsquared_adj:.4f}"
    )

    c3.metric("AIC", f"{modelo_multiple.aic:.2f}")

    c4.metric(
        "p-valor F",
        f"{modelo_multiple.f_pvalue:.5g}"
    )

    st.subheader("Coeficientes")

    tabla = pd.DataFrame({
        "Coeficiente": modelo_multiple.params,
        "Error estándar": modelo_multiple.bse,
        "t": modelo_multiple.tvalues,
        "p-valor": modelo_multiple.pvalues
    })

    st.dataframe(tabla.round(5))

    st.text(
        modelo_multiple.summary().as_text()
    )

    grafico_residuos(
        modelo_multiple,
        "Regresión múltiple"
    )


# ============================================================
# 12. DIAGNÓSTICO Y COMPARACIÓN
# ============================================================

def mostrar_diagnostico_multiple(
    modelo_simple,
    modelo_multiple,
    datos
):

    st.title("12. Diagnóstico y comparación de modelos")

    mostrar_pruebas_residuos(
        modelo_multiple,
        "Regresión múltiple"
    )

    st.subheader("Multicolinealidad: VIF")

    X = sm.add_constant(
        datos[PREDICTORES_MULTIPLES]
    )

    vif = []

    for i, variable in enumerate(X.columns):

        if variable != "const":

            vif.append([
                variable,
                variance_inflation_factor(
                    X.values,
                    i
                )
            ])

    st.dataframe(
        pd.DataFrame(
            vif,
            columns=["Variable", "VIF"]
        )
    )

    st.subheader("Comparación de modelos")

    comparacion = pd.DataFrame({
        "Indicador": ["R²", "AIC", "BIC"],
        "Modelo simple": [
            modelo_simple.rsquared,
            modelo_simple.aic,
            modelo_simple.bic
        ],
        "Modelo múltiple": [
            modelo_multiple.rsquared,
            modelo_multiple.aic,
            modelo_multiple.bic
        ]
    })

    st.dataframe(
        comparacion.round(4)
    )

    resultado = modelo_multiple.compare_f_test(
        modelo_simple
    )

    st.write(
        f"Test F parcial: F = {resultado[0]:.4f}"
    )

    st.write(
        f"p-valor = {resultado[1]:.6g}"
    )

    if resultado[1] < ALFA:

        st.success(
            "El modelo múltiple mejora significativamente "
            "el ajuste respecto del modelo simple."
        )

    else:

        st.info(
            "No se observa una mejora estadísticamente "
            "significativa."
        )


# ============================================================
# 13. CONCLUSIONES
# ============================================================

def mostrar_conclusiones(
    datos,
    modelo_simple,
    modelo_multiple
):

    st.title("13. Conclusiones generales")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Observaciones",
        len(datos)
    )

    c2.metric(
        "R² simple",
        f"{modelo_simple.rsquared:.4f}"
    )

    c3.metric(
        "R² múltiple",
        f"{modelo_multiple.rsquared:.4f}"
    )

    mejora = (
        modelo_multiple.rsquared
        - modelo_simple.rsquared
    ) * 100

    st.write(
        f"El modelo simple explica el "
        f"{modelo_simple.rsquared * 100:.2f}% "
        f"de la variabilidad de la radiación global."
    )

    st.write(
        f"El modelo múltiple explica el "
        f"{modelo_multiple.rsquared * 100:.2f}%."
    )

    st.write(
        f"La mejora del modelo múltiple es de "
        f"{mejora:.2f} puntos porcentuales."
    )

    st.info(
        "La asociación estadística no implica necesariamente "
        "causalidad. Las conclusiones deben interpretarse "
        "considerando los diagnósticos de residuos."
    )


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("📊 Trabajo práctico integrador")

st.subheader(
    "Métodos Estadísticos para Inteligencia Artificial"
)

st.caption(
    "Abadi y Lezcano | Estación meteorológica Castelar"
)

st.markdown(
    "Aplicación interactiva para analizar los datos "
    "meteorológicos de la estación Castelar."
)

with st.sidebar:

    st.header("Carga de datos")

    archivo = st.file_uploader(
        "Subir archivo CSV",
        type=["csv"]
    )

    st.caption(
        "La base debe contener Fecha y las cuatro "
        "variables del trabajo."
    )

    st.divider()

    st.caption("Nivel de significación: α = 0,05")


if archivo is None:

    st.info(
        "Para comenzar, cargá el archivo "
        "01_Buenos_Aires_Castelar.csv."
    )

    st.markdown("""
    **Columnas requeridas:**

    - Fecha
    - Temperatura_Abrigo_150cm
    - Humedad_Media
    - Presion_Media
    - Radiacion_Global
    """)

    st.stop()


try:

    datos = cargar_datos(archivo)

except Exception as error:

    st.error(
        f"No se pudo cargar la base: {error}"
    )

    st.stop()


if len(datos) < 12:

    st.error(
        "Se necesitan al menos 12 observaciones."
    )

    st.stop()


modelo_simple, modelo_multiple = ajustar_modelos(
    datos
)


with st.sidebar:

    st.success(
        f"Base cargada: {len(datos)} observaciones"
    )

    seccion = st.radio(
        "Secciones del trabajo",
        [
            "1. Análisis estructural",
            "2. Descripción numérica",
            "3. Descripción gráfica",
            "4. Series temporales",
            "5. Correlaciones",
            "6. Conclusión descriptiva",
            "7. Regresión lineal simple",
            "8. Residuos del modelo simple",
            "9. Test de hipótesis",
            "10. Errores robustos Newey-West",
            "11. Regresión lineal múltiple",
            "12. Diagnóstico y comparación",
            "13. Conclusiones generales"
        ]
    )


if seccion.startswith("1."):

    mostrar_datos(datos)

elif seccion.startswith("2."):

    mostrar_descriptiva(datos)

elif seccion.startswith("3."):

    mostrar_graficos(datos)

elif seccion.startswith("4."):

    mostrar_temporales(datos)

elif seccion.startswith("5."):

    mostrar_correlaciones(datos)

elif seccion.startswith("6."):

    mostrar_conclusion_descriptiva(datos)

elif seccion.startswith("7."):

    mostrar_regresion_simple(modelo_simple)

elif seccion.startswith("8."):

    mostrar_residuos_simple(modelo_simple)

elif seccion.startswith("9."):

    mostrar_test_pendiente(modelo_simple)

elif seccion.startswith("10."):

    mostrar_newey_west(modelo_simple)

elif seccion.startswith("11."):

    mostrar_multiple(
        modelo_simple,
        modelo_multiple
    )

elif seccion.startswith("12."):

    mostrar_diagnostico_multiple(
        modelo_simple,
        modelo_multiple,
        datos
    )

else:

    mostrar_conclusiones(
        datos,
        modelo_simple,
        modelo_multiple
    )
