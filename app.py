
from pathlib import Path
import io
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Trabajo práctico integrador | Métodos Estadísticos",
    page_icon="📊",
    layout="wide",
)

sns.set_theme(style="whitegrid")
ALFA = 0.05

VARIABLES = [
    "Temperatura_Abrigo_150cm",
    "Humedad_Media",
    "Presion_Media",
    "Radiacion_Global",
]

DEPENDIENTE = "Radiacion_Global"
PREDICTOR_SIMPLE = "Temperatura_Abrigo_150cm"

PREDICTORES_MULTIPLES = [
    "Temperatura_Abrigo_150cm",
    "Humedad_Media",
    "Presion_Media",
]

ARCHIVO_EJEMPLO = "01_Buenos_Aires_Castelar.csv"


# --------------------------------------------------
# CARGA Y VALIDACIÓN DE DATOS
# --------------------------------------------------

def cargar_datos(archivo):
    """Lee y valida el CSV utilizado en el trabajo."""

    contenido = archivo.getvalue()

    if not contenido:
        raise ValueError("El archivo CSV está vacío.")

    try:
        datos = pd.read_csv(io.BytesIO(contenido), sep=";")

        if len(datos.columns) == 1:
            datos = pd.read_csv(io.BytesIO(contenido), sep=",")

    except Exception as exc:
        raise ValueError(
            "No se pudo interpretar el CSV. "
            "Revisá el separador y los encabezados."
        ) from exc

    if datos.empty:
        raise ValueError("El archivo CSV no contiene registros.")

    obligatorias = ["Fecha"] + VARIABLES

    faltantes = [
        c for c in obligatorias
        if c not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(faltantes)
        )

    datos = datos[obligatorias].copy()

    # Conversión de fechas compatible con distintos formatos.
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

        columnas = (
            datos[VARIABLES]
            .columns[datos[VARIABLES].isna().any()]
            .tolist()
        )

        raise ValueError(
            "Hay valores faltantes o no numéricos en: "
            + ", ".join(columnas)
        )

    datos = (
        datos
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    return datos


# --------------------------------------------------
# ESTADÍSTICA DESCRIPTIVA
# --------------------------------------------------

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
    tabla["curtosis_exceso"] = datos[VARIABLES].kurt()

    return tabla


# --------------------------------------------------
# MODELOS DE REGRESIÓN
# --------------------------------------------------

def ajustar_modelos(datos):

    y = datos[DEPENDIENTE]

    X_simple = sm.add_constant(
        datos[[PREDICTOR_SIMPLE]]
    )

    modelo_simple = sm.OLS(
        y,
        X_simple
    ).fit()

    X_multiple = sm.add_constant(
        datos[PREDICTORES_MULTIPLES]
    )

    modelo_multiple = sm.OLS(
        y,
        X_multiple
    ).fit()

    return modelo_simple, modelo_multiple


def interpretar_pvalor(p, alfa=ALFA):

    return (
        "Se rechaza H₀"
        if p < alfa
        else "No se rechaza H₀"
    )


# --------------------------------------------------
# PRUEBAS Y DIAGNÓSTICO DE RESIDUOS
# --------------------------------------------------

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
            f"{shapiro.statistic:.4f}",
            shapiro.pvalue,
            interpretar_pvalor(shapiro.pvalue)
        ],
        [
            "Jarque-Bera",
            f"{jb.statistic:.4f}",
            jb.pvalue,
            interpretar_pvalor(jb.pvalue)
        ],
        [
            "Breusch-Pagan",
            f"{bp[0]:.4f}",
            bp[1],
            interpretar_pvalor(bp[1])
        ],
        [
            "Durbin-Watson",
            f"{dw:.4f}",
            np.nan,
            "Valor de referencia cercano a 2"
        ],
        [
            "Ljung-Box (rezago 10)",
            f"{lb['lb_stat'].iloc[0]:.4f}",
            lb["lb_pvalue"].iloc[0],
            interpretar_pvalor(lb["lb_pvalue"].iloc[0])
        ],
    ], columns=[
        "Prueba",
        "Estadístico",
        "p-valor",
        "Interpretación"
    ])

    st.subheader(f"Pruebas de residuos — {nombre}")

    st.dataframe(
        resultados,
        use_container_width=True,
        hide_index=True
    )

    return resultados


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

    ejes[0, 0].set_xlabel("Valores ajustados")
    ejes[0, 0].set_ylabel("Residuos")

    sm.qqplot(
        residuos,
        line="45",
        ax=ejes[0, 1]
    )

    ejes[0, 1].set_title("Gráfico Q-Q")

    ejes[1, 0].plot(
        residuos.index,
        residuos,
        alpha=0.75
    )

    ejes[1, 0].axhline(
        0,
        color="red",
        linestyle="--"
    )

    ejes[1, 0].set_title("Residuos en el tiempo")
    ejes[1, 0].set_xlabel("Observación")
    ejes[1, 0].set_ylabel("Residuo")

    sns.histplot(
        residuos,
        kde=True,
        ax=ejes[1, 1]
    )

    ejes[1, 1].set_title(
        "Distribución de los residuos"
    )

    fig.suptitle(
        f"Diagnóstico de residuos — {nombre}",
        fontsize=14
    )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def mostrar_resumen_modelo(modelo):

    st.text(
        modelo.summary().as_text()
    )


# --------------------------------------------------
# 1. ANÁLISIS ESTRUCTURAL
# --------------------------------------------------

def mostrar_datos(datos):

    st.title("1. Análisis estructural de los datos")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Registros",
        f"{len(datos):,}".replace(",", ".")
    )

    c2.metric(
        "Variables",
        len(datos.columns)
    )

    c3.metric(
        "Valores faltantes",
        int(datos.isna().sum().sum())
    )

    c4.metric(
        "Filas duplicadas",
        int(datos.duplicated().sum())
    )

    st.write(
        f"**Período:** "
        f"{datos['Fecha'].min().date()} al "
        f"{datos['Fecha'].max().date()}"
    )

    diferencias = (
        datos["Fecha"]
        .diff()
        .dt.days
        .dropna()
    )

    st.write(
        f"**Intervalos de un día:** "
        f"{(diferencias == 1).sum()} · "
        f"**Otros intervalos:** "
        f"{(diferencias != 1).sum()}"
    )

    st.subheader("Vista de los datos")

    st.dataframe(
        datos.head(20),
        use_container_width=True
    )

    st.subheader("Tipos de variables")

    tipos = pd.DataFrame({
        "Variable": datos.dtypes.index,
        "Tipo": datos.dtypes.astype(str).values
    })

    st.dataframe(
        tipos,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 2. DESCRIPCIÓN NUMÉRICA
# --------------------------------------------------

def mostrar_descriptiva(datos):

    st.title("2. Descripción numérica")

    tabla = tabla_descriptiva(datos)

    columnas = [
        "count",
        "mean",
        "mediana",
        "std",
        "varianza",
        "min",
        "Q1",
        "50%",
        "Q3",
        "max",
        "IQR",
        "CV_%",
        "asimetria",
        "curtosis_exceso"
    ]

    st.dataframe(
        tabla[columnas].round(4),
        use_container_width=True
    )

    mayor = tabla["CV_%"].idxmax()
    menor = tabla["CV_%"].idxmin()

    st.info(
        f"Mayor variabilidad relativa: **{mayor}**. "
        f"Menor variabilidad relativa: **{menor}**."
    )

    st.subheader(
        "Valores atípicos — criterio de Tukey"
    )

    filas = []

    for variable in VARIABLES:

        q1, q3 = datos[variable].quantile(
            [0.25, 0.75]
        )

        iqr = q3 - q1

        li = q1 - 1.5 * iqr
        ls = q3 + 1.5 * iqr

        cantidad = int(
            (
                (datos[variable] < li)
                | (datos[variable] > ls)
            ).sum()
        )

        filas.append([
            variable,
            li,
            ls,
            cantidad
        ])

    st.dataframe(
        pd.DataFrame(
            filas,
            columns=[
                "Variable",
                "Límite inferior",
                "Límite superior",
                "Cantidad de atípicos"
            ]
        ).round(3),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Pruebas de normalidad de las variables"
    )

    norm = []

    for variable in VARIABLES:

        resultado = stats.shapiro(
            datos[variable]
        )

        norm.append([
            variable,
            resultado.statistic,
            resultado.pvalue,
            interpretar_pvalor(resultado.pvalue)
        ])

    st.dataframe(
        pd.DataFrame(
            norm,
            columns=[
                "Variable",
                "W",
                "p-valor",
                "Interpretación"
            ]
        ).round(6),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 3. DESCRIPCIÓN GRÁFICA
# --------------------------------------------------

def mostrar_graficos(datos):

    st.title("3. Descripción gráfica")

    for variable in VARIABLES:

        with st.expander(
            variable,
            expanded=(variable == DEPENDIENTE)
        ):

            fig, ejes = plt.subplots(
                1, 2,
                figsize=(11, 3.5)
            )

            sns.histplot(
                datos[variable],
                kde=True,
                ax=ejes[0],
                color="skyblue"
            )

            ejes[0].set_title(
                f"Distribución de {variable}"
            )

            sns.boxplot(
                x=datos[variable],
                ax=ejes[1],
                color="lightgreen"
            )

            ejes[1].set_title(
                "Diagrama de caja"
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


# --------------------------------------------------
# 4. SERIES TEMPORALES
# --------------------------------------------------

def mostrar_temporales(datos):

    st.title("4. Series temporales")

    seleccion = st.multiselect(
        "Variables para visualizar",
        VARIABLES,
        default=VARIABLES
    )

    if not seleccion:

        st.warning(
            "Seleccioná al menos una variable."
        )

        return

    fig, ejes = plt.subplots(
        len(seleccion),
        1,
        figsize=(12, max(3, 2.7 * len(seleccion))),
        sharex=True,
        squeeze=False
    )

    for eje, variable in zip(
        ejes[:, 0],
        seleccion
    ):

        eje.plot(
            datos["Fecha"],
            datos[variable],
            linewidth=0.8
        )

        eje.set_title(variable)
        eje.set_ylabel("Valor")

    ejes[-1, 0].set_xlabel("Fecha")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 5. MATRIZ DE CORRELACIÓN
# --------------------------------------------------

def mostrar_correlaciones(datos):

    st.title("5. Matriz de correlación")

    matriz = datos[VARIABLES].corr()

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        matriz,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        matriz.round(4),
        use_container_width=True
    )

    correl = (
        matriz[DEPENDIENTE]
        .drop(DEPENDIENTE)
        .sort_values()
    )

    st.subheader(
        "Correlaciones con Radiacion_Global"
    )

    st.dataframe(
        correl.rename(
            "Correlación de Pearson"
        ).to_frame(),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    sns.regplot(
        data=datos,
        x=PREDICTOR_SIMPLE,
        y=DEPENDIENTE,
        scatter_kws={"alpha": 0.45},
        ax=ax
    )

    ax.set_title(
        "Relación entre temperatura y radiación global"
    )

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 6. CONCLUSIÓN DESCRIPTIVA
# --------------------------------------------------

def mostrar_conclusion_descriptiva(datos):

    tabla = tabla_descriptiva(datos)

    correl = (
        datos[VARIABLES]
        .corr()[DEPENDIENTE]
        .drop(DEPENDIENTE)
    )

    st.title(
        "6. Conclusión de estadística descriptiva"
    )

    st.write(
        f"La base contiene **{len(datos)} observaciones**, "
        f"desde **{datos['Fecha'].min().date()}** "
        f"hasta **{datos['Fecha'].max().date()}**."
    )

    st.write(
        f"La variable con mayor variabilidad relativa "
        f"(CV) es **{tabla['CV_%'].idxmax()}**; "
        f"la de menor variabilidad relativa es "
        f"**{tabla['CV_%'].idxmin()}**."
    )

    st.write(
        "Correlaciones con Radiacion_Global:"
    )

    st.dataframe(
        correl.sort_values().rename("r").to_frame(),
        use_container_width=True
    )


# --------------------------------------------------
# 7. REGRESIÓN LINEAL SIMPLE
# --------------------------------------------------

def mostrar_regresion_simple(modelo):

    st.title("7. Regresión lineal simple")

    b0 = modelo.params["const"]
    b1 = modelo.params[PREDICTOR_SIMPLE]

    st.latex(
        r"\widehat{Radiacion\_Global} = "
        r"\beta_0 + \beta_1 \cdot Temperatura"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Intercepto", f"{b0:.4f}")
    c2.metric("Pendiente", f"{b1:.4f}")
    c3.metric("R²", f"{modelo.rsquared:.4f}")

    st.write(
        f"**Ecuación estimada:** "
        f"Radiacion_Global = {b0:.4f} + "
        f"({b1:.4f} × Temperatura_Abrigo_150cm)"
    )

    st.write(
        f"El modelo explica el "
        f"**{modelo.rsquared * 100:.2f}%** "
        f"de la variabilidad de la radiación global."
    )

    st.subheader("Resumen estadístico")

    mostrar_resumen_modelo(modelo)

    st.subheader("Recta de regresión")

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.scatter(
        modelo.model.exog[:, 1],
        modelo.model.endog,
        alpha=0.45
    )

    orden = np.argsort(
        modelo.model.exog[:, 1]
    )

    ax.plot(
        modelo.model.exog[orden, 1],
        modelo.fittedvalues.iloc[orden],
        color="red"
    )

    ax.set_xlabel(PREDICTOR_SIMPLE)
    ax.set_ylabel(DEPENDIENTE)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 8. RESIDUOS DEL MODELO SIMPLE
# --------------------------------------------------

def mostrar_residuos_simple(modelo):

    st.title(
        "8. Análisis de residuos del modelo simple"
    )

    st.write(
        "Se evalúan normalidad, homocedasticidad "
        "e independencia de los errores mediante "
        "pruebas estadísticas y gráficos de diagnóstico."
    )

    mostrar_pruebas_residuos(
        modelo,
        "Regresión lineal simple"
    )

    grafico_residuos(
        modelo,
        "Regresión lineal simple"
    )


# --------------------------------------------------
# 9. TEST DE HIPÓTESIS SOBRE LA PENDIENTE
# --------------------------------------------------

def mostrar_test_pendiente(modelo):

    st.title(
        "9. Test de hipótesis sobre la pendiente"
    )

    beta = modelo.params[PREDICTOR_SIMPLE]
    error = modelo.bse[PREDICTOR_SIMPLE]
    t_obs = modelo.tvalues[PREDICTOR_SIMPLE]
    p_valor = modelo.pvalues[PREDICTOR_SIMPLE]

    gl = int(modelo.df_resid)

    t_critico = stats.t.ppf(
        1 - ALFA / 2,
        gl
    )

    ic_inf, ic_sup = (
        modelo.conf_int(alpha=ALFA)
        .loc[PREDICTOR_SIMPLE]
    )

    st.markdown(
        f"- **H₀:** β₁ = 0\n"
        f"- **H₁:** β₁ ≠ 0\n"
        f"- **Nivel de significación:** α = {ALFA}\n"
        f"- **Grados de libertad:** {gl}"
    )

    a, b, c, d = st.columns(4)

    a.metric("Pendiente", f"{beta:.6f}")
    b.metric("Error estándar", f"{error:.6f}")
    c.metric("t observado", f"{t_obs:.4f}")
    d.metric("p-valor", f"{p_valor:.4g}")

    st.write(
        f"Valor crítico bilateral: ±{t_critico:.4f}"
    )

    st.write(
        f"Intervalo de confianza del 95%: "
        f"[{ic_inf:.4f}, {ic_sup:.4f}]"
    )

    if p_valor < ALFA:

        st.success(
            "Se rechaza H₀: existe evidencia estadística "
            "de una relación lineal entre temperatura "
            "y radiación global."
        )

    else:

        st.info(
            "No se rechaza H₀: no hay evidencia estadística "
            "suficiente para afirmar una relación lineal."
        )

    xs = np.linspace(
        -max(5, abs(t_obs) + 1),
        max(5, abs(t_obs) + 1),
        500
    )

    fig, ax = plt.subplots(
        figsize=(8, 3)
    )

    ax.plot(
        xs,
        stats.t.pdf(xs, gl)
    )

    ax.fill_between(
        xs,
        stats.t.pdf(xs, gl),
        where=np.abs(xs) >= t_critico,
        alpha=0.3,
        color="red"
    )

    ax.axvline(
        t_obs,
        color="blue",
        linestyle="--",
        label=f"t observado = {t_obs:.2f}"
    )

    ax.set_title(
        "Distribución t bajo H₀"
    )

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 10. ERRORES ROBUSTOS NEWEY-WEST
# --------------------------------------------------

def mostrar_newey_west(modelo):

    st.title(
        "10. Supuestos y errores robustos Newey-West"
    )

    n = int(modelo.nobs)
    maxlags = min(10, n - 1)

    robusto = modelo.get_robustcov_results(
        cov_type="HAC",
        maxlags=maxlags
    )

    indice = list(
        modelo.params.index
    ).index(PREDICTOR_SIMPLE)

    beta = modelo.params[PREDICTOR_SIMPLE]
    se = robusto.bse[indice]
    t = robusto.tvalues[indice]
    p = robusto.pvalues[indice]

    ic = robusto.conf_int(
        alpha=ALFA
    )[indice]

    st.write(
        f"Rezagos HAC utilizados: **{maxlags}**"
    )

    st.dataframe(
        pd.DataFrame({
            "Indicador": [
                "Pendiente",
                "Error estándar HAC",
                "t robusto HAC",
                "p-valor robusto HAC",
                "IC 95% inferior",
                "IC 95% superior"
            ],
            "Resultado": [
                f"{beta:.6f}",
                f"{se:.6f}",
                f"{t:.4f}",
                f"{p:.6g}",
                f"{ic[0]:.4f}",
                f"{ic[1]:.4f}"
            ],
        }),
        use_container_width=True,
        hide_index=True
    )

    if p < ALFA:

        st.success(
            "Con errores estándar robustos, "
            "la pendiente continúa siendo "
            "estadísticamente significativa."
        )

    else:

        st.warning(
            "Con errores estándar robustos, "
            "la pendiente no resulta estadísticamente significativa."
        )


# --------------------------------------------------
# 11. REGRESIÓN LINEAL MÚLTIPLE
# --------------------------------------------------

def mostrar_multiple(
    modelo_simple,
    modelo_multiple,
    datos
):

    st.title(
        "11. Regresión lineal múltiple"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "R²",
        f"{modelo_multiple.rsquared:.4f}"
    )

    c2.metric(
        "R² ajustado",
        f"{modelo_multiple.rsquared_adj:.4f}"
    )

    c3.metric(
        "AIC",
        f"{modelo_multiple.aic:.2f}"
    )

    c4.metric(
        "p-valor F",
        f"{modelo_multiple.f_pvalue:.3g}"
    )

    st.subheader(
        "Ecuación y coeficientes"
    )

    st.write(
        "**Variable dependiente:** Radiacion_Global"
    )

    st.dataframe(
        pd.DataFrame({
            "Coeficiente": modelo_multiple.params,
            "Error estándar": modelo_multiple.bse,
            "t": modelo_multiple.tvalues,
            "p-valor": modelo_multiple.pvalues,
        }).round(5),
        use_container_width=True
    )

    mostrar_resumen_modelo(
        modelo_multiple
    )

    st.subheader(
        "Diagnóstico gráfico"
    )

    grafico_residuos(
        modelo_multiple,
        "Regresión lineal múltiple"
    )


# --------------------------------------------------
# 12. DIAGNÓSTICO Y COMPARACIÓN
# --------------------------------------------------

def mostrar_diagnostico_multiple(
    modelo_simple,
    modelo_multiple,
    datos
):

    st.title(
        "12. Diagnóstico del modelo múltiple y comparación"
    )

    mostrar_pruebas_residuos(
        modelo_multiple,
        "Regresión lineal múltiple"
    )

    st.subheader(
        "Factor de inflación de la varianza (VIF)"
    )

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

    vif_df = pd.DataFrame(
        vif,
        columns=["Variable", "VIF"]
    )

    st.dataframe(
        vif_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Comparación de modelos"
    )

    comparacion = (
        modelo_multiple.compare_f_test(
            modelo_simple
        )
    )

    resumen = pd.DataFrame([
        [
            "R²",
            modelo_simple.rsquared,
            modelo_multiple.rsquared
        ],
        [
            "AIC",
            modelo_simple.aic,
            modelo_multiple.aic
        ],
        [
            "BIC",
            modelo_simple.bic,
            modelo_multiple.bic
        ],
    ], columns=[
        "Indicador",
        "Modelo simple",
        "Modelo múltiple"
    ])

    st.dataframe(
        resumen.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"**Test F parcial (múltiple vs. simple):** "
        f"F = {comparacion[0]:.4f} · "
        f"p-valor = {comparacion[1]:.6g}"
    )

    if comparacion[1] < ALFA:

        st.success(
            "El modelo múltiple mejora significativamente "
            "el ajuste respecto del modelo simple."
        )

    else:

        st.info(
            "No se observa una mejora estadísticamente "
            "significativa del modelo múltiple "
            "respecto del simple."
        )


# --------------------------------------------------
# 13. CONCLUSIONES GENERALES
# --------------------------------------------------

def mostrar_conclusiones(
    datos,
    modelo_simple,
    modelo_multiple
):

    st.title(
        "13. Conclusiones generales"
    )

    mejora = (
        modelo_multiple.rsquared
        - modelo_simple.rsquared
    ) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Observaciones",
        len(datos)
    )

    c2.metric(
        "R² modelo simple",
        f"{modelo_simple.rsquared:.4f}"
    )

    c3.metric(
        "R² modelo múltiple",
        f"{modelo_multiple.rsquared:.4f}"
    )

    st.write(
        f"El modelo simple explica el "
        f"**{modelo_simple.rsquared * 100:.2f}%** "
        f"de la variabilidad de la radiación global."
    )

    st.write(
        f"El modelo múltiple explica el "
        f"**{modelo_multiple.rsquared * 100:.2f}%** "
        f"de la variabilidad."
    )

    st.write(
        f"La ganancia de explicación del modelo múltiple "
        f"es de **{mejora:.2f} puntos porcentuales**."
    )

    st.caption(
        "La asociación estadística no implica necesariamente "
        "causalidad. Las conclusiones inferenciales deben "
        "interpretarse considerando los diagnósticos de residuos."
    )


# ==================================================
# INTERFAZ PRINCIPAL DE LA APLICACIÓN
# ==================================================

# LOGO: guardá la imagen como logo.png junto a app.py.

logo_path = Path(__file__).with_name("logo.png")

col_logo, col_encabezado = st.columns(
    [1, 5],
    vertical_alignment="center"
)

with col_logo:

    if logo_path.exists():

        st.image(
            str(logo_path),
            width=130
        )

    else:

        st.markdown("# 📊")


with col_encabezado:

    st.title(
        "Trabajo práctico integrador"
    )

    st.subheader(
        "Métodos Estadísticos para Inteligencia Artificial"
    )

    st.caption(
        "Estación meteorológica Castelar · "
        "Buenos Aires · Abadi y Lezcano"
    )


st.markdown(
    "Aplicación interactiva basada en el trabajo práctico de "
    "Abadi y Lezcano. Cargá la base meteorológica en formato "
    "CSV para recorrer los análisis."
)


# --------------------------------------------------
# PANEL LATERAL
# --------------------------------------------------

with st.sidebar:

    st.header("Carga de datos")

    archivo = st.file_uploader(
        "Subir archivo CSV",
        type=["csv"]
    )

    st.caption(
        "El archivo debe contener Fecha y las cuatro "
        "variables del trabajo."
    )

    st.divider()

    st.caption(
        "Nivel de significación utilizado: α = 0,05"
    )


# --------------------------------------------------
# VALIDACIÓN DE LA BASE
# --------------------------------------------------

if archivo is None:

    st.info(
        "Para comenzar, cargá el archivo "
        "`01_Buenos_Aires_Castelar.csv` "
        "desde el panel lateral."
    )

    st.markdown(
        "**Columnas requeridas:** `Fecha`, "
        "`Temperatura_Abrigo_150cm`, "
        "`Humedad_Media`, `Presion_Media` "
        "y `Radiacion_Global`."
    )

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
        "Se necesitan al menos 12 observaciones "
        "para ejecutar todos los análisis, "
        "incluido Ljung-Box con rezago 10."
    )

    st.stop()


# --------------------------------------------------
# AJUSTE DE LOS MODELOS
# --------------------------------------------------

modelo_simple, modelo_multiple = ajustar_modelos(
    datos
)


# --------------------------------------------------
# NAVEGACIÓN POR LAS 13 SECCIONES
# --------------------------------------------------

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
            "13. Conclusiones generales",
        ]
    )


# --------------------------------------------------
# EJECUCIÓN DE LA SECCIÓN SELECCIONADA
# --------------------------------------------------

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
        modelo_multiple,
        datos
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
