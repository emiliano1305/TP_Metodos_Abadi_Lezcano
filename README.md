# Trabajo práctico integrador — Abadi y Lezcano

Aplicación interactiva en Streamlit basada en el notebook `TPFinal-Estadistica.Abadi-Lezcano.ipynb`.

## Ejecutar localmente

1. Clonar o descargar este repositorio.
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar:

   ```bash
   streamlit run app.py
   ```

4. En la barra lateral, cargar el archivo CSV de la estación meteorológica.

## Publicar en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt` y `README.md`.
3. Entrar en https://share.streamlit.io/
4. Conectar GitHub y seleccionar el repositorio.
5. Elegir `app.py` como archivo principal y desplegar.

## Formato de la base CSV

El archivo debe incluir estos encabezados:

- `Fecha`
- `Temperatura_Abrigo_150cm`
- `Humedad_Media`
- `Presion_Media`
- `Radiacion_Global`

Se admite separador coma o punto y coma. La aplicación valida la presencia de las columnas y los valores antes de ejecutar los análisis.

## Análisis incluidos

1. Análisis estructural de los datos.
2. Estadística descriptiva, valores atípicos y normalidad.
3. Histogramas y diagramas de caja.
4. Series temporales.
5. Matriz de correlaciones y regresión gráfica.
6. Conclusión descriptiva.
7. Regresión lineal simple.
8. Diagnóstico de residuos.
9. Test bilateral de hipótesis sobre la pendiente.
10. Errores robustos Newey-West.
11. Regresión lineal múltiple.
12. Diagnóstico, VIF y comparación de modelos mediante AIC y test F parcial.
13. Conclusiones generales.

**Importante:** el CSV con los datos no está incluido en este paquete. Debe cargarse desde la aplicación.
