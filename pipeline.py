"""Pipeline BigQuery de viajes TLC, primer trimestre de 2022.

Instalar: pip install google-cloud-bigquery pandas pyarrow db-dtypes
Autenticar en Colab con google.colab.auth.authenticate_user(), o en un equipo
con gcloud auth application-default login. Ejecutar:
python pipeline.py --proyecto TU_PROYECTO
Reemplaza las tablas y el modelo de movilidad_urbana, no el dataset completo.
Las fechas y controles corresponden únicamente al trimestre indicado.
"""

import argparse
import json
import re
import time
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
ETAPAS = [
    ('01_ingesta', r'''
CREATE SCHEMA IF NOT EXISTS `{project}.movilidad_urbana`
OPTIONS(location='US', default_table_expiration_days=60);

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.bronze_viajes`
PARTITION BY RANGE_BUCKET(data_file_month, GENERATE_ARRAY(1, 13, 1))
CLUSTER BY pickup_location_id
OPTIONS(description='Viajes originales TLC primer trimestre 2022; sin correcciones') AS
SELECT *
FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`
WHERE data_file_month BETWEEN 1 AND 3;
'''),
    ('02_calidad', r'''
CREATE OR REPLACE TABLE `{project}.movilidad_urbana.silver_viajes`
PARTITION BY RANGE_BUCKET(zona_id, GENERATE_ARRAY(0, 272, 16))
CLUSTER BY pickup_datetime
OPTIONS(description='Viajes con fecha del trimestre y zona TLC valida; no se deduplican por falta de identificador unico') AS
SELECT pickup_datetime, dropoff_datetime,
  SAFE_CAST(pickup_location_id AS INT64) AS zona_id,
  SAFE_CAST(dropoff_location_id AS INT64) AS destino_id,
  CAST(trip_distance AS FLOAT64) AS trip_distance,
  CAST(total_amount AS FLOAT64) AS total_amount,
  passenger_count
FROM `{project}.movilidad_urbana.bronze_viajes`
WHERE pickup_datetime >= TIMESTAMP('2022-01-01')
  AND pickup_datetime < TIMESTAMP('2022-04-01')
  AND SAFE_CAST(pickup_location_id AS INT64) BETWEEN 1 AND 263;

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.control_calidad` AS
SELECT COUNT(*) AS filas_originales,
  COUNTIF(pickup_datetime IS NULL) AS fecha_nula,
  COUNTIF(pickup_datetime < TIMESTAMP('2022-01-01') OR pickup_datetime >= TIMESTAMP('2022-04-01')) AS fuera_periodo,
  COUNTIF(SAFE_CAST(pickup_location_id AS INT64) IS NULL OR SAFE_CAST(pickup_location_id AS INT64) NOT BETWEEN 1 AND 263) AS zona_invalida,
  COUNTIF(trip_distance < 0) AS distancia_negativa,
  COUNTIF(total_amount < 0) AS importe_negativo,
  COUNTIF(dropoff_datetime < pickup_datetime) AS fechas_invertidas,
  (SELECT COUNT(*) FROM `{project}.movilidad_urbana.silver_viajes`) AS filas_limpias
FROM `{project}.movilidad_urbana.bronze_viajes`;

ASSERT (SELECT filas_limpias > 1000000 AND filas_limpias <= filas_originales FROM `{project}.movilidad_urbana.control_calidad`)
AS 'El volumen limpio debe superar un millon y no exceder el original';
'''),
    ('03_agregacion', r'''
CREATE OR REPLACE TABLE `{project}.movilidad_urbana.gold_demanda`
PARTITION BY RANGE_BUCKET(zona_id, GENERATE_ARRAY(0, 272, 16))
CLUSTER BY hora AS
WITH conteos AS (
  SELECT TIMESTAMP_TRUNC(pickup_datetime, HOUR) AS hora,
    zona_id, COUNT(*) AS viajes
  FROM `{project}.movilidad_urbana.silver_viajes`
  WHERE DATE(pickup_datetime) BETWEEN '2022-01-01' AND '2022-03-31'
  GROUP BY 1, 2
), grilla AS (
  SELECT hora, zona_id
  FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(TIMESTAMP('2022-01-01'), TIMESTAMP('2022-03-31 23:00:00'), INTERVAL 1 HOUR)) AS hora
  CROSS JOIN UNNEST(GENERATE_ARRAY(1, 263)) AS zona_id
)
SELECT grilla.hora, grilla.zona_id, COALESCE(conteos.viajes, 0) AS viajes
FROM grilla LEFT JOIN conteos USING(hora, zona_id);

ASSERT (SELECT COUNT(*) = 568080 FROM `{project}.movilidad_urbana.gold_demanda`)
AS 'La grilla debe contener 90 dias por 24 horas por 263 zonas';
ASSERT (SELECT SUM(viajes) FROM `{project}.movilidad_urbana.gold_demanda`) =
  (SELECT COUNT(*) FROM `{project}.movilidad_urbana.silver_viajes`)
AS 'La agregacion debe conservar el total de viajes';
'''),
    ('04_variables', r'''
CREATE OR REPLACE TABLE `{project}.movilidad_urbana.features_demanda`
PARTITION BY RANGE_BUCKET(zona_id, GENERATE_ARRAY(0, 272, 16))
CLUSTER BY hora AS
WITH rezagos AS (
 SELECT *,
   LAG(viajes, 1) OVER ventana_zona AS rezago_1,
   LAG(viajes, 24) OVER ventana_zona AS rezago_24,
   LAG(viajes, 168) OVER ventana_zona AS rezago_168,
   AVG(viajes) OVER (PARTITION BY zona_id ORDER BY hora ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS media_24h
 FROM `{project}.movilidad_urbana.gold_demanda`
 WINDOW ventana_zona AS (PARTITION BY zona_id ORDER BY hora)
)
SELECT *,
  SIN(2*ACOS(-1)*EXTRACT(HOUR FROM hora)/24) AS seno_hora,
  COS(2*ACOS(-1)*EXTRACT(HOUR FROM hora)/24) AS coseno_hora,
  CAST(EXTRACT(DAYOFWEEK FROM hora) AS STRING) AS dia_semana,
  CASE WHEN hora < TIMESTAMP('2022-02-16') THEN 'entrenamiento'
       WHEN hora < TIMESTAMP('2022-03-01') THEN 'validacion'
       ELSE 'prueba' END AS conjunto
FROM rezagos WHERE rezago_168 IS NOT NULL;
'''),
    ('05_modelo', r'''
CREATE OR REPLACE MODEL `{project}.movilidad_urbana.modelo_demanda`
OPTIONS(model_type='LINEAR_REG', input_label_cols=['viajes'],
 data_split_method='CUSTOM', data_split_col='es_validacion',
 l2_reg=1.0, max_iterations=30, early_stop=TRUE) AS
SELECT viajes, CAST(zona_id AS STRING) AS zona,
  rezago_1, rezago_24, rezago_168, media_24h, seno_hora, coseno_hora, dia_semana,
  conjunto='validacion' AS es_validacion
FROM `{project}.movilidad_urbana.features_demanda`
WHERE conjunto IN ('entrenamiento', 'validacion');

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.predicciones`
PARTITION BY RANGE_BUCKET(zona_id, GENERATE_ARRAY(0, 272, 16))
CLUSTER BY hora AS
SELECT hora, zona_id, viajes, rezago_1, rezago_24, rezago_168,
  GREATEST(0.0, predicted_viajes) AS predicted_viajes
FROM ML.PREDICT(MODEL `{project}.movilidad_urbana.modelo_demanda`,
 (SELECT *, CAST(zona_id AS STRING) AS zona
  FROM `{project}.movilidad_urbana.features_demanda` WHERE conjunto='prueba'));

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.metricas` AS
WITH pronosticos AS (
 SELECT 'Regresion lineal BQML' AS modelo, viajes AS real, predicted_viajes AS estimado FROM `{project}.movilidad_urbana.predicciones`
 UNION ALL
 SELECT 'Persistencia ultima hora', viajes, rezago_1 FROM `{project}.movilidad_urbana.predicciones`
 UNION ALL
 SELECT 'Estacional dia anterior', viajes, rezago_24 FROM `{project}.movilidad_urbana.predicciones`
 UNION ALL
 SELECT 'Estacional semana anterior', viajes, rezago_168 FROM `{project}.movilidad_urbana.predicciones`
)
SELECT modelo, COUNT(*) AS filas_evaluadas,
 AVG(ABS(real-estimado)) AS mae,
 SQRT(AVG(POW(real-estimado,2))) AS rmse,
 SAFE_DIVIDE(SUM(ABS(real-estimado)),SUM(real))*100 AS wape,
 1-SAFE_DIVIDE(SUM(POW(real-estimado,2)),SUM(POW(real,2))-POW(SUM(real),2)/COUNT(*)) AS r2
FROM pronosticos GROUP BY modelo;
'''),
    ('06_eda', r'''
CREATE OR REPLACE TABLE `{project}.movilidad_urbana.eda_diario` AS
SELECT DATE(hora) AS fecha, SUM(viajes) AS viajes
FROM `{project}.movilidad_urbana.gold_demanda` GROUP BY 1;

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.eda_horario` AS
SELECT EXTRACT(DAYOFWEEK FROM hora) AS dia_semana,
 EXTRACT(HOUR FROM hora) AS hora_del_dia, AVG(viajes) AS mean_trips
FROM `{project}.movilidad_urbana.gold_demanda` GROUP BY 1,2;

CREATE OR REPLACE TABLE `{project}.movilidad_urbana.eda_zonas` AS
SELECT zona_id, SUM(viajes) AS viajes
FROM `{project}.movilidad_urbana.gold_demanda` GROUP BY 1;
'''),
    ('08_validaciones', r'''
ASSERT (SELECT COUNT(*) = 568080 FROM `{project}.movilidad_urbana.gold_demanda`)
AS 'Cantidad incorrecta de combinaciones zona hora';

ASSERT (SELECT COUNT(*) FROM `{project}.movilidad_urbana.silver_viajes`) =
(SELECT SUM(viajes) FROM `{project}.movilidad_urbana.gold_demanda`)
AS 'La agregacion no conserva el total';

ASSERT (SELECT COUNT(*)=195672 AND COUNTIF(predicted_viajes IS NULL OR predicted_viajes < 0)=0
FROM `{project}.movilidad_urbana.predicciones`)
AS 'Predicciones de marzo incompletas o no validas';

ASSERT (SELECT COUNT(*)=4 AND COUNT(DISTINCT filas_evaluadas)=1
FROM `{project}.movilidad_urbana.metricas`)
AS 'Los cuatro modelos deben compartir el conjunto de prueba';

ASSERT (SELECT COUNT(*)=0 FROM `{project}.movilidad_urbana.features_demanda`
WHERE rezago_1 IS NULL OR rezago_24 IS NULL OR rezago_168 IS NULL OR media_24h IS NULL)
AS 'Las variables predictoras contienen valores faltantes';

SELECT 'Cinco controles de integridad superados' AS resultado;
''')
]


def ejecutar_consulta(cliente, consulta, configuracion):
    """Ejecuta SQL con límite de bytes y cuatro intentos ante fallos transitorios."""
    from google.api_core.exceptions import InternalServerError, ServiceUnavailable, TooManyRequests
    from google.cloud import bigquery

    opciones = bigquery.QueryJobConfig(
        maximum_bytes_billed=configuracion["limite_bytes_consulta"],
        labels={"trabajo": "ev2-bigdata"},
    )
    for intento in range(4):
        try:
            trabajo = cliente.query(consulta, job_config=opciones, location=configuracion["ubicacion"])
            filas = list(trabajo.result())
            return trabajo, filas
        except (InternalServerError, ServiceUnavailable, TooManyRequests):
            if intento == 3:
                raise
            time.sleep(2 ** intento)


def reconstruir(proyecto=None):
    """Reconstruye capas, entrena, valida y exporta los resultados del propio proyecto."""
    import pandas as pd
    from google.cloud import bigquery

    configuracion = json.loads((RAIZ / "configuracion.json").read_text(encoding="utf-8"))
    proyecto = proyecto or configuracion["proyecto_id"]
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", proyecto):
        raise ValueError("Identificador de proyecto no válido")
    cliente = bigquery.Client(project=proyecto)
    ejecuciones = []
    for nombre, consulta in ETAPAS:
        print("Ejecutando", nombre, flush=True)
        trabajo, _ = ejecutar_consulta(cliente, consulta.format(project=proyecto), configuracion)
        ejecuciones.append({"etapa": nombre, "job_id": trabajo.job_id,
                            "bytes_procesados": trabajo.total_bytes_processed,
                            "inicio": str(trabajo.started), "fin": str(trabajo.ended)})
    consultas = {
        "calidad": "SELECT * FROM `{project}.movilidad_urbana.control_calidad`",
        "metricas": "SELECT * FROM `{project}.movilidad_urbana.metricas` ORDER BY rmse",
        "diario": "SELECT * FROM `{project}.movilidad_urbana.eda_diario` ORDER BY fecha",
        "horario": "SELECT * FROM `{project}.movilidad_urbana.eda_horario`",
        "zonas": "SELECT * FROM `{project}.movilidad_urbana.eda_zonas` ORDER BY viajes DESC",
        "distribucion": "SELECT conjunto, COUNT(*) filas, SUM(viajes) viajes, MIN(hora) primera_hora, MAX(hora) ultima_hora FROM `{project}.movilidad_urbana.features_demanda` GROUP BY conjunto",
        "muestra_predicciones": "SELECT * FROM `{project}.movilidad_urbana.predicciones` WHERE zona_id IN (132,161,237) AND hora < TIMESTAMP('2022-03-08') ORDER BY zona_id,hora",
        "muestra_variables": "SELECT * FROM `{project}.movilidad_urbana.features_demanda` WHERE zona_id IN (132,161,237) AND hora >= TIMESTAMP('2022-02-28') AND hora < TIMESTAMP('2022-03-02') ORDER BY zona_id,hora",
        "coeficientes": "SELECT * FROM ML.WEIGHTS(MODEL `{project}.movilidad_urbana.modelo_demanda`, STRUCT(TRUE AS standardize))",
        "entrenamiento": "SELECT * FROM ML.TRAINING_INFO(MODEL `{project}.movilidad_urbana.modelo_demanda`)",
        "errores_zona": "SELECT zona_id,COUNT(*) filas,SUM(viajes) viajes,AVG(ABS(viajes-predicted_viajes)) mae,SQRT(AVG(POW(viajes-predicted_viajes,2))) rmse,SAFE_DIVIDE(SUM(ABS(viajes-predicted_viajes)),SUM(viajes))*100 wape FROM `{project}.movilidad_urbana.predicciones` GROUP BY zona_id",
        "almacenamiento": "SELECT table_id tabla,row_count filas,size_bytes bytes_tabla FROM `{project}.movilidad_urbana.__TABLES__`",
    }
    resultados = {"procedencia": {"proyecto": proyecto, "dataset": "movilidad_urbana",
                                 "periodo": "2022-01-01 / 2022-03-31"}, "ejecuciones": ejecuciones}
    for nombre, consulta in consultas.items():
        _, filas = ejecutar_consulta(cliente, consulta.format(project=proyecto), configuracion)
        resultados[nombre] = [dict(fila) for fila in filas]
    _, series = ejecutar_consulta(cliente, "SELECT zona_id, ARRAY_AGG(viajes ORDER BY hora) conteos FROM `" + proyecto + ".movilidad_urbana.gold_demanda` GROUP BY zona_id", configuracion)
    horas = pd.date_range("2022-01-01", periods=2160, freq="h", tz="UTC")
    demanda = pd.concat([pd.DataFrame({"hora": horas, "zona_id": fila["zona_id"], "viajes": fila["conteos"]}) for fila in series], ignore_index=True)
    assert len(demanda) == 568080 and demanda.viajes.sum() == resultados["calidad"][0]["filas_limpias"]
    demanda.to_parquet(RAIZ / "demanda_horaria.parquet", index=False, compression="snappy")
    (RAIZ / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Reconstrucción y cinco controles cloud completados.")


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--proyecto", help="Proyecto propio con BigQuery habilitado")
    reconstruir(argumentos.parse_args().proyecto)
