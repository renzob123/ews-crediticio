# EWS crediticio

Sistema de alerta temprana (Early Warning System) de deterioro crediticio para
clientes persona física de Santander Argentina (entidad 72), construido a partir
de la Central de Deudores del BCRA (archivo 24DSF, 24 meses de historia).

## Targets

- `target_a`: caída a situación >= 2 en cualquier entidad en los próximos 3 meses
  (entrada: situación 1 en todo el sistema).
- `target_c`: caída a situación >= 3 en cualquier entidad en los próximos 12 meses
  (entrada: situación <= 2 en todo el sistema).

## Pipeline

| Script | Qué hace | Genera |
|---|---|---|
| `01_descargar_macro_bcra.py` | Descarga series macro del BCRA | `macro_bcra.csv` |
| `02_construir_dataset.py` | Dos pasadas sobre el 24DSF: features y targets por CUIT × período | `dataset_ews.parquet` |
| `03_split.py` | Merge con macro y split train/test a nivel CUIT | `dataset_train.parquet`, `dataset_test.parquet` |

`02_extraer_santander.py` es una versión anterior del paso 2 que quedó solo de referencia.

## Datos

Los datos **no están en el repo**: son archivos pesados (~22 GB) y contienen
CUITs de personas. Para correr el pipeline:

1. Descargar el 24DSF desde la
   [Central de Deudores del BCRA](https://www.bcra.gob.ar/BCRAyVos/Situacion_Crediticia.asp)
   y descomprimirlo en esta carpeta como `24DSF.txt`.
2. Ajustar `ARCHIVO_24DSF` y `BASE_PERIODO` en `02_construir_dataset.py` si el
   archivo corresponde a otro mes.
3. Correr los scripts en orden.

## Requisitos

Python 3 con `pyarrow`, `pandas` y `numpy`:

```
pip install pyarrow pandas numpy
```
