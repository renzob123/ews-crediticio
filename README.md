# EWS crediticio

Sistema de alerta temprana (Early Warning System) de deterioro crediticio para
clientes persona física de Santander Argentina (entidad 72), construido a partir
de la Central de Deudores del BCRA (archivos 24DSF de 24 meses cada uno; se pueden
combinar varios para tener más historia).

## Targets

| Target | Evento | Horizonte | Entrada (en t) | Rol |
|---|---|---|---|---|
| `target_a` | 31+ días: situación >= 2 | 3 meses | situación 1 en todo el sistema | Alerta muy temprana |
| `target_b` | ~60+ días: situación >= 2 dos meses seguidos | 6 meses | situación 1 en todo el sistema | Principal |
| `target_c` | 90+ días: situación >= 3 | 12 meses | situación <= 2 en todo el sistema | Horizonte regulatorio |

El evento se mide en cualquier entidad del sistema. `target_b` es una
aproximación: el 24DSF no trae días de atraso y la situación 2 cubre de 31 a 90
días, así que "dos meses seguidos en situación >= 2" se usa como proxy de 60+.

## Pipeline

| Script | Qué hace | Genera |
|---|---|---|
| `01_descargar_macro_bcra.py` | Descarga series macro del BCRA | `macro_bcra.csv` |
| `02_construir_dataset.py` | Dos pasadas sobre el 24DSF: features y targets por CUIT × período | `dataset_ews.parquet` |
| `03_split.py` | Merge con macro y split train/test a nivel CUIT | `dataset_train.parquet`, `dataset_test.parquet` |

`02_extraer_santander.py` es una versión anterior del paso 2 que quedó solo de referencia.

## Validación

- **Por persona:** train y test tienen CUITs distintos (80/20, estratificado por
  "alguna vez `target_c` = 1"). Cada CUIT entra con su historia completa.
- **Fuera de tiempo:** las columnas `oot_a`, `oot_b` y `oot_c` marcan cada fila como
  `train`, `hueco` o `test` según su período. Test son los últimos 3 meses
  válidos del target; train, los períodos cuya ventana del target termina antes
  del test. Las filas del hueco no se usan, porque su respuesta depende de meses
  del test.
- **Historia mínima:** las features miran como mucho 6 meses atrás y solo hay
  filas para períodos con esos 6 meses dentro de los archivos, para que signifiquen
  lo mismo que en producción.

Con 12 meses de horizonte, `target_c` necesita mucha historia para tener train
fuera de tiempo (el hueco ocupa 12 meses); con los archivos actuales no alcanza,
así que se valida solo por persona.

## Datos

Los datos **no están en el repo**: son archivos pesados (~22 GB) y contienen
CUITs de personas. Para correr el pipeline:

1. Descargar el 24DSF desde la
   [Central de Deudores del BCRA](https://www.bcra.gob.ar/BCRAyVos/Situacion_Crediticia.asp)
   y descomprimirlo en esta carpeta.
2. Listar los archivos y el mes más reciente de cada uno en `ARCHIVOS_24DSF`
   (`02_construir_dataset.py`). Los meses tienen que ser contiguos: cada archivo
   tiene que empezar antes de que termine el siguiente más nuevo.
3. Correr los scripts en orden.

## Requisitos

Python 3 con `pyarrow`, `pandas` y `numpy`:

```
pip install pyarrow pandas numpy
```
