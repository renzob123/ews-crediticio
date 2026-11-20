# Decisiones, limitaciones y complicaciones

Registro vivo de la metodología del sistema de alerta temprana (EWS) crediticio.
Se actualiza cada vez que se toma una decisión o aparece una limitación.
Sirve de insumo para los capítulos de metodología y limitaciones de la tesis.

Última actualización: 2026-11-17

---

## 1. Caso de uso

Cuando el BCRA publica la Central de Deudores del mes *t* (aprox. a fines de *t+1*),
se puntúa a todos los clientes en situación 1 en *t* y se marca a quienes probablemente
caigan en mora, para que el banco actúe. El banco ya conoce los atrasos con su propia
cartera por procesos internos; el valor del modelo es la **visión de deuda en todo el
sistema financiero** que aporta el BCRA.

---

## 2. Decisiones tomadas

### 2.1 Datos y construcción del panel
| Decisión | Detalle | Motivo |
|---|---|---|
| Fuente | Central de Deudores BCRA (24DSF), dos archivos combinados (31 meses, filas desde 202407) | Ampliar la historia disponible para validación fuera de tiempo |
| Unidad | Una fila por CUIT por período *t* | Panel mensual |
| Ventanas de features | Fijas de 3 y 6 meses (no 12 por ahora) | Evitar fuga de información señalada por el tutor; ventanas comparables entre filas |
| Historia mínima | 6 meses previos | Que todas las ventanas estén completas; las filas censuradas a izquierda se descartan |
| Macro | Variables BCRA unidas por período en 03 | Contexto macroeconómico |

### 2.2 Targets
Población: CUITs en situación 1 en *t*.

| Target | Definición | Rol |
|---|---|---|
| A | Mora 31+ días en un horizonte de 3 meses | Secundario |
| B | ~60+ días en 6 meses: situación ≥ 2 dos meses consecutivos | **Principal** |
| C | Mora 90+ días en 12 meses | Secundario, solo validación por CUIT |

### 2.3 Duda 1: rezago de publicación (decidida 2026-11-15)
- **Ventana del target:** de *t+2* a *t+H+1* (A: t+2..t+4; B: t+2..t+7, con los dos meses consecutivos dentro de la ventana; C: t+2..t+13). Las features se toman en *t*.
- **Motivo:** una situación 2 al cierre de *t+1* implica una cuota vencida en *t*, antes del momento de scoring. No es accionable para el banco.
- Las caídas en *t+1* que persisten en *t+2* o después cuentan como positivas. No se excluyen filas.
- **Sensibilidad:** se entrena también con la ventana *t+1..t+H* y se reporta la caída de la métrica.
- **Implementación (02):** el inicio de la ventana es un parámetro (`INICIO_VENTANA = 2`). Los targets de sensibilidad (`target_a_t1`, `target_b_t1`, `target_c_t1`) se generan en la misma corrida. Los parámetros se guardan en los metadatos del parquet y 03 los lee para la validación OOT.

### 2.4 Duda 2: inflación (decidida 2026-11-15)
- **Deflactar con CER** (se eligió CER sobre IPC INDEC; se agrega en 01): monto_sant_t, monto_sant_t3 (con el CER del propio mes), monto_otras_t, sus logs, var_monto_3m_sant y las variables macro nominales.
- **Features libres de inflación:** share_sant = monto_sant / (monto_sant + monto_otras) en *t* y *t3* (requiere crear monto_otras_t3). También deciles por período en *t* y *t3* para monto_sant y monto_otras (en otras: decil 0 = sin deuda y deciles 1-10 entre los que tienen deuda > 0). Se calculan en 03, con una primera pasada para los cortes de cada período.
- **Sensibilidad OOT:** se comparan montos nominales, montos reales y solo ratios/deciles.
- **Implementación (03_split.py):** los montos reales se expresan en pesos del último período del dataset. Los cortes de deciles de cada período se calculan con las filas en sit 1 en todo el sistema (la población de A y B), para que el decil signifique lo mismo en todos los períodos: desde 202508 solo hay filas válidas para A/B, y si se incluyeran los deudores en sit 2 de C en los períodos anteriores, la población cambiaría. El decil 0 (sin deuda) se aplica a los cuatro montos, porque monto_sant_t3 también puede ser 0.

### 2.5 Duda 3: target C (decidida 2026-11-15)
- C queda como **target secundario**, validado solo con el split por CUIT, sin validación fuera de tiempo.
- Con la ventana de la duda 1, C solo puede etiquetarse para *t* = 202407..202507 (13 meses).
- No se consigue un 24DSF más antiguo y no se acorta el horizonte de C.
- Las conclusiones principales salen de B (y de A).

### 2.6 Duda 4: filas correlacionadas por CUIT (decidida 2026-11-17)
Un mismo CUIT aparece en muchos períodos, con filas muy parecidas, así que esas filas no son independientes. Se decidió:
- **Entrenar con todas las filas**, sin submuestrear por CUIT: cada mes es una situación distinta del deudor.
- **Ajustar hiperparámetros con GroupKFold por CUIT**: todas las filas de un CUIT caen en el mismo fold, para que la validación evalúe siempre deudores no vistos.
- **Métricas principales** (AUC, PR-AUC, captura en el top X%) calculadas **por mes de scoring *t*** y reportadas como media ± desvío entre meses. Esto replica el uso real; dentro de un mes hay una fila por CUIT.
- **Intervalos de confianza** con bootstrap por clusters, remuestreando CUITs con todas sus filas.
- Como métrica de robustez opcional, se repite la evaluación con una fila al azar por CUIT.
- Solo afecta al modelado, no a los scripts 02 y 03.

### 2.7 Duda 5: variables macro (decidida 2026-11-17, rezagos en revisión)
- Se entrenan modelos **con y sin variables macro** y se comparan. No está claro que aporten: son constantes dentro de cada mes, así que no pueden ordenar deudores dentro de un mismo mes de scoring salvo por interacciones con las variables individuales. Además, con pocos meses de entrenamiento hay riesgo de sobreajustar al período.
- En la versión con macro, cada variable debe usar **el último dato publicado al momento del scoring** (fines de *t+1*) para evitar fuga de información. La tabla de rezagos está en revisión (ver 4.4).
- **Se eliminan las 7 columnas macro rotas** (decidido 2026-11-18): resultado_neto_mensual, intereses_ganados, intereses_pagados, margen_intermediacion, roa_proxy, tasa_hipotecarios_privado_pct y n_entidades_financieras.
- **Regla de rezago** (decidida 2026-11-18): la macro describe el mes *t*, el mismo momento en que se informó al cliente. Se usa el valor de *t* si ya estaba publicado al momento del scoring (fines de *t+1*); si no, el de *t-1*. La primera descarga (27/10/2026) sugería *t-1* para la mora del sistema y BADLAR. La segunda (18/11/2026) mostró que la mora y BADLAR de 202610 ya estaban publicadas a mediados de noviembre, y que 202609 era un **hueco** de la serie, no un atraso. **Rezagos finales:** todas las variables en *t*, salvo var_credito_real_3m_pct en *t-1* (el crédito privado de 202610 todavía no estaba publicado el 18/11). Los meses faltantes (202601 y 202609 en la mora y BADLAR) quedan vacíos.
- **Reducción a 10 variables macro** (decidida 2026-11-18). Motivos: (a) redundancia: sistema y privado correlacionan en 1,000, y todos los niveles en pesos correlacionan entre 0,97 y 0,999 entre sí y con el tipo de cambio, porque reflejan inflación y no crédito ni liquidez; (b) hay solo ~25 valores distintos por variable (uno por mes), con alto riesgo de sobreajustar al período.

  | Variable | Mes |
  |---|---|
  | mora_3plus_privado_pct | *t* |
  | sit2_privado_pct | *t* |
  | var_mora3plus_3m (sobre el sector privado) | *t* |
  | var_sit2_3m (sobre el sector privado) | *t* |
  | tasa_consumo_corto_pct | *t* |
  | badlar_privados_tna_pct | *t* |
  | spread_consumo_badlar_pct | *t* |
  | var_tc_3m_pct (nueva: variación % del tipo de cambio en 3 meses) | *t* |
  | var_credito_real_3m_pct (nueva: crédito privado deflactado por CER) | *t-1* |
  | pct_personales | *t* |

  Se descartan: versiones "sistema", sit3/sit4/sit5, cartera_irregular, variaciones a 1 mes, reservas, base monetaria, depósitos, niveles de crédito y préstamos en pesos, y la composición de la cartera salvo personales.

### 2.8 Duda 6: fuga de información en el preprocesamiento (decidida 2026-11-18)
Regla general: todo lo que se aprende de los datos se ajusta **solo con train**, y en el tuning, solo con los folds de entrenamiento de cada ronda. El test (por CUIT y OOT) se usa **una sola vez**, al final.
- **Deciles por período:** se calculan con todas las filas del período, incluido test. No hay fuga, porque no usan el target y solo usan información de la foto de *t*, que es lo mismo que tiene el banco al momento del scoring.
- **Deflactar con CER:** es una transformación fija y no aprende nada de los datos.
- **Desbalance de clases:** se usa peso de clase (`scale_pos_weight`). Cualquier remuestreo se hace solo en train. Las métricas siempre se calculan con la prevalencia real.
- **Selección de variables, early stopping, calibración y punto de corte de alerta (top X%):** se definen solo con train o validación.
- **Modelos:** XGBoost/LightGBM como modelos principales y **regresión logística como modelo base** (interpretable y estándar en la industria). Para la logística, el escalado y la imputación van dentro de un `Pipeline` de sklearn ajustado solo con train.

### 2.9 Esquemas de validación
- **Split por CUIT** (aprox. 80/20): ningún deudor aparece en train y en test a la vez.
- **Out-of-time (OOT)** para A y B: tramos train / hueco / test, con los últimos 3 meses válidos en test.

---

### 2.10 Dataset resultante (corrida del 2026-11-19, main = 44457eb)
- 35.378.076 filas (CUIT × período, 202407–202605), 2.110.047 CUITs. Split por CUIT: 28.303.504 filas en train y 7.074.572 en test (422.009 CUITs). 72 columnas.

  | Target | Filas válidas | % positivos | OOT train / hueco / test (meses) |
  |---|---|---|---|
  | A (t+2..t+4) | 33,3M | 6,5% | 16 / 3 / 3 |
  | B (t+2..t+7) | 29,0M | 7,5% | 10 / 6 / 3 |
  | C (t+2..t+13) | 20,7M | 13,3% | 0 / 10 / 3 |
  | A_t1 (t+1..t+3) | 34,7M | 5,3% | 18 / 2 / 3 |
  | B_t1 (t+1..t+6) | 30,5M | 6,5% | 12 / 5 / 3 |
  | C_t1 (t+1..t+12) | 22,3M | 12,6% | 0 / 11 / 3 |

- Con la ventana t+2 la tasa de positivos sube (A: de 5,3% a 6,5%; B: de 6,5% a 7,5%): casi nadie cae a sit 2 el mes siguiente de estar en sit 1, así que correr la ventana captura más caídas. A cambio, cada target pierde un mes de períodos válidos y B pierde 2 meses de train OOT.
- Los targets `_t1`, las filas y el split coinciden exactamente con la corrida anterior (2026-11-06).

### 2.11 Implementación del modelado (04_modelado.py, 2026-11-19)
Cada corrida es una combinación de modelo × target × validación × montos × con/sin macro. Elecciones de implementación (se pueden cambiar con parámetros):
- **Modelo principal: XGBoost; LightGBM como sensibilidad** (decisión 2026-11-20). Se eligió XGBoost como principal por ser el más difundido en la literatura de riesgo crediticio y más fácil de justificar. LightGBM se corre como sensibilidad para mostrar que el resultado no depende de la implementación del boosting. Los dos usan espacios de búsqueda equivalentes y el mismo esquema de tuning. XGBoost usa `tree_method="hist"` y `QuantileDMatrix` (datos discretizados, aprox. 1 byte por valor) para que el entrenamiento con todo train entre en memoria. En la prueba con 2% de los CUITs dieron prácticamente lo mismo (AUC 0,797 vs 0,799; PR-AUC 0,291 vs 0,295).
- **Búsqueda de hiperparámetros:** búsqueda aleatoria de 10 configuraciones, GroupKFold(3) por CUIT, early stopping (50 árboles sin mejora) en el fold de validación y PR-AUC como métrica de selección, porque con 7% de positivos es más exigente que el AUC. Por costo, se hace sobre una muestra del 20% de los CUITs de train. El modelo final se entrena con todo train, usando el promedio de árboles de los folds.
- **Regresión logística:** C = 1 sin tuning (con millones de filas la regularización casi no cambia nada). Montos en log; recorte de extremos en los percentiles 0,5 y 99,5; imputación con mediana más indicador de faltante; escalado y one-hot de sexo y rango de edad. Todo dentro de un Pipeline ajustado solo con train. **Se entrena con el 25% de los CUITs de train**, porque con todas las filas no entra en memoria (16 GB).
- **Métricas:** por mes de scoring (media ± desvío entre meses), IC 95% con bootstrap Poisson por CUIT (100 réplicas), además de las métricas con todas las filas juntas y con una fila al azar por CUIT.
- **Validación OOT:** entrena con los CUITs de train en los meses OOT "train" y evalúa con los CUITs de test en los meses OOT "test".
- **Hallazgo (prueba con 2% de los CUITs):** con una fila al azar por CUIT, la prevalencia de B sube de ~7,7% a ~15,5%. Los que caen en mora salen de sit 1 y aportan menos filas al panel; los buenos pagadores aportan muchas. Con todas las filas pesan más los clientes estables, lo que es coherente con el uso real (cada mes se puntúa a toda la cartera en sit 1).

## 3. Limitaciones

1. **Target C sin validación fuera de tiempo.** Sus métricas son una cota superior optimista. Como referencia se usa la caída CUIT→OOT observada en A y B.
2. **Historia corta.** Son 31 meses en total y quedan aprox. 9 meses de *t* en el train OOT de B tras aplicar la ventana t+2. Esto limita la validación temporal y la captura de ciclos.
3. **Montos nominales en contexto de alta inflación.** Se mitiga con el CER y con ratios/deciles, pero la deflación es una aproximación.
4. **Granularidad de la Central de Deudores.** Es mensual, informa la situación por entidad y no la fecha exacta de atraso. Los días de mora se infieren de la situación.
5. **Rezago de publicación.** Obliga a dejar fuera *t+1*, lo que reduce la ventana efectiva de predicción.
6. **Series macro revisadas, no en tiempo real.** El BCRA revisa datos hacia atrás (por ejemplo, la tasa de consumo de 202608 pasó de 79,37 a 78,83 entre el 27/10 y el 18/11/2026). Se usan los valores revisados a la fecha de descarga, no los que el banco veía en cada momento. Los rezagos de publicación se infieren de solo dos descargas.
7. **Huecos en las series del BCRA.** Faltan 202601 y 202609 en la situación de cartera del sistema y en BADLAR. Quedan vacíos, igual que las variaciones a 3 meses que dependen de esos meses.

---

## 4. Complicaciones técnicas

1. **Memoria.** El panel tiene aprox. 35M de filas. La versión original de 03_split.py se quedaba sin memoria al unir aprox. 48 variables macro float64, así que se reescribió para procesar por row group. La corrida de 02 llegó a un pico de 18,2 de 19,1 GB, por eso los cambios de las dudas 1 a 3 se implementan en **una única** recorrida de 02.
2. **Dos archivos 24DSF.** Se combinaron dos archivos superpuestos (corrimiento de 7 meses). Se verificó una coincidencia del 100% en el solapamiento.
4. **Problemas de calidad en macro_bcra.csv** (detectados el 2026-11-17, sin corregir todavía):
   - Los datos no publicados quedan en **0 en vez de vacío (NaN)**. Ejemplos: 202609 en casi todas las series y **202601 en las series de situación de cartera (din1)**. Eso corrompe var_mora3plus_1m/3m y var_sit2_1m/3m en 202601-202604 (por ejemplo, -5,3 y luego +6,1 puntos).
   - Las 5 series de rentabilidad (resultado_neto, intereses ganados/pagados, margen, roa_proxy) son **siempre 0**: los códigos de serie no traen datos.
   - tasa_hipotecarios_privado_pct está en 0 en 22 de los 33 meses desde 2024.
   - n_entidades_financieras toma valores de cientos de millones: el código de serie no corresponde a una cantidad de entidades.
   - Rezagos de publicación: ver 2.7. La primera descarga hizo pasar un hueco de la serie (202609) por un atraso de publicación, y la segunda lo corrigió.
   - Corregido en 01 (rama implementar-decisiones-1-6): los faltantes quedan vacíos, las variaciones se calculan por mes calendario y quedan vacías si falta un extremo, y el CER se baja de la API v4.0 del BCRA (variable 30, último día de cada mes).
5. **Datos sensibles.** Los datos contienen CUITs reales, por lo que nunca se suben al repositorio público; solo se versiona el código.

---

## 5. Dudas abiertas

