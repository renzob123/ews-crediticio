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
- Estado: **pendiente de implementar** (el inicio de la ventana pasa a ser un parámetro en 02).

### 2.4 Duda 2: inflación (decidida 2026-11-15)
- **Deflactar con CER** (se eligió CER sobre IPC INDEC; se agrega en 01): monto_sant_t, monto_sant_t3 (con el CER del propio mes), monto_otras_t, sus logs, var_monto_3m_sant y las variables macro nominales.
- **Features libres de inflación:** share_sant = monto_sant / (monto_sant + monto_otras) en *t* y *t3* (requiere crear monto_otras_t3). También deciles por período en *t* y *t3* para monto_sant y monto_otras (en otras: decil 0 = sin deuda y deciles 1-10 entre los que tienen deuda > 0). Se calculan en 03, con una primera pasada para los cortes de cada período.
- **Sensibilidad OOT:** se comparan montos nominales, montos reales y solo ratios/deciles.
- Estado: **pendiente de implementar**.

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
- **Regla de rezago** (decidida 2026-11-18): la macro describe el mes *t*, el mismo momento en que se informó al cliente. Se usa el valor de *t* si ya estaba publicado al momento del scoring (fines de *t+1*); si no, el de *t-1*. Según la descarga del 27/10/2026: tipo de cambio y tasas de consumo en *t*; mora del sistema, BADLAR, crédito y composición de la cartera en *t-1*. Se va a confirmar con una segunda descarga a principios de diciembre de 2026.
- **Reducción a 10 variables macro** (decidida 2026-11-18). Motivos: (a) redundancia: sistema y privado correlacionan en 1,000, y todos los niveles en pesos correlacionan entre 0,97 y 0,999 entre sí y con el tipo de cambio, porque reflejan inflación y no crédito ni liquidez; (b) hay solo ~25 valores distintos por variable (uno por mes), con alto riesgo de sobreajustar al período.

  | Variable | Mes |
  |---|---|
  | mora_3plus_privado_pct | *t-1* |
  | sit2_privado_pct | *t-1* |
  | var_mora3plus_3m | *t-1* |
  | var_sit2_3m | *t-1* |
  | tasa_consumo_corto_pct | *t* |
  | badlar_privados_tna_pct | *t-1* |
  | spread_consumo_badlar_pct | *t-1* |
  | var_tc_3m_pct (nueva: variación % del tipo de cambio en 3 meses) | *t* |
  | var_credito_real_3m_pct (nueva: crédito privado deflactado por CER) | *t-1* |
  | pct_personales | *t-1* |

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

## 3. Limitaciones

1. **Target C sin validación fuera de tiempo.** Sus métricas son una cota superior optimista. Como referencia se usa la caída CUIT→OOT observada en A y B.
2. **Historia corta.** Son 31 meses en total y quedan aprox. 9 meses de *t* en el train OOT de B tras aplicar la ventana t+2. Esto limita la validación temporal y la captura de ciclos.
3. **Montos nominales en contexto de alta inflación.** Se mitiga con el CER y con ratios/deciles, pero la deflación es una aproximación.
4. **Granularidad de la Central de Deudores.** Es mensual, informa la situación por entidad y no la fecha exacta de atraso. Los días de mora se infieren de la situación.
5. **Rezago de publicación.** Obliga a dejar fuera *t+1*, lo que reduce la ventana efectiva de predicción.

---

## 4. Complicaciones técnicas

1. **Memoria.** El panel tiene aprox. 35M de filas. La versión original de 03_split.py se quedaba sin memoria al unir aprox. 48 variables macro float64, así que se reescribió para procesar por row group. La corrida de 02 llegó a un pico de 18,2 de 19,1 GB, por eso los cambios de las dudas 1 a 3 se implementan en **una única** recorrida de 02.
2. **Dos archivos 24DSF.** Se combinaron dos archivos superpuestos (corrimiento de 7 meses). Se verificó una coincidencia del 100% en el solapamiento.
4. **Problemas de calidad en macro_bcra.csv** (detectados el 2026-11-17, sin corregir todavía):
   - Los datos no publicados quedan en **0 en vez de vacío (NaN)**. Ejemplos: 202609 en casi todas las series y **202601 en las series de situación de cartera (din1)**. Eso corrompe var_mora3plus_1m/3m y var_sit2_1m/3m en 202601-202604 (por ejemplo, -5,3 y luego +6,1 puntos).
   - Las 5 series de rentabilidad (resultado_neto, intereses ganados/pagados, margen, roa_proxy) son **siempre 0**: los códigos de serie no traen datos.
   - tasa_hipotecarios_privado_pct está en 0 en 22 de los 33 meses desde 2024.
   - n_entidades_financieras toma valores de cientos de millones: el código de serie no corresponde a una cantidad de entidades.
   - Rezagos de publicación observados en la descarga del 27/10/2026 (aprox. el momento de scoring de *t* = 202609): situación de cartera y préstamos por tipo llegan hasta *t-1*; reservas, tipo de cambio, base monetaria y tasas de consumo/adelantos, hasta *t*; depósitos, crédito privado y BADLAR, hasta *t-1*. Es una sola foto: el rezago real puede variar entre meses.
5. **Datos sensibles.** Los datos contienen CUITs reales, por lo que nunca se suben al repositorio público; solo se versiona el código.

---

## 5. Dudas abiertas

- Se confirmarán los rezagos de la macro con una segunda descarga (principios de diciembre de 2026).
- La regresión logística sobre ~28M de filas puede no entrar en memoria. Si pasa, se decidirá si se entrena sobre una muestra de train (por CUIT).
