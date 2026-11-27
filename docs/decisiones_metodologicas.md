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
- **Resultado (2026-11-25):** la macro no aporta fuera de tiempo (2.13c, 2.13d). **El modelo principal pasa a ser sin macro**; las corridas con macro quedan como sensibilidad.
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
- **Búsqueda de hiperparámetros:** búsqueda aleatoria de 6 configuraciones (bajado de 10 el 2026-11-20 por tiempo: cada configuración con todos los datos tarda más de 1 hora), GroupKFold(3) por CUIT, early stopping (50 árboles sin mejora) en el fold de validación y PR-AUC como métrica de selección, porque con 7% de positivos es más exigente que el AUC. Por costo, se hace sobre una muestra del 20% de los CUITs de train. El modelo final se entrena con todo train, usando el promedio de árboles de los folds.
- **Regresión logística:** C = 1 sin tuning (con millones de filas la regularización casi no cambia nada). Montos en log; recorte de extremos en los percentiles 0,5 y 99,5; imputación con mediana más indicador de faltante; escalado y one-hot de sexo y rango de edad. Todo dentro de un Pipeline ajustado solo con train. **Se entrena con el 25% de los CUITs de train**, porque con todas las filas no entra en memoria (16 GB).
- **Métricas:** por mes de scoring (media ± desvío entre meses), IC 95% con bootstrap Poisson por CUIT (100 réplicas), además de las métricas con todas las filas juntas y con una fila al azar por CUIT.
- **Validación OOT:** entrena con los CUITs de train en los meses OOT "train" y evalúa con los CUITs de test en los meses OOT "test".
- **Hallazgo (prueba con 2% de los CUITs):** con una fila al azar por CUIT, la prevalencia de B sube de ~7,7% a ~15,5%. Los que caen en mora salen de sit 1 y aportan menos filas al panel; los buenos pagadores aportan muchas. Con todas las filas pesan más los clientes estables, lo que es coherente con el uso real (cada mes se puntúa a toda la cartera en sit 1).

### 2.12 Resultados: modelo principal (XGBoost, target B, validación por CUIT, con macro; 2026-11-22)
- Train: 23,2M filas, 1,64M CUITs, 202407-202601, 7,54% positivos. Test: 5,8M filas, 409 mil CUITs, 19 meses de scoring, 7,56% positivos.
- Hiperparámetros (6 configuraciones, todas con PR-AUC CV entre 0,318 y 0,323): max_depth 6, learning_rate 0,05, min_child_weight 1, colsample 0,8, subsample 1, lambda 1, 783 árboles.
- **Métricas por mes de scoring (media ± desvío entre meses; IC 95% bootstrap por CUIT):** AUC 0,807 ± 0,013 [0,806; 0,808]; PR-AUC 0,315 ± 0,054 [0,312; 0,317]; captura top 5% 27,1%; **top 10% 42,6%** [42,4; 42,9]; top 20% 61,7%.
- Todas las filas juntas: AUC 0,813, PR-AUC 0,327. Una fila por CUIT: AUC 0,818, PR-AUC 0,494 (prevalencia 14,9%).
- **Estabilidad entre meses:** el AUC va de 0,780 (202407) a 0,823 (202504) y la captura del top 10% se mantiene entre 41% y 44%. El PR-AUC varía más (0,20-0,38) porque acompaña a la prevalencia del mes (4,5%-9,2%): con más positivos el PR-AUC sube aunque la capacidad de ordenar sea similar.
- **Importancia (ganancia):** n_entidades_otras_t 23,5%, otras_tiene_billetera 16,2%, meses_consec_normal_sant_6m 9,9%, rango_edad_proxy 7,3%, otras_tiene_prov_nofin 5,0%. El endeudamiento en otras entidades (sobre todo billeteras y proveedores no financieros) es la señal principal. Las 10 variables macro suman 3,0% de la ganancia: aportan poco a nivel individual, como se esperaba (son iguales para todos los clientes de un mes). Falta confirmarlo con la corrida sin macro.
- Duración: ~6,7 h en total (con 2 h de hibernación de la PC en el medio; el proceso siguió igual).

- **Reevaluación con las métricas de 2.14 (2026-11-22, `--solo-evaluar`, mismos AUC y PR-AUC):** Gini 0,614 [0,611; 0,617]; precisión top 5/10/20% = 40,7% / 32,1% / 23,3%; lift = 5,4 / 4,3 / 3,1; Brier 0,172; probabilidad media sobre tasa = 5,1 (no calibrado, como se esperaba por `scale_pos_weight`). La primera reevaluación de la logística se cortó por memoria al predecir test (el pipeline de sklearn pasa a float64 y copia los datos); se resolvió prediciendo de a 500 mil filas.

### 2.13 Resultados: modelo base (regresión logística, mismo escenario; 2026-11-22)
- Entrenada con el 25% de los CUITs de train (5,8M filas). La primera vez se cortó por memoria: el pipeline de sklearn llega a ~7,8 GB de pico al preparar los datos (float64 y copias). En el segundo intento, con ~7 GB libres, terminó en 10 minutos.
- **Métricas por mes:** AUC 0,778 ± 0,016 [0,776; 0,779]; PR-AUC 0,261 ± 0,051; captura top 10% 37,8% [37,5; 38,1]; top 20% 56,8%.
- **XGBoost vs logística:** +0,029 de AUC, +0,054 de PR-AUC y +4,8 puntos de captura en el top 10% (42,6% vs 37,8%). XGBoost es mejor en los 19 meses y los IC no se superponen. La ventaja es mayor en los meses extremos (202407: AUC 0,780 vs 0,743), lo que sugiere relaciones no lineales que la logística no capta.
- Coeficientes (variables estandarizadas): la edad es lo que más pesa (65+ y 50-65 reducen el riesgo frente a 18-30), seguida de los montos en otras entidades y los meses seguidos en situación normal. Hay variables muy correlacionadas (monto real y decil de monto en otras entidades, con signos opuestos), así que los coeficientes individuales no deben interpretarse como efectos aislados.
- Reevaluación con las métricas de 2.14: Gini 0,556 [0,552; 0,558]; precisión top 10% 28,5% (XGBoost 32,1%); lift top 10% 3,8× (XGBoost 4,3×); Brier 0,185; probabilidad media sobre tasa 5,4. La logística también sale descalibrada porque usa `class_weight="balanced"`, que tiene el mismo efecto que `scale_pos_weight`.

### 2.13b Resultados: XGBoost con validación OOT (target B, con macro; 2026-11-22)
- Train OOT: 12,3M filas, 1,51M CUITs, t de 202407 a 202504 (10 meses, aprox. la mitad de filas que la validación por CUIT). Hueco de 6 meses. Test: CUITs de test en 202511–202601 (875.703 filas, 3 meses, 7,35% de positivos). Tardó 2,9 h (el train más chico acelera el tuning).
- Tuning: la mejor configuración es la misma que en la validación por CUIT (max_depth 6, lr 0,05, min_child_weight 1, colsample 0,8, subsample 1, lambda 1), con 593 árboles (783 por CUIT). Las 6 configuraciones dieron PR-AUC CV entre 0,288 y 0,296: el resultado es poco sensible a los hiperparámetros.
- **Métricas por mes:** AUC 0,787 ± 0,004 [0,785; 0,790]; PR-AUC 0,274 [0,268; 0,278]; captura top 10% 39,3% [38,8; 39,7]; top 20% 57,9%.
- **Comparación justa con la validación por CUIT:** el test OOT son exactamente las mismas filas que el test por CUIT en esos 3 meses (mismos n). En esos meses el modelo OOT pierde en promedio **−0,010 de AUC, −0,018 de PR-AUC y −1,8 puntos de captura top 10%** (por ejemplo, 202511: AUC 0,800 → 0,791). La caída de la media general (0,807 → 0,787) se explica sobre todo porque los últimos meses son más difíciles también para el modelo por CUIT (menor prevalencia, AUC ~0,795), no por la validación temporal.
- **Lectura:** el modelo es estable en el tiempo: entrenado con datos que terminan 6 meses antes y con la mitad de historia, conserva casi toda la capacidad de discriminación.
- Reevaluación con las métricas de 2.14: Gini 0,574 [0,570; 0,580]; precisión top 10% 28,8%; lift top 10% 3,9×; Brier 0,204; probabilidad media sobre tasa 5,7.
- Importancia: las mismas variables arriba, con otro orden (otras_tiene_billetera 22,9%, n_entidades_otras_t 14,9%, meses_consec_normal_sant_6m 14,3%, rango_edad_proxy 7,6%). Macro: 2,1% del total.

### 2.13c Resultados: XGBoost sin macro (target B, validación por CUIT; 2026-11-23)
- 41 variables (sin las 10 macro). Misma mejor configuración que con macro, con 721 árboles. Tardó 5,7 h.
- **Métricas por mes:** AUC 0,805 [0,803; 0,806]; Gini 0,609; PR-AUC 0,308; captura top 10% 42,0%; precisión top 10% 31,7%; lift top 10% 4,2×.
- **Con macro vs sin macro:** la macro suma **+0,0025 de AUC, +0,006 de PR-AUC y +0,6 puntos de captura top 10%**. Es poco, pero consistente: es mejor en los **19 de 19 meses**. La ganancia es mayor en los meses de los extremos (202407: +0,006 de AUC y +1,9 puntos de captura; 202601: +0,004), que es donde las condiciones macro se apartan más del promedio.
- **Cómo puede ayudar la macro dentro de un mes:** las métricas se calculan mes a mes y la macro vale lo mismo para todos los clientes de un mes, así que por sí sola no cambia el orden. Solo puede ayudar por **interacciones** (por ejemplo, que tener billetera o muchas entidades pese distinto según la mora del sistema o las tasas). Eso explica que el efecto sea chico.
- **Cuidado:** en la validación por CUIT el modelo vio esos mismos meses en train, así que la macro podría estar funcionando en parte como un identificador del mes. La prueba más honesta es la OOT sin macro (pendiente).
- Sin macro, la importancia se concentra más en n_entidades_otras_t (30,6%), seguida de otras_tiene_billetera (13,2%) y meses_consec_normal_sant_6m (10,9%).

### 2.13d Resultados: XGBoost OOT sin macro (target B; 2026-11-24)
- Mismo train y test que la OOT con macro. El tuning eligió otra configuración (max_depth 4, lr 0,1, min_child_weight 1, colsample 0,6, lambda 10; 816 árboles), aunque empatada con la de siempre (PR-AUC CV 0,2906 vs 0,2905). Tardó 2,5 h.
- **Métricas por mes:** AUC 0,787 [0,785; 0,790]; PR-AUC 0,271; captura top 10% 39,0%; precisión top 10% 28,7%; lift 3,9×.
- **OOT con macro vs sin macro:** +0,0001 de AUC, +0,003 de PR-AUC y +0,2 puntos de captura top 10%. En AUC es empate (202512 sale mejor sin macro). La ganancia de la macro que se veía en la validación por CUIT (+0,0025 de AUC, 19 de 19 meses) **prácticamente desaparece en meses que el modelo no vio**.
- **Interpretación:** es consistente con que, en la validación por CUIT, la macro funcionaba en parte como un identificador del mes y no como una señal que se generalice al futuro. Atenuantes: el train OOT tiene solo 10 meses de *t*, con poca variación macro para aprender, y los meses de test pueden tener valores macro fuera del rango visto en train.
- **Conclusión para la tesis:** con estos datos, la macro **no aporta capacidad predictiva fuera de tiempo**. El riesgo lo explica el comportamiento del cliente en el sistema financiero.
- **Decisión (2026-11-25): el modelo principal pasa a ser sin macro.** Con el mismo desempeño fuera de tiempo, es más simple de explicar y de mantener (no depende de la publicación de las series del BCRA). Las corridas con macro quedan como sensibilidad. En `04_modelado.py` el valor por defecto pasa a sin macro y la opción `--sin-macro` se reemplaza por `--macro`. Nuevo resultado principal: XGBoost por CUIT sin macro (AUC 0,805, PR-AUC 0,308, captura top 10% 42,0%; sección 2.13c). La logística base se vuelve a correr sin macro para comparar en igualdad de condiciones.
- Importancia sin macro (OOT): otras_tiene_billetera 23,0%, meses_consec_normal_sant_6m 12,3%, n_entidades_otras_t 9,8%, rango_edad_proxy 8,1%.

### 2.13e Resultados: LightGBM y logística sin macro (target B, validación por CUIT; 2026-11-25)
- **LightGBM** (sensibilidad: otro boosting). Tuning reutilizado de la corrida cortada (num_leaves 63, lr 0,05, min_child_samples 100, feature_fraction 0,8, sin bagging ni L2; 716 árboles). El entrenamiento final con el arreglo de memoria tardó 19 min. AUC 0,805 [0,804; 0,806]; Gini 0,610; PR-AUC 0,310; captura top 10% 42,1%; precisión 31,8%; lift 4,2×; probabilidad media sobre tasa 5,1.
- **LightGBM vs XGBoost:** prácticamente iguales (LightGBM +0,0005 de AUC, +0,001 de PR-AUC, +0,1 puntos de captura; los IC se superponen). LightGBM sale apenas mejor en los 19 meses, pero la diferencia no tiene relevancia práctica. **El resultado no depende de la implementación del boosting.** Se mantiene XGBoost como principal (decisión del 2026-11-20). Importancia: mismas variables arriba (n_entidades_otras_t 24,3%, otras_tiene_billetera 10,7%, meses_consec_normal_sant_6m 10,2%, rango_edad_proxy 7,2%).
- **Logística sin macro** (base, 25% de los CUITs de train): AUC 0,778; Gini 0,555; PR-AUC 0,261; captura top 10% 37,8%; lift 3,8×. Igual que con macro (+0,0002 de AUC con macro): en la logística, la macro tampoco aporta.
- **XGBoost vs logística (ambos sin macro):** +0,027 de AUC, +0,048 de PR-AUC, +4,3 puntos de captura top 10%; XGBoost gana en los 19 meses.

### 2.13f Resultados: variantes de montos (XGBoost, target B, validación por CUIT, sin macro; 2026-11-26)
| Montos | Variables | AUC | PR-AUC | Captura top 10% | Mejor que principal (AUC) |
|---|---|---|---|---|---|
| principal (reales + ratios) | 41 | 0,805 | 0,308 | 42,0% | — |
| solo nominales | 35 | 0,805 (+0,0002) | 0,308 | 42,1% | 16/19 meses |
| solo reales | 35 | 0,804 (−0,0007) | 0,307 | 42,0% | 2/19 |
| solo ratios/deciles | 34 | 0,798 (−0,0063) | 0,296 | 40,8% | 0/19 |
- Cada variante tuvo su tuning propio (unas 4,8 h cada una).
- **Nominal ≈ real ≈ principal** en la validación por CUIT. Es lo esperable: las métricas se calculan mes a mes y, dentro de un mes, el CER es el mismo para todos, así que deflactar no cambia el orden de los clientes. Además, el modelo vio todos los meses en train, así que puede "aprender" el nivel de precios de cada mes.
- **Solo ratios pierde** (−0,006 de AUC, −1,2 puntos de captura, en los 19 meses): el tamaño de la deuda en pesos aporta información que los shares y deciles no capturan del todo.
- **Error de diseño (detectado al terminar):** la decisión 2.4 decía que esta sensibilidad se hace con **validación OOT**, que es donde la inflación importa (en meses futuros los montos nominales quedan fuera del rango visto en train). Estas corridas se hicieron con validación por CUIT; sirven como referencia, pero la comparación relevante es la OOT (pendiente: 3 corridas de unas 2,5 h).

### 2.13g Resultados: variantes de montos con validación OOT (XGBoost, target B, sin macro; 2026-11-27)
| Montos (OOT) | AUC [IC 95%] | PR-AUC | Captura top 10% | Lift top 10% |
|---|---|---|---|---|
| principal (reales + ratios) | 0,787 [0,785; 0,790] | 0,271 | 39,0% | 3,9× |
| solo nominales | 0,787 [0,785; 0,790] | 0,270 | 39,0% | 3,9× |
| solo reales | 0,787 [0,785; 0,790] | 0,271 | 39,1% | 3,9× |
| solo ratios/deciles | 0,780 [0,779; 0,783] | 0,258 | 37,8% | 3,8× |
- Cada variante con su propio tuning. La de reales se cortó una vez por memoria del sistema (5 GB libres por otros programas) después del tuning y se terminó con `--reusar-tuning` sobre su propio tuning.
- **Nominal = real = principal también fuera de tiempo.** Aun con 6 a 9 meses de inflación entre el final de train y el test, deflactar no mejora el modelo. Una explicación posible es que los montos pesan poco frente a las variables de conteo e indicadores (entidades, billeteras, meses en situación normal), y que el árbol usa los montos sobre todo para separar deudas chicas de grandes, una distinción que la inflación de unos meses no llega a borrar. Con un horizonte más largo o con más inflación, los nominales sí podrían degradarse.
- **Solo ratios pierde también en OOT** (−0,0065 de AUC, −1,3 puntos de captura, en los 3 meses), igual que en la validación por CUIT en esos mismos meses (−0,007 / −1,5): el tamaño de la deuda en pesos aporta información propia.
- **Conclusión:** se mantiene el conjunto principal (reales + ratios). Deflactar con CER no cambia los resultados, pero es lo correcto conceptualmente y protege frente a horizontes más largos; los ratios solos no alcanzan.

### 2.14 Métricas de evaluación con clases desbalanceadas (decidida 2026-11-22)
- **Problema:** con 7,6% de positivos, el AUC no depende de la prevalencia y puede verse bien aunque la lista de alertas tenga muchos falsos positivos.
- **Métricas principales:** PR-AUC (la base al azar es la prevalencia, ~0,076) y, para el top 5/10/20% de scores, **captura, precisión y lift**. Responden la pregunta operativa del EWS: si se revisa a los N clientes de peor score, cuántos problemas se encuentran.
- **Métricas secundarias:** AUC y **Gini (= 2·AUC − 1)**. Son el estándar en riesgo de crédito y permiten comparar con la literatura.
- **Calibración:** Brier y el cociente entre la probabilidad media predicha y la tasa observada (1 = calibrado). En la prueba con el 2% de los CUITs, XGBoost da 5,2: la probabilidad media predicha es ~5 veces la tasa real, por `scale_pos_weight`. El Brier (0,177) es peor que predecir la prevalencia para todos (~0,07). **Los scores ordenan bien, pero no son probabilidades.** Para usarlos como PD habría que recalibrarlos (por ejemplo, con regresión isotónica o Platt sobre una parte de train). Si el EWS solo usa el ranking, no hace falta.
- **Implementación:** `04_modelado.py --solo-evaluar` carga el modelo guardado y recalcula las métricas de test sin reentrenar, conservando train, hiperparámetros y tiempos de la corrida original. Se verificó con el modelo de prueba del 2%: AUC y PR-AUC idénticos a la corrida original (0,7971 / 0,2907).

## 3. Limitaciones

1. **Target C sin validación fuera de tiempo.** Sus métricas son una cota superior optimista. Como referencia se usa la caída CUIT→OOT observada en A y B.
2. **Historia corta.** Son 31 meses en total y quedan aprox. 9 meses de *t* en el train OOT de B tras aplicar la ventana t+2. Esto limita la validación temporal y la captura de ciclos.
3. **Montos nominales en contexto de alta inflación.** Se mitiga con el CER y con ratios/deciles, pero la deflación es una aproximación.
4. **Granularidad de la Central de Deudores.** Es mensual, informa la situación por entidad y no la fecha exacta de atraso. Los días de mora se infieren de la situación.
5. **Rezago de publicación.** Obliga a dejar fuera *t+1*, lo que reduce la ventana efectiva de predicción.
6. **Series macro revisadas, no en tiempo real.** El BCRA revisa datos hacia atrás (por ejemplo, la tasa de consumo de 202608 pasó de 79,37 a 78,83 entre el 27/10 y el 18/11/2026). Se usan los valores revisados a la fecha de descarga, no los que el banco veía en cada momento. Los rezagos de publicación se infieren de solo dos descargas.
7. **Huecos en las series del BCRA.** Faltan 202601 y 202609 en la situación de cartera del sistema y en BADLAR. Quedan vacíos, igual que las variaciones a 3 meses que dependen de esos meses.
8. **La macro no generaliza fuera de tiempo.** Con 31 meses de historia (10 de train OOT para B) hay muy poca variación macro para aprender; su aporte en la validación por CUIT desaparece en OOT (2.13d). Un panel más largo, con un ciclo completo, podría cambiar este resultado.

---

## 4. Complicaciones técnicas

1. **Memoria.** El panel tiene aprox. 35M de filas. La versión original de 03_split.py se quedaba sin memoria al unir aprox. 48 variables macro float64, así que se reescribió para procesar por row group. La corrida de 02 llegó a un pico de 18,2 de 19,1 GB, por eso los cambios de las dudas 1 a 3 se implementan en **una única** recorrida de 02.
   - **Memoria en 04_modelado.py (2026-11-20):** la primera corrida completa de XGBoost se cortó por falta de memoria del sistema durante la búsqueda de hiperparámetros. En ese momento estaban en memoria todo train y, además, la muestra de tuning, y el lector de parquet leía bloques por adelantado. La PC tiene 16 GB y unos 8 GB libres. Cambios (sin efecto en los resultados: la prueba con 20% dio idéntico): (1) la muestra de tuning se carga sola y se libera antes de cargar todo train; (2) lectura de a un bloque y conversión Arrow→pandas sin copia doble; (3) se descartan las columnas que no son variables en lugar de copiar `tr[features]`. Pico con 20% de los CUITs: de 5,0 a 3,8 GB.
   - **Segundo corte (2026-11-21, 00:10):** con esos cambios el tuning terminó entero (6 configuraciones, unas 4,5 h; PR-AUC entre 0,318 y 0,323), pero la corrida se cortó al armar la matriz de entrenamiento final. Mediciones con todo train (23,2M filas): cargar la tabla llega a 5,4 GB de pico para 2,8 GB de datos (el lector de parquet y el asignador de Arrow retienen memoria), y armar el `QuantileDMatrix` desde esa tabla supera los 8,2 GB. Leer de a bloques con un iterador tampoco alcanzó (XGBoost recorre los datos 4 veces y la memoria crece en cada pasada: ~7,8 GB). **Solución:** XGBoost en **memoria externa** (`ExtMemQuantileDMatrix`): las páginas ya discretizadas van a una carpeta temporal en disco que se borra al terminar. Pico con todo train: 3,4 GB. Costo: más lento (~14 s por árbol, ~3 h para 783 árboles). Resultados idénticos en la prueba con 2%. Se agregó `--reusar-tuning` para no repetir la búsqueda ya hecha.
   - **LightGBM con todo train (2026-11-22):** tenía el mismo problema (armaba su `Dataset` desde la tabla en memoria). Ahora train se escribe de a bloques a un archivo binario float32 en una carpeta temporal y se abre como *memory-map*: el sistema operativo lo lee del disco a demanda y LightGBM arma su matriz discretizada a partir de ahí. Las categóricas van como su código (con las categorías fijas en todo el script) y se predice con la misma codificación. Prueba con 2%: resultados idénticos a la versión anterior. Pico con 30% de train: 3,6 GB (incluye páginas del archivo, que el sistema puede liberar). Falta medir con todo train.
   - **LightGBM cortado por memoria con todo train (2026-11-25):** el tuning terminó y quedó guardado, pero el entrenamiento final se cortó. Medición: al armar el `Dataset` desde el *memory-map*, el sistema contaba como ocupadas las páginas leídas del archivo (3,8 GB) y la memoria libre bajaba a 2,9 GB; al entrenar, el modo de histogramas por filas duplicaba los datos discretizados y bajaba a 2,4 GB. **Solución:** (a) LightGBM lee el archivo de a tramos con lecturas comunes (`lgb.Sequence`), sin *memory-map*; (b) el `Dataset` se construye antes de entrenar y se cierra el archivo; (c) `force_col_wise=True` en el modelo final (histogramas por columna; es solo una forma de cálculo). Pico del proceso: de 7,0 GB a 3,6 GB; mínimo libre: de 2,4 a 5,5 GB. Además es más rápido (1,6 s por árbol en vez de 2,5). Prueba con 2% y macro: AUC 0,7986 y PR-AUC 0,2959 (antes 0,7991 / 0,2945); la pequeña diferencia viene de que la muestra de filas con que LightGBM elige los cortes de cada variable es otra.
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

