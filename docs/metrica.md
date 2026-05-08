# La métrica: Weighted Pearson Correlation

Documento de referencia sobre la métrica de evaluación del reto **Wunder Predictorium**, qué representa, y cómo afecta al diseño del modelo y la loss.

Implementación real en `src/utils.py:6` (`weighted_pearson_correlation`).

---

## 1. Recordatorio: Pearson "normal"

La correlación de Pearson mide la relación lineal entre dos variables. Su fórmula:

$$
\rho = \frac{\text{Cov}(y_{\text{true}},\, y_{\text{pred}})}{\sigma_{y_{\text{true}}} \cdot \sigma_{y_{\text{pred}}}}
$$

Equivalentemente:

$$
\rho = \frac{\sum_i (y_{\text{true},i} - \bar y_{\text{true}})(y_{\text{pred},i} - \bar y_{\text{pred}})}{\sqrt{\sum_i (y_{\text{true},i} - \bar y_{\text{true}})^2}\;\sqrt{\sum_i (y_{\text{pred},i} - \bar y_{\text{pred}})^2}}
$$

Propiedades clave:

- **Rango**: \[−1, 1\]. 1 = perfecto, 0 = sin relación lineal, −1 = perfectamente inverso.
- **Insensible a escala y a desplazamientos**: si multiplicas tus predicciones por 10 o les sumas 100, la correlación no cambia. Mide *forma*, no magnitud absoluta.
- Cada punto pesa lo mismo en la suma.

---

## 2. Qué cambia en la versión "weighted"

Mirando la implementación real, cambian dos cosas respecto a Pearson estándar:

### (a) Pesos por amplitud del target

Cada punto tiene un peso $w_i = \max(|y_{\text{true},i}|,\, 10^{-8})$. Las medias, varianzas y covarianza pasan a ser **versiones ponderadas**:

$$
\bar y_{\text{true}}^{(w)} = \frac{\sum_i w_i \cdot y_{\text{true},i}}{\sum_i w_i}
$$

$$
\text{cov}^{(w)} = \frac{\sum_i w_i \cdot (y_{\text{true},i} - \bar y_{\text{true}}^{(w)})(y_{\text{pred},i} - \bar y_{\text{pred}}^{(w)})}{\sum_i w_i}
$$

$$
\text{var}_{\text{true}}^{(w)} = \frac{\sum_i w_i \cdot (y_{\text{true},i} - \bar y_{\text{true}}^{(w)})^2}{\sum_i w_i} \quad \text{(análogo para pred)}
$$

$$
\rho_w = \frac{\text{cov}^{(w)}}{\sqrt{\text{var}_{\text{true}}^{(w)} \cdot \text{var}_{\text{pred}}^{(w)}}}
$$

### (b) Clipping de predicciones a \[−6, 6\]

`y_pred_clipped = np.clip(y_pred, -6.0, 6.0)`. Antes de calcular nada, se recortan las predicciones extremas. Sirve para que un único valor disparatado no desbarate la métrica.

---

## 3. Qué representa la ponderación

La idea más importante: **los puntos donde el target es grande pesan más; los puntos cercanos a cero pesan casi nada.**

Dos casos extremos:

- Un step donde $y_{\text{true}} = 0.01$ (movimiento minúsculo, ruido). Peso = 0.01. Aunque te equivoques mucho ahí, el daño en la métrica es ínfimo.
- Un step donde $y_{\text{true}} = 4.5$ (movimiento gordo). Peso = 4.5, **450 veces más influencia** que el caso anterior. Aquí acertar (o fallar) es lo que decide el score.

**Interpretación financiera**: en trading lo que importa es acertar los movimientos grandes, porque son los que generan PnL real. Los movimientos pequeños son ruido y aunque los predigas mal, no pasa nada. La métrica está diseñada para reflejar eso.

---

## 4. Implicaciones para el modelado

### (a) La escala de tus predicciones no importa

Pearson es invariante a escala. Si el modelo escupe 0.0001 y los targets son del orden de 1, mientras la **forma** sea correcta, sacarás buen score. **No hace falta normalizar las predicciones para que coincidan en magnitud con el target**. Esto da flexibilidad: puedes entrenar con `tanh` final, con normalización, lo que sea.

### (b) MSE no es una buena loss subrogada

Un MSE estándar trata todos los puntos por igual. Pero la métrica de evaluación pondera por $|y|$. Hay un mismatch:

- MSE penaliza mucho equivocarte por 0.5 cuando el target es 0.05 (error relativo enorme).
- La métrica apenas penaliza eso (peso = 0.05).

**Mejores opciones de loss**:

1. **Weighted MSE** con pesos $\propto |y_{\text{true}}|$ — alinea la loss con la métrica.
2. **Pearson loss directa** ($1 - \rho_w$ o $-\rho_w$) — diferenciable, optimiza la métrica directamente. Es lo que se suele hacer en concursos con esta métrica.
3. **Huber + weighting** si preocupan outliers en el target.

### (c) El clipping a \[−6, 6\] da una pista sobre los targets

Si han puesto ese rango, es porque en train hay valores que llegan ahí, o cerca. Los targets están **anonimizados pero estandarizados** — probablemente son z-scores de retornos o similares, y \[−6, 6\] cubre prácticamente todo manteniendo unos pocos outliers a raya. **No tiene sentido predecir 100**, te lo van a recortar.

### (d) La métrica final es el promedio de t0 y t1

```python
scores["weighted_pearson"] = np.mean(list(scores.values()))
```

El score final es $(\rho_w(t_0) + \rho_w(t_1)) / 2$. Esto significa:

- No se puede "compensar" un $t_0$ horrible con un $t_1$ buenísimo más allá del promedio simple.
- Ambos targets pesan igual; conviene un modelo multi-output equilibrado y no uno que sacrifique uno por el otro.
- Puede valer la pena entrenar **dos modelos separados** o un **multi-output con dos heads**, según cuán correlados/distintos sean $t_0$ y $t_1$ (esto se descubre en exploración).

---

## 5. Qué valores son "buenos"

Para no asustarse con números bajos:

- En HFT/microestructura, una correlación entre **0.05 y 0.20** ya suele considerarse decente.
- **0.30+** sería excelente.
- **0.50+** improbable salvo que el target esté trivialmente relacionado con features.
- **Negativa** = el modelo predice al revés. Bug seguro.

Esto contrasta con problemas de ML "clásicos" donde se esperan correlaciones de 0.8–0.95. Aquí el ratio señal/ruido es muchísimo más bajo.

---

## 6. Resumen

```
Para cada target (t0, t1):
   1. Recorta predicciones a [-6, 6]
   2. Calcula pesos = |y_true|
   3. Calcula correlación de Pearson PONDERADA por esos pesos
   4. Devuelve un número en [-1, 1]

Score final = promedio de los dos
```

Tres ideas a recordar:

1. **Es una correlación**, así que la *escala* de las predicciones no importa, solo la *forma*.
2. **Está sesgada hacia los movimientos grandes**, así que la loss también debería estarlo.
3. **El clipping a \[−6, 6\] indica que las predicciones no necesitan ir más allá de ese rango** — diseñarlas pensando en eso.
