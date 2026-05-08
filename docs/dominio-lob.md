# Dominio: Limit Order Book (LOB) y forecasting de movimientos de precio

Documento de referencia con los conceptos del dominio necesarios para entender el dataset del reto **Wunder Predictorium** y por qué la tarea está planteada como está.

---

## 1. ¿Qué es un Limit Order Book (LOB)?

En un mercado financiero electrónico, no compras/vendes directamente con otra persona: lanzas **órdenes** a un libro centralizado. Hay dos tipos básicos:

- **Limit order**: "compro hasta 100 acciones a 50€ o menos" (o "vendo a 51€ o más"). Se queda **pendiente** en el libro hasta que alguien la cruce.
- **Market order**: "compro 100 acciones ya, al precio que sea". Se ejecuta inmediatamente contra las órdenes pendientes mejores del lado contrario.

El **LOB es la foto de todas las limit orders pendientes** en cada instante. Tiene dos lados:

| Lado    | Significado                          | Ordenado por |
|---------|--------------------------------------|--------------|
| **Bid** | Gente que quiere **comprar**         | Precio descendente (mejor = más alto) |
| **Ask** | Gente que quiere **vender**          | Precio ascendente (mejor = más bajo) |

Visualmente:

```
              ASK (ofertas de venta)
   nivel 5:  51.05  →  v=200
   nivel 4:  51.04  →  v=150
   nivel 3:  51.03  →  v=100
   nivel 2:  51.02  →  v=80
   nivel 1:  51.01  →  v=50      ← best ask
   ─────────── spread ───────────
   nivel 1:  51.00  →  v=70      ← best bid
   nivel 2:  50.99  →  v=120
   nivel 3:  50.98  →  v=200
   ...
              BID (ofertas de compra)
```

### Conceptos clave

- **Best bid / best ask**: las mejores órdenes de cada lado (las que se cruzarían primero).
- **Spread**: `best_ask - best_bid`. Cuanto más estrecho, más líquido el mercado.
- **Mid-price**: `(best_bid + best_ask) / 2`. Suele usarse como "precio actual" porque no hay un único precio "real" en cada instante.
- **Niveles (levels)**: el libro no es solo la mejor oferta, hay órdenes a precios peores esperando. La forma del libro (cuánto volumen hay a cada nivel) **es información predictiva**: si hay una pared enorme de bids en el nivel 3, el precio probablemente no caerá por debajo de ahí.

---

## 2. Niveles del libro en detalle

### ¿Qué es exactamente un "nivel"?

Un nivel es **un precio concreto del libro donde hay al menos una orden pendiente, agregando todo el volumen que existe a ese precio**.

No es una "categoría de calidad". Es literalmente un *bucket* por precio. Si en el lado bid hay tres personas con órdenes a 50.99€ (de 30, 40 y 50 acciones), eso es **un solo nivel**: precio = 50.99, volumen = 120.

```
   ASK
   ─────────────────────────
   precio   volumen   nivel
   51.05      200       6
   51.04      150       5
   51.03      100       4    ← niveles de ask
   51.02       80       3       (creciendo en precio)
   51.01      120       2
   51.00       50       1    ← best ask
   ─── spread ──────────────
   50.99      120       1    ← best bid
   50.98       70       2
   50.97      200       3    ← niveles de bid
   50.95       80       4       (decreciendo en precio)
   50.94      150       5
   50.93      300       6
   BID
```

Tres ideas clave:

- **Un nivel = un precio único** del lado del libro. Múltiples órdenes al mismo precio se *agregan*.
- Los niveles se numeran **por cercanía al spread**, no por orden de llegada. El nivel 1 es siempre el "mejor" (el más cercano al cruce).
- El nivel 1 bid ≠ nivel 1 ask. Tienen precios distintos (de hecho su diferencia es el spread).

### ¿Quién asigna cada oferta a cada nivel?

**Nadie. Es automático y emerge del precio de la orden.** Lo hace el motor de cruce (matching engine) del exchange. La regla es:

1. Llega una orden limit a precio X.
2. Si ya existe el nivel a precio X en su lado, su volumen **se suma** al volumen de ese nivel. Dentro del nivel se mantiene un orden FIFO (price-time priority): la primera orden que llegó al precio X es la primera que se ejecuta cuando alguien la cruza.
3. Si no existe el nivel a precio X, se **crea un nivel nuevo** en su sitio según el orden de precios.

Y los niveles **se renumeran solos** todo el rato, porque:

- Si el best ask (nivel 1, precio 51.00) se vacía (alguien lo compra entero), entonces lo que era el nivel 2 (51.01) **pasa a ser el nuevo nivel 1**. Todos los demás se desplazan también.
- Si llega una orden bid a 51.005 (entre el best bid y el best ask), se "mete" en el spread y se convierte en el nuevo nivel 1 bid, empujando al resto un puesto.
- Si cancelan todas las órdenes a un nivel, ese nivel desaparece y los más profundos se reordenan.

> **Implicación práctica para el modelo:** las features `p0..p11` no son una secuencia estable de "estos seis precios" — son una representación dinámica que cambia de significado cuando el libro se mueve. El "nivel 3 bid" en `step=10` y el "nivel 3 bid" en `step=11` no son la misma orden ni necesariamente el mismo precio. Son siempre "el tercer mejor precio bid en ese instante", lo que sea. El modelo tiene que aprender a interpretar la **forma del libro**, no precios concretos.

### ¿Por qué 6 niveles a cada lado?

Es **una decisión de diseño del dataset**, no algo intrínseco al mercado. El libro real puede tener cientos de niveles, pero te dan solo los **6 más cercanos al spread (top-6)** porque:

- La gran mayoría del valor predictivo está cerca del spread. Una orden a 10€ por encima del best ask es prácticamente irrelevante a corto plazo.
- Mantienes la dimensionalidad acotada (6 × 2 lados × 2 métricas = 24 features de LOB).
- Es el formato estándar en datasets académicos de LOB (típicamente top-5 o top-10).

A esto se le llama **L2 data con depth = 6**. Niveles de granularidad típicos:

- **L1**: solo el best bid y best ask.
- **L2**: top-K niveles agregados (lo del dataset).
- **L3**: cada orden individual identificada (lo más detallado, raro de obtener).

---

## 3. Trades vs estados del LOB

Son cosas distintas y en el dataset se distinguen por columnas:

- **Estado del LOB** (`p0..p11`, `v0..v11`): es la *foto* del libro en un instante. Son órdenes **pendientes**, no ejecuciones.
- **Trades** (`dp0..dp3`, `dv0..dv3`): son las **ejecuciones reales** que han ocurrido recientemente. Cuando alguien lanza una market order, "se come" los niveles del libro y eso genera un trade.

Ambas señales son complementarias: el libro te dice las *intenciones*, los trades te dicen el *flujo real* de dinero/presión.

### ¿Por qué solo 4 features para trades (y no 6)?

Aquí cambia la lógica: **los trades no son niveles**. Son eventos.

Un nivel es un *estado* (qué órdenes hay ahora mismo). Un trade es una *transacción que ya ocurrió*: alguien lanzó una market order que cruzó el libro y se ejecutó contra una o más órdenes pendientes. Entonces no tiene sentido hablar de "niveles de trade" — los trades ocurren a un precio puntual y se acabó.

Las 4 features `dp0..dp3` (precios) y `dv0..dv3` (volúmenes) probablemente representan, según pistas del README ("derived from **both bid and ask trades**"):

**Hipótesis más plausible**: separación por **lado agresor**. Cuando ocurre un trade, se etiqueta:

- **Buy-side trade** ("aggressor compró"): alguien lanzó una market buy que se comió ofertas del ask. Es presión compradora.
- **Sell-side trade** ("aggressor vendió"): alguien lanzó una market sell que se comió ofertas del bid. Es presión vendedora.

Con 4 features, lo natural sería algo como:

- `dp0, dv0`: trade más reciente del lado bid (o agregado de ventas agresivas).
- `dp1, dv1`: trade más reciente del lado ask (o agregado de compras agresivas).
- `dp2, dv2, dp3, dv3`: el segundo trade más reciente de cada lado, o un agregado a ventana mayor.

No podemos saberlo con certeza porque está anonimizado. Pero la idea importante es:

- **Pocos features de trade no significa "menos información"**, significa que la información de trades es naturalmente más compacta. No hay "depth" que representar — solo *qué se ejecutó y de qué lado*.
- La señal predictiva más clásica derivada de trades es el **trade sign / order flow**: ¿está dominando la presión compradora o vendedora? Eso suele anticipar movimientos de mid-price a corto.

> **Cosas a verificar empíricamente en `exploration/`**:
> - ¿Los `dv` son siempre positivos o tienen signo? Si tienen signo, igual ya codifican el lado agresor.
> - ¿Hay timesteps donde `dp0..dp3` son todos iguales/cero? Si los trades no ocurren en cada step, debe haber un esquema de "padding" o "última observación".

---

## 4. Mapeo al dataset concreto

| Columnas      | Contenido                                | Interpretación |
|---------------|------------------------------------------|----------------|
| `p0..p5`      | Precios del lado **bid**, 6 niveles      | "A qué precios está la gente esperando comprar" |
| `p6..p11`     | Precios del lado **ask**, 6 niveles      | "A qué precios está la gente esperando vender" |
| `v0..v5`      | Volúmenes bid (asociados a `p0..p5`)     | Cuánto hay disponible en cada nivel bid |
| `v6..v11`     | Volúmenes ask (asociados a `p6..p11`)    | Cuánto hay disponible en cada nivel ask |
| `dp0..dp3`    | Precios de trades recientes              | A qué precio se ha ejecutado |
| `dv0..dv3`    | Volúmenes de trades                      | Cuánto se ha ejecutado |

Estructura adicional del dataset:

- `seq_ix`: ID de la secuencia. Cuando cambia, empiezas una secuencia nueva e independiente.
- `step_in_seq`: paso dentro de la secuencia (0–999).
- `need_prediction`: bool, `True` si hay que predecir en ese step.
- `t0`, `t1`: targets (anonimizados).

---

## 5. Anonimización

Los features `p`, `v`, `dp`, `dv` y los targets `t0`, `t1` están **anonimizados**. No son euros ni centavos. Probablemente son normalizaciones / transformaciones (logs, deltas relativos al mid, etc.) para que no se pueda identificar el activo.

> **Implicación**: no se debe razonar en términos absolutos ("el precio es 51"), sino en términos relativos y de patrones.

---

## 6. Qué se intenta predecir

`t0` y `t1` son "indicadores futuros del movimiento de precio". El reto no dice exactamente qué son, pero por la métrica (Weighted Pearson, clipping a [-6, 6], peso por amplitud) y el dominio típico del LOB forecasting, casi seguro son algo del estilo:

- Retornos futuros del mid-price a distintos horizontes (p. ej. `t0` = retorno en N ticks, `t1` = retorno en M ticks).
- O señales derivadas tipo "dirección del próximo movimiento significativo".

Que la métrica pondere por amplitud da una pista útil: **importa más acertar los movimientos grandes que los pequeños**. Acertar el ruido alrededor de cero apenas suma puntos.

---

## 7. Por qué este problema es duro

Cosas que no son obvias si vienes de ML "convencional":

1. **No estacionariedad**: la dinámica del mercado cambia constantemente. Un patrón que funcionaba ayer puede dejar de funcionar. Por eso insisten en "validation que respete splits temporales".
2. **Ratio señal/ruido bajísimo**: la mayoría de los movimientos a corto plazo son ruido. Pearson correlations de 0.05–0.15 ya pueden ser resultados decentes en HFT.
3. **Microestructura > macro**: a esta escala (ticks, no días), lo que mueve el precio son cosas como "¿hay un desequilibrio bid/ask?", "¿se ha vaciado un nivel?", no noticias o fundamentales.
4. **Cada secuencia es independiente**: el modelo debe **resetear estado** entre `seq_ix`s. Un GRU/LSTM/Transformer no puede arrastrar contexto de una secuencia a la siguiente porque no son contiguas en el tiempo.
5. **Warm-up**: los primeros 99 pasos no se evalúan, son para que el modelo "caliente" su estado interno. Esto es típico de modelos recurrentes — necesitan algunos steps para construir un buen hidden state.

---

## 8. Features clásicas que se suelen extraer del LOB

Útiles a la hora de ingeniar features adicionales:

- **Order Flow Imbalance (OFI)**: desequilibrio entre volumen bid y ask.
- **Mid-price returns**: cambio del mid-price.
- **Spread** y su evolución.
- **Depth**: volumen acumulado en los primeros K niveles.
- **Trade sign / aggressor side**: ¿el último trade golpeó al ask (comprador agresivo) o al bid (vendedor agresivo)?
- **Volatility** sobre ventanas cortas.

---

## 9. Glosario rápido

| Término | Definición breve |
|---------|------------------|
| **LOB (Limit Order Book)** | Conjunto de todas las órdenes limit pendientes en un mercado, agregadas por precio. |
| **Limit order** | Orden a un precio específico que espera en el libro hasta ser cruzada. |
| **Market order** | Orden inmediata al mejor precio disponible; consume liquidez del libro. |
| **Bid** | Lado del libro donde hay órdenes de compra pendientes. |
| **Ask (u Offer)** | Lado del libro donde hay órdenes de venta pendientes. |
| **Best bid / best ask** | Mejor precio disponible en cada lado (más alto en bid, más bajo en ask). |
| **Spread** | `best_ask - best_bid`. Indicador de liquidez. |
| **Mid-price** | `(best_bid + best_ask) / 2`. "Precio actual" sintético. |
| **Nivel (level)** | Precio único del libro, con todo el volumen pendiente a ese precio agregado. |
| **Depth** | Cuántos niveles del libro se reportan o se observan. |
| **L1 / L2 / L3** | Granularidades de datos LOB (best-only / top-K agregados / orden por orden). |
| **Trade** | Ejecución real de una transacción cuando una orden cruza el libro. |
| **Aggressor side** | Lado que inició la ejecución (comprador o vendedor agresivo). |
| **Order Flow Imbalance (OFI)** | Métrica del desequilibrio neto entre actividad bid y ask. |
| **Tick** | Unidad mínima de variación de precio en un mercado. También se usa como sinónimo de "evento" o "step". |
| **Microestructura** | Conjunto de fenómenos a la escala del libro y los trades individuales. |
