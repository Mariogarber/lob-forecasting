# Temporal Convolutional Network (TCN)

Documento de referencia sobre la arquitectura **TCN** tal y como está implementada en
`src/models/sequence/tcn.py`. La idea es que después de leer esto puedas:

- entender de dónde viene la arquitectura y por qué encaja con LOB,
- saber qué hace cada bloque del modelo,
- razonar sobre el *receptive field* y los hiperparámetros,
- compararla con los otros modelos del repo (LSTM, DeepLOB, Transformer, Mamba-2).

---

## 1. De dónde viene

**Paper original**: Bai, Kolter & Koltun (2018) — *"An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"*, [arXiv:1803.01271](https://arxiv.org/abs/1803.01271).
Implementación de referencia: [locuslab/TCN](https://github.com/locuslab/TCN).

La motivación del paper era contundente: en la comunidad había una asunción extendida
de que "para secuencias, recurrente; para imágenes, convolucional". Los autores
construyeron una arquitectura **puramente convolucional** que respetara causalidad y
mostraron que **iguala o supera a LSTM/GRU** en una decena de benchmarks (Penn Treebank,
LAMBADA, MNIST permutado, polifónica MIDI…), con ventajas adicionales:

- entrenamiento totalmente paralelo (no hay dependencia temporal en el forward),
- gradientes estables (no hay vanishing por la longitud, solo por la profundidad),
- memoria predecible (no se acumulan estados ocultos a través del tiempo).

La arquitectura ganó tracción rápido porque era simple, fuerte y fácil de modificar.
Las dos piezas no triviales son:

1. **Causal convolutions** — el kernel solo lee el pasado.
2. **Dilated convolutions** — el receptive field crece exponencialmente con la
   profundidad sin disparar el número de parámetros.

---

## 2. La idea: convoluciones causales con dilatación

### 2.1 Conv estándar vs causal

Una `Conv1d` con `kernel_size=3` y `padding="same"` calcula en el paso `t`:

```
y[t] = w[-1]·x[t-1] + w[0]·x[t] + w[+1]·x[t+1]
```

El término `x[t+1]` lee **el futuro**. Para una tarea de regresión per-step donde cada
`t` es un punto de evaluación, esto es **leakage puro**: el modelo ve la respuesta antes
de tener que predecirla.

Una **causal conv** soluciona esto colocando todo el padding al **lado izquierdo** del
tensor temporal:

```
y[t] = w[0]·x[t-2] + w[1]·x[t-1] + w[2]·x[t]
```

En PyTorch eso se hace con `F.pad(x, (k-1, 0))` antes de la conv (la `Conv1d` se
construye con `padding=0`). Es exactamente el mismo truco que usamos en `deeplob.py`
con `CausalConv2d` pero ahora en 1-D.

### 2.2 Dilation

Una conv normal de kernel 3 ve **3 pasos contiguos**. Una conv **dilatada con factor `d`
de kernel 3** ve los pasos `t`, `t - d`, `t - 2d` — saltos de `d` en lugar de 1:

```
dilation = 1:   .  .  .  x  x  x   ← lee t-2, t-1, t
dilation = 2:   .  x  .  x  .  x   ← lee t-4, t-2, t
dilation = 4:   x  .  .  .  x  .  .  .  x   ← lee t-8, t-4, t
```

Apilando capas con dilatación 1, 2, 4, 8, 16… el *receptive field* crece **exponencialmente**
con la profundidad, mientras que el número de parámetros crece **linealmente**. Esa es la
magia del TCN: ver 100 pasos hacia atrás con 5-6 capas, no 100.

### 2.3 Receptive field exacto

Para `L` bloques (cada uno con dos convs causales de kernel `k` y dilatación `2^i`), el
*receptive field* es:

$$
\text{RF} = 1 + 2(k-1)\sum_{i=0}^{L-1} 2^i = 1 + 2(k-1)(2^L - 1)
$$

Algunos valores típicos (kernel 3):

| L (bloques) | RF (pasos) | Params (canales=64) |
|---|---|---|
| 3 |   29 | ~ 50 k |
| 4 |   61 | ~ 90 k |
| 5 |  125 | ~125 k |
| 6 |  253 | ~160 k |
| 7 |  509 | ~195 k |

Comparativa: para tener un RF de 125 con convs **sin** dilatación necesitarías 62 capas
seguidas. Con dilatación, 5.

---

## 3. Por qué encaja con LOB

El reto Wunder pide predecir `t0`, `t1` en cada paso de secuencias de 1000 steps. Los
requisitos implícitos son:

1. **Causalidad estricta** — el output en `t` no puede depender de inputs en `> t`.
2. **Receptive field largo pero no infinito** — la microestructura tiene memoria de
   decenas-centenas de pasos, no de los 1000 enteros (en HFT el régimen cambia rápido).
3. **Coste por paso bajo** — vamos a pedir predicciones en cientos de miles de filas.

El TCN cumple las tres limpiamente:

- la causalidad **es estructural** (el padding está construido así), no se puede romper
  por accidente como en DeepLOB original;
- el RF se controla con `num_layers` — bajamos a 3 capas para tareas con ventana corta,
  subimos a 7 si necesitamos memoria larga;
- el coste por paso es **constante**: no hay estado recurrente que recorrer, todo se
  computa en una pasada de convoluciones (en GPU).

Compara con DeepLOB: el paper original era para **clasificación de toda la ventana de
100 pasos** → una sola predicción → las convs se podían permitir mirar al futuro dentro
de esa ventana porque toda la ventana era pasado respecto al target. Al portarlo a
**regresión per-step** tuvimos que hacer causales todas las convs (ver `docs` adentro
de `src/models/sequence/deeplob.py`). El TCN es esa misma filosofía pero **construida
desde el principio** para per-step regression.

---

## 4. Componentes de la implementación

Ver `src/models/sequence/tcn.py` para el código completo.

### 4.1 `CausalConv1d` (lectura unitaria)

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))   # pad solo a la izquierda
        return self.conv(x)
```

Es lo más simple posible: padding asimétrico + conv sin padding. Si lo construyes con
`dilation=4` y `kernel_size=3`, el output `y[t]` depende de `x[t-8], x[t-4], x[t]`.

### 4.2 `TemporalBlock` (un bloque residual)

```
                  ┌─────────────────────────────────────┐
input  ──────────►│ CausalConv1d (dilation=d) → ReLU → Dropout  │
                  │ CausalConv1d (dilation=d) → ReLU → Dropout  │
                  └─────────────────────────────────────┘
                                    │
                                    ▼
                              ReLU(out + 1x1·input)
```

Dos convs causales seguidas, con dropout entre ellas, y una conexión residual. Si los
canales de entrada y salida coinciden, el residual es identidad; si no, se hace una
conv 1×1 para ajustar dimensiones. Esto es lo que permite apilar 6-8 bloques sin que
los gradientes mueran.

Las dos convs llevan **weight normalization** (`nn.utils.parametrizations.weight_norm`),
que reparametriza `W = (g/||v||) · v` y estabiliza el entrenamiento — es la receta
original del paper.

### 4.3 `TCN` (la red completa)

```python
class TCN(SequenceModel):
    def __init__(self, config):
        super().__init__(config)
        channels = config["channels"]
        num_layers = config["num_layers"]
        ...
        layers = []
        in_ch = n_features
        for i in range(num_layers):
            layers.append(TemporalBlock(in_ch, channels, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = RegressionHead(channels, n_targets, dropout=dropout)
```

La pila de bloques tiene dilatación 1, 2, 4, 8, 16 — una por bloque, exponencial. La
cabeza es un `Linear(channels → n_targets)` per-step.

Forward:

```
(B, T, F)  ──►  transpose ──►  (B, F, T)  ──►  TCN stack  ──►  (B, C, T)
            ──►  transpose ──►  (B, T, C)  ──►  Linear     ──►  (B, T, K)
```

Conv1d trabaja con `(batch, channels, time)`, así que hacen falta las transposes; nada
más profundo que eso.

---

## 5. Hiperparámetros que importan

| Param | Default | Efecto |
|---|---|---|
| `channels` | 64 | Capacidad por capa. Más → más expresivo y más caro. 32 / 64 / 128 es el rango habitual. |
| `num_layers` | 5 | Controla el *receptive field*. Doblar `L` dobla el RF (aprox.). |
| `kernel_size` | 3 | Ancho de la conv. Subir a 5 o 7 amplía el RF a costa de más parámetros. |
| `dropout` | 0.1 | Regularización entre las dos convs de cada bloque. |
| `window_size` | RF | Para el `step()` streaming: cuánto buffer mantener. Default = RF (no aporta más). |

La regla práctica:

1. Elige `kernel_size = 3` salvo que tengas una razón fuerte para subir.
2. Calcula el RF que quieres (≈ 100-200 para LOB) y deriva `num_layers` con la fórmula
   $\text{RF} = 1 + 2(k-1)(2^L - 1)$.
3. Empieza con `channels = 64`; sube a 96/128 si subajusta, baja a 32 si overfittea.

---

## 6. Comparativa con el resto de modelos del repo

| Modelo | Entrenamiento | Receptive field | Streaming | Params típicos |
|---|---|---|---|---|
| **GRU/LSTM** | secuencial (O(T)) | implícito, "efectivo" | O(1) por step | 300-500 k |
| **Transformer (causal)** | paralelo, O(T²) attention | exacto, ilimitado | O(T) por step (rolling buffer) | 600-900 k |
| **DeepLOB (causal)** | paralelo | corto (~30-50) | rolling buffer | 200-400 k |
| **TCN** | paralelo | **exacto y controlable** | rolling buffer | 100-200 k |
| **Mamba-2** | paralelo, O(T) SSD | implícito, lineal | O(1) por step | 500 k-1 M |

Puntos diferenciales del TCN:

- **El más barato en parámetros** para un RF dado.
- **El más rápido en epochs cortos** — converge bien con menos datos que el Transformer.
- **El más limpio en causalidad** — no hay mask, no hay hidden state, no hay convs no
  causales que se cuelen. Si pasa los tests, no hay leakage.
- **El más fácil de tunear** — solo 4 hiperparámetros importantes.

Lo que pierde frente a los otros:

- vs **LSTM**: la LSTM aprende mejor regímenes muy no estacionarios porque el estado
  oculto puede "memorizar" eventos lejanos sin ocupar todos los pasos del RF. Si tu
  serie tiene un evento que cambia el régimen y dura 500 steps, el TCN necesita un RF de
  500 para "verlo"; la LSTM solo necesita codificarlo en el estado.
- vs **Transformer**: el Transformer puede atender a cualquier paso del pasado con
  igual peso; el TCN tiene un sesgo inductivo fuerte hacia los pasos cercanos
  (resoluciones bajas requieren capas profundas).
- vs **Mamba-2**: Mamba modela la dinámica con un SSM continuo, lo que en algunas series
  capta patrones que la convolución discreta se deja.

---

## 7. Cómo usarlo en el pipeline

Como cualquier otro modelo:

```bash
# Entrenamiento standalone con el config YAML por defecto
python -m pipeline.cli train --config configs/models/tcn.yaml

# Listar modelos registrados (debe aparecer "tcn")
python -m pipeline.cli list

# En el notebook ya está integrado en la sección 7
python scripts/make_notebook.py     # regenera main.ipynb con la celda del TCN
```

YAML por defecto (`configs/models/tcn.yaml`):

```yaml
model:
  name: tcn
  params:
    channels: 64
    num_layers: 5
    kernel_size: 3
    dropout: 0.1
```

El modelo se persiste igual que los demás (`experiments/tcn/<run_id>/model.pt`) y el
packager `solution.zip` ya sabe empacarlo (`src/submission/packager.py` tiene el
fichero `models/sequence/tcn.py` en `_RUNTIME_FILES`).

---

## 8. Tests que lo validan

- `tests/models/test_registry.py::test_sequence_forward_shape[tcn]` — shape del forward.
- `tests/models/test_registry.py::test_tcn_streaming_matches_forward` — paridad
  batched ↔ streaming.
- `tests/models/test_registry.py::test_tcn_causality` — test independiente de leakage:
  perturba `x[t+1:]` y verifica que `y[:t+1]` no cambia.

Si añades algún hack al modelo, estos tres tests son la red de seguridad.

---

## 9. Referencias

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic
  Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.
- van den Oord et al. (2016). *WaveNet: A Generative Model for Raw Audio*. arXiv:1609.03499
  (la inspiración directa: WaveNet ya usaba dilated causal convs).
- Repositorio de referencia: [locuslab/TCN](https://github.com/locuslab/TCN).
