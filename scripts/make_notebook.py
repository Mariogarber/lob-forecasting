#!/usr/bin/env python3
"""Generate the `main.ipynb` deliverable.

The notebook is the entrypoint of the project handed in for grading:
introduction, EDA, feature engineering, three rounds of model
iterations, final comparison, and submission packaging. We keep the
source in this script so the notebook can be regenerated cleanly if any
section needs an edit.

Usage:
    python scripts/make_notebook.py        # writes main.ipynb at the repo root

Design notes:

* All training cells respect a top-level ``QUICK_MODE`` flag. The default
  (True) trains on a subset of sequences with a short epoch budget so
  the notebook completes in ~10-20 min. Setting it to False runs the
  full training (~hours), useful for the final submission run.
* Every model run produces a row in a shared ``results`` list that
  drives the comparison table at the end.
* Cells avoid duplicating logic from ``src/`` — we always call the
  pipeline helpers (Trainer, Evaluator, packagers) so the notebook
  stays a thin orchestration layer.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "main.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    # -------- 0 · Header --------
    cells.append(md("""
        # Wunder Predictorium — Solución completa

        > **Autor:** Eder Tarifa &nbsp;·&nbsp; **Reto:** [Wunder Predictorium](https://wundernn.io/predictorium)
        > **Métrica oficial:** *Weighted Pearson Correlation* (ponderada por `|y_true|`, predicciones recortadas a `[-6, 6]`).

        Este cuaderno reúne **todas las iteraciones** del trabajo realizado para el reto:

        1. **Análisis exploratorio de los datos** (LOB + trades).
        2. **Feature engineering** sobre microestructura y dinámica del libro.
        3. **Iteración 1 — Baselines clásicos**: Linear / Ridge / Random Forest / LightGBM.
        4. **Iteración 2 — Modelos secuenciales DL básicos**: GRU / LSTM / Transformer causal.
        5. **Iteración 3 — Modelos específicos / SOTA**: DeepLOB (Zhang 2019), TCN (Bai 2018) y Mamba-2 (Dao 2024).
        6. **Comparativa final y selección del mejor modelo**.

        El cuaderno se apoya en el pipeline modular del paquete `src/` —cada modelo se entrena con
        las mismas utilidades para que la comparativa sea justa y reproducible. **Cada modelo se
        persiste en `experiments/<model>/<run_id>/`** (weights + métricas + plots + predicciones)
        y la tabla acumulada se vuelca a `experiments/notebook_results.csv` tras cada iteración,
        de modo que la información sobrevive al reinicio del kernel.
    """))

    # -------- 1 · Setup --------
    cells.append(md("""
        ## 1 · Setup

        Cargamos dependencias, fijamos semillas y un *toggle* `QUICK_MODE` que controla si entrenamos
        en modo rápido (subset + pocos epochs, ~10–20 min) o completo (varias horas).
    """))

    cells.append(code("""
        %load_ext autoreload
        %autoreload 2
        %matplotlib inline

        import os
        import sys
        import gc
        import json
        import time
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        try:
            import psutil
            _PROC = psutil.Process()
            def mem_gb() -> float:
                return _PROC.memory_info().rss / 1024**3
        except ImportError:
            def mem_gb() -> float:
                return float("nan")

        # Silenciar el ruido de mamba-ssm (FutureWarnings de torch.cuda.amp).
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

        # Asegurar que src/ está en el path cuando el notebook se ejecuta fuera de pytest.
        ROOT = Path.cwd()
        SRC = ROOT / "src"
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))

        sns.set_theme(context="notebook", style="whitegrid")
        np.random.seed(0)
        torch.manual_seed(0)

        print(f"PyTorch:   {torch.__version__}")
        print(f"CUDA:      {torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  ({torch.cuda.get_device_name(0)})")
        else:
            print()
        print(f"NumPy:     {np.__version__}")
        print(f"Pandas:    {pd.__version__}")
    """))

    cells.append(code("""
        # ⚙️  Toggle global. Cambia a True para iteración rápida.
        QUICK_MODE = False

        # Subconjunto de secuencias usadas en modo rápido. Pon None para usar todas.
        QUICK_N_TRAIN_SEQS = 800
        QUICK_N_VAL_SEQS = 250

        # Epochs por modelo DL en modo rápido. En modo completo se usan 20 epochs.
        QUICK_EPOCHS = 4

        # RandomForest escala mal (no usa sample_weight para early-stop). En LOB/HFT
        # rara vez gana a LightGBM; ponlo a True si te crashea la RAM o tienes prisa.
        SKIP_RF = False

        # Mamba-2 con d_model=256 en GPU consumer (RTX 30-series Laptop) es ~30x más
        # lento que el resto de DL en full-mode (kernels CUDA nativos saturan el SM).
        # Activa este flag si solo quieres validar la arquitectura del resto.
        SKIP_MAMBA = False

        # TLOB (Garcia et al. 2024) — SOTA específico de LOB. Es el modelo nuevo
        # que añadimos a esta iteración. Activa para saltarlo si solo quieres
        # validar el resto.
        SKIP_TLOB = False

        # Salta toda la sección de modelos clásicos (linear/ridge/RF/lightgbm) y va
        # directo a los DL. Útil cuando la construcción tabular en CPU no cabe en RAM.
        SKIP_CLASSICAL = False

        # Rolling features (mid_rmean_{5,20}, mid_rstd_{5,20}, vol_rmean_{5,20}) son
        # costosas en memoria (groupby.transform). Las features 'spread', 'imbalance',
        # 'momentum'... ya capturan la dinámica básica. Déjalo a False si tu RAM va justa.
        WITH_ROLLING = False

        # Procesar las features tabulares por chunks de N secuencias para acotar
        # el peak de RAM. 200 es un buen compromiso (≈200·1000=200K filas por chunk).
        TABULAR_CHUNK_SEQS = 200

        SEED = 0
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"QUICK_MODE={QUICK_MODE}  DEVICE={DEVICE}  SKIP_CLASSICAL={SKIP_CLASSICAL}  SKIP_RF={SKIP_RF}  SKIP_MAMBA={SKIP_MAMBA}  SKIP_TLOB={SKIP_TLOB}  WITH_ROLLING={WITH_ROLLING}")
        print(f"RAM inicial: {mem_gb():.2f} GB")
    """))

    cells.append(code("""
        # API del pipeline.
        from data import (
            load_train_valid, split_by_seq_ix, FeatureScaler,
            FEATURE_COLS, TARGET_COLS, SEQ_LEN, WARMUP_STEPS, N_FEATURES, N_TARGETS,
            sequences_from_dataframe, SequenceDataset,
            build_tabular_features, make_tabular_dataset,
        )
        from data.constants import (
            BID_PRICES, ASK_PRICES, BID_VOLUMES, ASK_VOLUMES,
            TRADE_PRICES, TRADE_VOLUMES,
        )
        from feature_engineering import add_engineered_features, ENGINEERED_FEATURES
        from metrics import (
            summary, weighted_pearson_per_target, per_sequence_weighted_pearson,
        )
        from losses import build_loss
        from models import get_model_class, list_models, MODEL_REGISTRY
        from training import TrainerConfig, train_classical_model, train_sequence_model
        from training.classical_trainer import ClassicalTrainerConfig
        from evaluation import Evaluator, write_report, make_run_id

        print("modelos registrados:")
        for kind in ("classical", "sequence"):
            print(f"  [{kind}] {', '.join(list_models(kind))}")
    """))

    # -------- 2 · EDA --------
    cells.append(md("""
        ## 2 · Análisis exploratorio de los datos (EDA)

        ### Origen de los datos

        El reto provee dos parquet ya pre-procesados en `competition_package/datasets/`:

        - `train.parquet` — 10 721 secuencias.
        - `valid.parquet` — 1 444 secuencias.

        Cada secuencia tiene **1000 pasos** independientes; los primeros 99 son *warm-up* (no se evalúan).
        Las columnas son 32 features anonimizadas del LOB + trades, más las dos targets (`t0`, `t1`).
    """))

    cells.append(code("""
        # Carga inteligente: si QUICK_MODE, leemos del parquet SOLO las secuencias
        # que vamos a usar (filter-pushdown de pyarrow). Saltamos ~92% de la RAM
        # vs cargar las 10 721 secuencias completas y luego subsettar.
        from data import load_subset

        if QUICK_MODE:
            print(f"Cargando subset directamente del parquet ({QUICK_N_TRAIN_SEQS} train + {QUICK_N_VAL_SEQS} val seqs)...")
            train_df = load_subset(
                "competition_package/datasets/train.parquet",
                n_seqs=QUICK_N_TRAIN_SEQS, seed=SEED,
            )
            valid_df = load_subset(
                "competition_package/datasets/valid.parquet",
                n_seqs=QUICK_N_VAL_SEQS, seed=SEED,
            )
        else:
            print("Cargando los parquets completos...")
            train_df, valid_df = load_train_valid()

        print(f"train: {len(train_df):,} filas · {train_df['seq_ix'].nunique():,} secuencias")
        print(f"valid: {len(valid_df):,} filas · {valid_df['seq_ix'].nunique():,} secuencias")
        print(f"RAM tras carga: {mem_gb():.2f} GB")
        train_df.head()
    """))

    cells.append(code("""
        # Tamaño en memoria y dtypes — confirmamos que load_dataset hace el downcast a float32.
        train_df.info(memory_usage="deep")
    """))

    cells.append(md("""
        ### Distribución de las targets

        Las dos targets son indicadores anonimizados de movimientos futuros. La métrica pondera
        por `|y_true|`, así que la mayor parte del *score* viene de los pasos con target grande.
    """))

    cells.append(code("""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for ax, tgt in zip(axes, TARGET_COLS):
            sns.histplot(train_df[tgt], bins=120, kde=False, ax=ax, color="steelblue")
            ax.set_xlim(-6.5, 6.5)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{tgt}  ·  mean={train_df[tgt].mean():+.3f}  std={train_df[tgt].std():.3f}")
            ax.set_xlabel(tgt)
        fig.suptitle("Distribución de las targets en train (recortado a [-6.5, 6.5])")
        fig.tight_layout()
        plt.show()

        # Cuánta proporción de pasos puntúa.
        pct_scored = train_df["need_prediction"].mean() * 100
        print(f"% de filas evaluadas (need_prediction=True): {pct_scored:.1f}%")
        print(f"correlación entre t0 y t1 (Pearson): {train_df[['t0','t1']].corr().iloc[0,1]:+.3f}")
    """))

    cells.append(md("""
        ### Ejemplo: una secuencia completa

        Visualizar una sola secuencia ayuda a entender el régimen temporal de los datos.
        El mid-price es `(p0 + p6) / 2` (mejor bid + mejor ask) y se construye on-the-fly aquí.
    """))

    cells.append(code("""
        sample_seq_id = int(train_df["seq_ix"].iloc[0])
        seq = train_df[train_df["seq_ix"] == sample_seq_id]
        mid = (seq["p0"] + seq["p6"]) / 2
        spread = seq["p6"] - seq["p0"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(seq["step_in_seq"], mid, color="tab:blue")
        axes[0].axvspan(0, WARMUP_STEPS, alpha=0.15, color="gray", label="warm-up")
        axes[0].set_ylabel("mid-price")
        axes[0].set_title(f"seq_ix={sample_seq_id}")
        axes[0].legend(loc="upper left")

        axes[1].plot(seq["step_in_seq"], spread, color="tab:orange")
        axes[1].axvspan(0, WARMUP_STEPS, alpha=0.15, color="gray")
        axes[1].set_ylabel("spread (p6 - p0)")

        axes[2].plot(seq["step_in_seq"], seq["t0"], label="t0", alpha=0.75)
        axes[2].plot(seq["step_in_seq"], seq["t1"], label="t1", alpha=0.75)
        axes[2].axvspan(0, WARMUP_STEPS, alpha=0.15, color="gray")
        axes[2].axhline(0, color="black", linewidth=0.7, linestyle="--")
        axes[2].set_ylabel("targets")
        axes[2].set_xlabel("step in seq")
        axes[2].legend(loc="upper right")
        fig.suptitle("Anatomía de una secuencia: mid-price, spread y targets")
        fig.tight_layout()
        plt.show()
    """))

    cells.append(md("""
        ### Estadísticos de los features

        Resumimos en un solo grid las primeras filas + un boxplot por grupo. Los features están
        anonimizados pero conservan la estructura típica de LOB (precios escalonados, volúmenes
        positivos con cola larga).
    """))

    cells.append(code("""
        groups = {
            "bid prices (p0..p5)": BID_PRICES,
            "ask prices (p6..p11)": ASK_PRICES,
            "bid volumes (v0..v5)": BID_VOLUMES,
            "ask volumes (v6..v11)": ASK_VOLUMES,
            "trade prices (dp0..dp3)": TRADE_PRICES,
            "trade volumes (dv0..dv3)": TRADE_VOLUMES,
        }

        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for ax, (title, cols) in zip(axes.flat, groups.items()):
            data = train_df[cols].melt(value_name="value", var_name="feature")
            sns.boxplot(data=data, x="feature", y="value", ax=ax, fliersize=0.5, color="steelblue")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            ax.set_xlabel(None)
        fig.suptitle("Distribución de los 32 features brutos (cajas y bigotes)")
        fig.tight_layout()
        plt.show()
    """))

    cells.append(md("""
        ### Correlación entre features y targets

        Mapa de calor: cada feature contra `t0` y `t1`. Esperamos correlaciones bajas (SNR
        microestructural típico) pero algunos features dominan: spread, desequilibrio de
        volumen, momentum del mid…
    """))

    cells.append(code("""
        # Calcular correlación solo sobre filas evaluadas (need_prediction=True) — coincide con la métrica.
        scored = train_df[train_df["need_prediction"]]
        corr_block = scored[FEATURE_COLS + TARGET_COLS].corr().loc[FEATURE_COLS, TARGET_COLS]
        plt.figure(figsize=(5, 8))
        sns.heatmap(corr_block, cmap="coolwarm", center=0, annot=True, fmt="+.2f",
                    cbar_kws={"label": "Pearson"})
        plt.title("Correlación lineal feature ↔ target (solo filas evaluadas)")
        plt.tight_layout()
        plt.show()
    """))

    # -------- 3 · Feature engineering --------
    cells.append(md("""
        ## 3 · Feature engineering

        El paquete `feature_engineering` añade 12 features derivadas:

        * **Microestructura**: `spread`, `mid_price`, `bid_ask_volume_imbalance`, 6 niveles de
          `depth_imbalance_level_k`.
        * **Dinámica**: `price_momentum`, `volume_momentum`, `bid_ask_momentum_diff`.

        El pipeline (`data.tabular.build_tabular_features`) adicionalmente añade *rolling stats*
        multi-ventana (5 y 20 pasos) sobre el mid-price y el volumen total. Todas las
        operaciones respetan los límites de secuencia vía `groupby('seq_ix')`.
    """))

    cells.append(code("""
        # Mostrar las features ingenieradas en una muestra pequeña.
        sample_ids = train_df["seq_ix"].unique()[:30]
        sample = train_df[train_df["seq_ix"].isin(sample_ids)].copy()
        sample_eng = add_engineered_features(sample)

        new_cols = [c for c in sample_eng.columns if c not in sample.columns]
        print(f"features añadidas: {len(new_cols)}")
        sample_eng[new_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    """))

    cells.append(code("""
        # Correlación de las features ingenieradas con los targets (mejor proxy de utilidad).
        scored_sample = sample_eng[sample_eng["need_prediction"]]
        corr_eng = scored_sample[new_cols + TARGET_COLS].corr().loc[new_cols, TARGET_COLS]
        plt.figure(figsize=(5, 5))
        sns.heatmap(corr_eng, cmap="coolwarm", center=0, annot=True, fmt="+.2f")
        plt.title("Correlación feature ingenierada ↔ target (muestra)")
        plt.tight_layout()
        plt.show()
    """))

    # -------- 4 · Modeling strategy --------
    cells.append(md("""
        ## 4 · Estrategia de modelado

        Probaremos 10 modelos en tres rondas. Las claves son:

        * **Splits por `seq_ix`** — usamos el `valid.parquet` oficial; jamás partimos filas dentro de una secuencia.
        * **Loss alineado con la métrica** — `Weighted Pearson Loss` para los modelos DL (con `clamp(-6, 6)` interno) y `sample_weight = |y_true|` para los clásicos.
        * **Evaluación** — `weighted_pearson` por target + media (idéntico al scorer oficial).
        * **Reproducibilidad** — semilla 0 en todos los `np.random` / `torch.manual_seed` y, donde aplique, en los estimadores sklearn.

        | # | Familia | Modelo | Notas |
        |---|---------|--------|-------|
        | 1 | Clásico | `linear` | Regresión lineal con `sample_weight=|y|`. |
        | 2 | Clásico | `ridge` | L2-regularizado. |
        | 3 | Clásico | `random_forest` | Sin escalado. |
        | 4 | Clásico | `lightgbm` | Early-stopping sobre weighted Pearson. |
        | 5 | DL | `gru` | GRU 2-layer, streamable. |
        | 6 | DL | `lstm` | LSTM 2-layer, streamable. |
        | 7 | DL | `transformer` | Pre-LN encoder + causal mask. |
        | 8 | DL/SOTA | `deeplob` | Zhang 2019 (CNN+LSTM), adaptado a regresión. |
        | 9 | DL/SOTA | `tcn` | Temporal Convolutional Network (Bai 2018), dilated causal convs. |
        | 10 | DL/SOTA | `mamba2` | Selective SSM (Mamba-2 SSD, PyTorch puro, scan fp32, lineal en T). |
        | 11 | DL/SOTA | `tlob` | **Dual-axis Transformer** (Garcia 2024). Attention sobre features + attention causal sobre tiempo. Independente del orden de columnas. |
    """))

    cells.append(code("""
        # Ya no hace falta volver a sub-muestrear — load_subset lo hizo en la carga.
        # Renombramos simplemente para mantener la semántica de "datos efectivos".
        train_use = train_df
        val_use   = valid_df
        del train_df, valid_df
        gc.collect()

        print(f"train_use: {train_use['seq_ix'].nunique()} secuencias · {len(train_use):,} filas")
        print(f"val_use:   {val_use['seq_ix'].nunique()} secuencias · {len(val_use):,} filas")
        print(f"RAM tras subset: {mem_gb():.2f} GB")

        # Scaler — se usa para los modelos DL (no para clásicos).
        scaler = FeatureScaler(mode="standard").fit(train_use, FEATURE_COLS)

        # Resultados acumulados — se rellenan en las siguientes secciones.
        results: list[dict] = []

        # Persistencia entre celdas: cada iteración añade su fila aquí.
        EXPERIMENTS_ROOT = Path("experiments")
        EXPERIMENTS_ROOT.mkdir(exist_ok=True)
        NOTEBOOK_RESULTS_CSV = EXPERIMENTS_ROOT / "notebook_results.csv"

        def _flush_results():
            \"\"\"Vuelca la tabla `results` a CSV tras cada iteración.\"\"\"
            if results:
                pd.DataFrame(results).to_csv(NOTEBOOK_RESULTS_CSV, index=False)
        ;
    """))

    # -------- 5 · Classical baselines --------
    cells.append(md("""
        ## 5 · Iteración 1 — Baselines clásicos

        Estos modelos usan los 32 features brutos **+ las 12 ingenieradas + rolling stats (5, 20)**.
        Cada modelo se ajusta por target (one-vs-rest) con `sample_weight = |y_true|`. Tras el
        entrenamiento, cada run se persiste en `experiments/<model>/<run_id>/` con `model.joblib`,
        `config.yaml`, `metrics.json`, `predictions.parquet`, `scatter.png` y `per_seq_corr.png`.

        **Optimización de RAM**: construimos las features tabulares **una sola vez** y las
        reutilizamos en los 4 modelos (antes se recomputaban en cada `run_classical`, ×4).
    """))

    cells.append(code("""
        # === Construir features tabulares por CHUNKS (acotamos el peak de RAM) ===
        # En vez de transformar el DataFrame entero (que copia ~3 GB transitorios),
        # procesamos `TABULAR_CHUNK_SEQS` secuencias a la vez y concatenamos los ndarrays
        # finales. Cada chunk libera sus copias intermedias en gc.collect().
        from metrics.pearson import _weighted_pearson_scalar  # type: ignore
        from evaluation.evaluator import EvalResult
        from dataclasses import replace as _dc_replace

        ROLLING_WINDOWS = (5, 20) if WITH_ROLLING else ()

        if SKIP_CLASSICAL:
            print("⊘ Saltando construcción tabular (SKIP_CLASSICAL=True).")
            train_data = val_data = feature_cols = None

        def _build_tabular_arrays_chunked(df, *, chunk_seqs, with_engineered, rolling_windows):
            unique_seqs = df["seq_ix"].unique()
            n_chunks = max(1, len(unique_seqs) // chunk_seqs + bool(len(unique_seqs) % chunk_seqs))
            chunks_X, chunks_y, chunks_w, chunks_sid, chunks_step = [], [], [], [], []
            feature_cols_final = None
            for i, sub_ids in enumerate(np.array_split(unique_seqs, n_chunks)):
                chunk_df = df[df["seq_ix"].isin(sub_ids)]
                tab, cols = build_tabular_features(
                    chunk_df, with_engineered=with_engineered, rolling_windows=rolling_windows,
                )
                ds = make_tabular_dataset(tab, cols, only_scored_rows=True, drop_warmup=True)
                feature_cols_final = cols
                chunks_X.append(ds.X);    chunks_y.append(ds.y)
                chunks_w.append(ds.weights); chunks_sid.append(ds.seq_ids); chunks_step.append(ds.step_ids)
                del chunk_df, tab, ds
                gc.collect()
                if (i + 1) % 5 == 0 or i + 1 == n_chunks:
                    print(f"    chunk {i+1}/{n_chunks}  ·  RAM {mem_gb():.2f} GB")
            return (
                np.concatenate(chunks_X), np.concatenate(chunks_y),
                np.concatenate(chunks_w), np.concatenate(chunks_sid),
                np.concatenate(chunks_step), feature_cols_final,
            )

        if not SKIP_CLASSICAL:
            t0 = time.time()
            print("Building train tabular cache (chunked)...")
            X_tr, y_tr, w_tr, sid_tr, step_tr, feature_cols = _build_tabular_arrays_chunked(
                train_use, chunk_seqs=TABULAR_CHUNK_SEQS, with_engineered=True,
                rolling_windows=ROLLING_WINDOWS,
            )
            print("Building val tabular cache (chunked)...")
            X_va, y_va, w_va, sid_va, step_va, _ = _build_tabular_arrays_chunked(
                val_use, chunk_seqs=TABULAR_CHUNK_SEQS, with_engineered=True,
                rolling_windows=ROLLING_WINDOWS,
            )

            # Re-empaquetar en los TabularDataset dataclass para compatibilidad con el resto.
            from data.tabular import TabularDataset
            train_data = TabularDataset(X=X_tr, y=y_tr, weights=w_tr, seq_ids=sid_tr, step_ids=step_tr,
                                        feature_names=feature_cols, target_names=list(TARGET_COLS))
            val_data   = TabularDataset(X=X_va, y=y_va, weights=w_va, seq_ids=sid_va, step_ids=step_va,
                                        feature_names=feature_cols, target_names=list(TARGET_COLS))
            del X_tr, y_tr, w_tr, sid_tr, step_tr, X_va, y_va, w_va, sid_va, step_va
            gc.collect()

            print(f"\\nfeatures:   {len(feature_cols)}  ({'con' if WITH_ROLLING else 'sin'} rolling)")
            print(f"train_data: X{tuple(train_data.X.shape)}  y{tuple(train_data.y.shape)}")
            print(f"val_data:   X{tuple(val_data.X.shape)}   y{tuple(val_data.y.shape)}")
            print(f"RAM tras tabular cache: {mem_gb():.2f} GB  ·  build en {time.time()-t0:.1f}s")
    """))

    cells.append(code("""
        def run_classical(name: str, params: dict | None = None):
            ModelCls, kind = get_model_class(name)
            assert kind == "classical", f"{name} no es clásico"
            model = ModelCls(params or {})

            t0 = time.time()
            model.fit(
                train_data.X, train_data.y,
                sample_weight=train_data.weights,
                eval_set=(val_data.X, val_data.y),
                feature_names=feature_cols,
                target_names=val_data.target_names,
            )
            fit_seconds = time.time() - t0

            val_pred = model.predict(val_data.X)
            metrics_dict = summary(val_data.y, val_pred)
            per_seq = {
                int(sid): _weighted_pearson_scalar(val_data.y[val_data.seq_ids == sid],
                                                    val_pred[val_data.seq_ids == sid])
                for sid in np.unique(val_data.seq_ids)
            }
            val_eval = EvalResult(
                metrics=metrics_dict, predictions=val_pred, targets=val_data.y,
                seq_ids=val_data.seq_ids, step_ids=val_data.step_ids,
                per_sequence_corr=per_seq,
            )
            wp = metrics_dict["weighted_pearson"]
            per = metrics_dict["per_target"]

            # Persistir el run a disco.
            run_id = make_run_id(name)
            run_dir = EXPERIMENTS_ROOT / name / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save(run_dir / "model.joblib")
            write_report(
                run_dir=run_dir,
                config={"model": {"name": name, "params": params or {}},
                        "classical": {"rolling_windows": list(ROLLING_WINDOWS)},
                        "quick_mode": QUICK_MODE},
                history=None,
                val_eval=val_eval,
                extra_meta={"kind": "classical", "model_name": name,
                            "feature_columns": feature_cols,
                            "fit_seconds": fit_seconds,
                            "wallclock_seconds": fit_seconds},
            )

            row = {
                "model": name, "kind": "classical",
                "val_weighted_pearson": wp,
                "val_t0": per["t0"], "val_t1": per["t1"],
                "fit_seconds": fit_seconds,
                "n_features": len(feature_cols),
                "wallclock_seconds": fit_seconds,
                "run_dir": str(run_dir),
            }
            results.append(row)
            _flush_results()
            print(f"  ✓ {name:<14} corr={wp:+.4f}  (t0={per['t0']:+.4f}  t1={per['t1']:+.4f})  in {fit_seconds:5.1f}s  →  {run_dir}")
            return model, val_eval
        ;
    """))

    cells.append(code("""
        if SKIP_CLASSICAL:
            print("⊘ Sección clásicos saltada (SKIP_CLASSICAL=True). Pasa directamente a la sec. 6 (DL).")
            linear_model = ridge_model = lgb_model = None
            linear_eval = ridge_eval = lgb_eval = None
            if 'rf_model' not in dir(): rf_model = None
            if 'rf_eval' not in dir():  rf_eval = None
        else:
            # Linear (un punto de partida — diagnóstico de señal lineal).
            linear_model, linear_eval = run_classical("linear")
    """))

    cells.append(code("""
        if not SKIP_CLASSICAL:
            # Ridge — versión regularizada.
            ridge_model, ridge_eval = run_classical("ridge", {"alpha": 1.0})
    """))

    cells.append(code("""
        # Random Forest — modelo no-lineal.
        #
        # ⚠️  RF de sklearn no soporta early-stopping con la métrica del reto y escala mal
        # con muchas filas + alta profundidad. En LOB/HFT casi nunca gana a LightGBM (ver
        # writeups Optiver, JPX, G-Research). Por eso usamos hiperparámetros conservadores
        # incluso en modo completo: lo importante es tenerlo en la comparativa, no batirlo.
        #
        # Tiempo aproximado en CPU 8-core:
        #   QUICK_MODE=True   → ~3-5 min  (40 árboles × depth 8 sobre 1.35M filas)
        #   QUICK_MODE=False  → ~30-60 min  (100 árboles × depth 10 sobre 9.6M filas)
        #
        # Si tienes prisa o poca RAM, pon SKIP_RF=True arriba.
        if SKIP_CLASSICAL or SKIP_RF:
            if not SKIP_CLASSICAL:
                print("  ⊘  random_forest saltado (SKIP_RF=True)")
        else:
            rf_params = (
                {  # QUICK_MODE
                    "n_estimators": 40, "max_depth": 8, "min_samples_leaf": 500,
                    "max_features": 0.5, "n_jobs": -1, "random_state": SEED,
                }
                if QUICK_MODE else
                {  # full mode — conservador a propósito
                    "n_estimators": 100, "max_depth": 10, "min_samples_leaf": 500,
                    "max_features": 0.5, "n_jobs": -1, "random_state": SEED,
                }
            )
            rf_model, rf_eval = run_classical("random_forest", rf_params)
    """))

    cells.append(code("""
        if not SKIP_CLASSICAL:
            # LightGBM — históricamente el rey en Kaggles de microestructura.
            lgb_model, lgb_eval = run_classical(
                "lightgbm",
                {
                    "params": {
                        "objective": "regression", "metric": "None",
                        "learning_rate": 0.05, "num_leaves": 63,
                        "min_data_in_leaf": 100, "feature_fraction": 0.8,
                        "bagging_fraction": 0.8, "bagging_freq": 5,
                        "lambda_l2": 1.0, "verbose": -1,
                    },
                    "num_boost_round": 400 if QUICK_MODE else 3000,
                    "early_stopping_rounds": 50,
                    "log_period": 100000,
                },
            )

            # Liberar la cache tabular: los clásicos ya están entrenados y persistidos.
            del train_data, val_data
            gc.collect()
            print(f"RAM tras liberar tabular cache: {mem_gb():.2f} GB")
    """))

    # -------- 6 · DL baselines --------
    cells.append(md("""
        ## 6 · Iteración 2 — Modelos secuenciales DL básicos

        Los DL ven la secuencia completa de 1000 pasos. El loss enmascara el warm-up (pasos < 99).
        En modo rápido entrenamos 4 epochs con d_hidden 64 para iterar cómodo; en modo completo,
        los hiperparámetros se mueven a los valores de `configs/models/<name>.yaml`.
    """))

    cells.append(code("""
        # Construir datasets una sola vez (reutilizables para todos los modelos secuenciales).
        # Escalamos justo ahora (mantenemos los DataFrames escalados solo durante la sección DL).
        train_scaled = scaler.transform(train_use)
        val_scaled   = scaler.transform(val_use)
        # Ya no necesitamos train_use / val_use crudos: el evaluator de DL trabaja sobre los escalados.
        del train_use, val_use
        gc.collect()

        # Augmentation SOLO en train: crops temporales aleatorios + jitter de features.
        # Con ~10.7k secuencias fijas, un modelo de alta capacidad memoriza en pocas
        # epochs (Mamba-2 daba su mejor val en la epoch ~5). Los crops convierten esas
        # 10.7k ventanas fijas en sub-trayectorias prácticamente ilimitadas -> es la
        # palanca anti-overfit nº1. El val_ds queda intacto (secuencias completas,
        # sin ruido) para que su weighted-Pearson sea comparable al scorer.
        AUGMENT = {} if QUICK_MODE else {"crop_len": 512, "jitter_std": 0.05}

        train_arr = sequences_from_dataframe(train_scaled)
        val_arr   = sequences_from_dataframe(val_scaled)
        train_ds  = SequenceDataset(train_arr, seed=SEED, **AUGMENT)
        val_ds    = SequenceDataset(val_arr)
        # Las arrays están ya copiadas dentro del Dataset; podemos soltar las versiones intermedias.
        del train_arr
        gc.collect()

        print(f"train_ds: {len(train_ds)} secuencias  ·  val_ds: {len(val_ds)} secuencias"
              f"  ·  augment: {AUGMENT or 'off'}")
        print(f"RAM tras montar sequence datasets: {mem_gb():.2f} GB")
    """))

    cells.append(code("""
        def run_sequence(name: str, params: dict | None = None, *, trainer_overrides: dict | None = None):
            ModelCls, kind = get_model_class(name)
            assert kind == "sequence", f"{name} no es secuencial"
            params = {**(params or {}), "n_features": N_FEATURES, "n_targets": N_TARGETS}
            model = ModelCls(params)

            cfg_dict = dict(
                epochs=QUICK_EPOCHS if QUICK_MODE else 20,
                batch_size=16 if QUICK_MODE else 32,
                learning_rate=1e-3,
                weight_decay=1e-4,
                warmup_epochs=0 if QUICK_MODE else 1,
                grad_clip=1.0,
                eval_every=1,
                device=DEVICE,
                amp=(DEVICE == "cuda"),
                seed=SEED,
                loss_name="weighted_pearson",
                early_stopping_patience=2 if QUICK_MODE else 5,
                num_workers=0,
                pin_memory=(DEVICE == "cuda"),
            )
            cfg_dict.update(trainer_overrides or {})
            trainer_cfg = TrainerConfig(**cfg_dict)

            t0 = time.time()
            info = train_sequence_model(model, train_ds, val_ds, trainer_cfg)
            elapsed = time.time() - t0

            ev = Evaluator()
            val_eval = ev.run_sequence_model(model, val_scaled, device=trainer_cfg.resolve_device())
            wp = val_eval.metrics["weighted_pearson"]
            per = val_eval.metrics["per_target"]

            # Persistir el run a disco (weights + métricas + plots + history).
            run_id = make_run_id(name)
            run_dir = EXPERIMENTS_ROOT / name / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save(run_dir / "model.pt")
            write_report(
                run_dir=run_dir,
                config={"model": {"name": name, "params": params},
                        "training": cfg_dict,
                        "quick_mode": QUICK_MODE},
                history=info["history"],
                val_eval=val_eval,
                extra_meta={"kind": "sequence", "model_name": name,
                            "best_score_during_train": float(info["best_score"]),
                            "wallclock_seconds": elapsed},
            )

            row = {
                "model": name,
                "kind": "sequence",
                "val_weighted_pearson": wp,
                "val_t0": per["t0"],
                "val_t1": per["t1"],
                "fit_seconds": elapsed,
                "epochs_done": len(info["history"]),
                "best_score_during_train": info["best_score"],
                "wallclock_seconds": elapsed,
                "run_dir": str(run_dir),
            }
            results.append(row)
            _flush_results()
            print(f"  ✓ {name:<14} corr={wp:+.4f}  (t0={per['t0']:+.4f}  t1={per['t1']:+.4f})  epochs={len(info['history'])}  in {elapsed:5.1f}s  →  {run_dir}")
            return model, info, val_eval
        ;
    """))

    cells.append(code("""
        gru_model, gru_info, gru_eval = run_sequence(
            "gru",
            {"hidden_size": 64 if QUICK_MODE else 128, "num_layers": 2, "dropout": 0.1},
        )
    """))

    cells.append(code("""
        lstm_model, lstm_info, lstm_eval = run_sequence(
            "lstm",
            {"hidden_size": 64 if QUICK_MODE else 128, "num_layers": 2, "dropout": 0.1},
        )
    """))

    cells.append(code("""
        transformer_model, tfm_info, tfm_eval = run_sequence(
            "transformer",
            {
                "d_model": 64 if QUICK_MODE else 128,
                "nhead": 4,
                "num_layers": 2 if QUICK_MODE else 3,
                "dim_feedforward": 128 if QUICK_MODE else 256,
                "dropout": 0.1,
                "max_len": 1024,
            },
        )
    """))

    # -------- 7 · SOTA-flavoured models --------
    cells.append(md("""
        ## 7 · Iteración 3 — Modelos específicos / SOTA

        **DeepLOB** (Zhang 2019) — CNN multi-escala + bloque inception + LSTM publicado en IEEE TSP.
        En el paper original cada ventana de 100 pasos produce **una sola** predicción softmax para
        el movimiento de mid-price en `T + k`, así que las convs internas no se preocupan por la
        causalidad: toda la ventana es pasado respecto al target.

        En este reto el target es **per-step** sobre secuencias de 1000 pasos, así que el port ingenuo
        (mismas convs `(4,1)` sin padding + `F.interpolate` para recuperar `T`) introduce *look-ahead*
        de ~20 pasos al material de cada predicción. Nuestra adaptación: (i) cabeza softmax → `Linear(2)`
        por paso, (ii) todas las convs temporales pasan a **causales** (left-pad solo), (iii) se
        elimina el upsample porque la pila causal preserva `T`. Esto se valida con
        `tests/models/test_registry.py::test_deeplob_streaming_matches_forward` (paridad batched↔streaming).

        **TCN** (Bai, Kolter & Koltun 2018) — *Temporal Convolutional Network*: pila residual de
        convs 1-D **causales** con dilatación exponencial (1, 2, 4, 8, 16, …). Con 5 bloques y
        kernel 3 cubre un *receptive field* de 125 pasos con ~125 k parámetros — mucho más eficiente
        que la pila kernel-4 de DeepLOB y sin el cuello de botella recurrente de los RNN. La
        causalidad es por construcción: cada bloque hace `F.pad(..., (k-1)*d, 0)` (left-only),
        así que el output en `t` depende solo de inputs `<= t`. Se valida con
        `tests/models/test_registry.py::test_tcn_streaming_matches_forward` y `test_tcn_causality`.
        Para entender la arquitectura en detalle, ver `docs/tcn.md`.

        **Mamba-2** (Dao & Gu 2024) — *Selective State-Space Model* con coste **lineal** en T (sin
        matriz de atención O(T²)). Implementación **propia en PyTorch puro** (algoritmo SSD por
        *chunks*); no depende de `mamba-ssm`/`causal-conv1d` (que no compilan fácil en WSL2). El
        *scan* selectivo corre en **fp32 aun bajo AMP fp16**, así que los `exp()`/`cumsum` del
        decaimiento nunca desbordan a NaN → no hay colapso de la métrica. Incluye streaming
        recurrente O(1) por paso para el scorer. Ver `src/models/sequence/mamba2.py`.
    """))

    cells.append(code("""
        deeplob_model, deeplob_info, deeplob_eval = run_sequence(
            "deeplob",
            {
                "window_size": 100,
                "lstm_hidden": 32 if QUICK_MODE else 64,
                "c1": 16, "c2": 16, "c3": 32,
                "inception_channels": 16 if QUICK_MODE else 32,
                "dropout": 0.1,
            },
            trainer_overrides={"learning_rate": 1e-4},
        )
    """))

    cells.append(code("""
        # Config tuneada vía scripts/sweep_tcn.py (variante C_L6_ch96, val=+0.2740).
        # En QUICK_MODE bajamos canales y capas para acotar tiempo.
        tcn_model, tcn_info, tcn_eval = run_sequence(
            "tcn",
            {
                "channels": 48 if QUICK_MODE else 96,
                "num_layers": 5 if QUICK_MODE else 6,    # RF: quick=125, full=253
                "kernel_size": 3,
                "dropout": 0.1,
            },
        )
    """))

    cells.append(code("""
        # Mamba-2 (SSD, PyTorch puro) — preset CAPACIDAD ALTA + AUGMENTATION:
        # Experimentos medidos: subir regularización BAJÓ el pico (0.262->0.248) y las
        # ramas por target (dual-trunk) NO ayudaron (t1 colapsa igual con rama propia,
        # luego no era interferencia t0/t1 sino que t1 es un target de baja señal que
        # sobre-ajusta a ~0 por sí solo). Receta ganadora: capacidad alta + augmentation
        # (crops 512, retrasan el colapso) + early-stopping (guarda el pico; el colapso
        # nunca llega al checkpoint shippeado).
        # · d_model=256 · 8 capas · d_state=128 · headdim=64  (~3.7M params)
        # · dropout 0.2 + drop_path 0.1 (reg ligera); weight_decay 0.01 (param groups)
        # · LR 2e-4, warmup 4, 100 epochs, paciencia 15; augment crops 512 + jitter
        # · ~4.6 GB pico @ batch=24, T=512 en la 3060 Ti  ->  ~1.5 min/epoch
        # (Dual-trunk sigue disponible vía split_targets=True pero no aporta; off.)
        if SKIP_MAMBA:
            mamba_model, mamba_info, mamba_eval = None, None, None
            print("Saltando Mamba-2 (SKIP_MAMBA=True).")
        else:
            mamba_model, mamba_info, mamba_eval = run_sequence(
                "mamba2",
                {
                    "d_model": 128 if QUICK_MODE else 256,
                    "num_layers": 4 if QUICK_MODE else 8,
                    "headdim": 64,
                    "d_state": 64 if QUICK_MODE else 128,
                    "expand": 2,
                    "chunk_size": 128,        # 512 es múltiplo de 128 -> sin padding
                    "dropout": 0.2,
                    "drop_path": 0.1,
                },
                trainer_overrides={
                    "learning_rate": 2e-4,
                    "weight_decay": 1e-2,
                    "warmup_epochs": 1 if QUICK_MODE else 4,
                    "epochs": QUICK_EPOCHS if QUICK_MODE else 100,
                    "early_stopping_patience": 4 if QUICK_MODE else 15,
                    "ema_decay": 0.999,
                    "batch_size": 24,         # ~4.6 GB pico; sube a 32 (6.2 GB) si quieres
                },
            )
            print(f"\\nMamba-2 topology: {mamba_model.backend_used} "
                  f"({sum(p.numel() for p in mamba_model.parameters())/1e6:.2f} M params)")
    """))

    cells.append(code("""
        # TLOB (Garcia et al., 2024, arXiv:2403.09989) — el SOTA específico de LOB.
        # Doble attention: features (cada timestep atiende a las 32 columnas)
        # + tiempo (cada feature atiende causalmente al pasado). Per-target heads.
        #
        # CONFIG BAJO-VRAM + MÁS REGULARIZADA (la anterior d64/L4 hacía OOM en la
        # 3060 Ti: 5.86 GB de pico + fragmentación tras un run largo). Fixes:
        #   · grad_checkpoint=True -> recomputa los bloques en backward: 5.86 -> 1.50 GB
        #     (mismo modelo, exacto, ~30% más lento). ES la palanca clave.
        #   · modelo más pequeño (d48/L3) + dropout 0.35 + drop_path 0.30
        #   · los crops de 512 (AUGMENT global) ~bajan a la mitad la activación dominante
        #   -> pico ~0.5-0.6 GB. Si aun así OOM en runs largos, lanza con
        #      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        if SKIP_TLOB:
            tlob_model, tlob_info, tlob_eval = None, None, None
            print("Saltando TLOB (SKIP_TLOB=True).")
        else:
            tlob_model, tlob_info, tlob_eval = run_sequence(
                "tlob",
                {
                    "d_model": 48,
                    "num_layers": 2 if QUICK_MODE else 3,
                    "n_heads": 4,
                    "ffn_mult": 4,
                    "dropout": 0.25 if QUICK_MODE else 0.35,
                    "drop_path": 0.10 if QUICK_MODE else 0.30,
                    "max_len": 1024,
                    "grad_checkpoint": not QUICK_MODE,
                },
                trainer_overrides={
                    "learning_rate": 3e-4,
                    "weight_decay": 5e-4 if QUICK_MODE else 8e-3,
                    "warmup_epochs": 1 if QUICK_MODE else 6,
                    "epochs": QUICK_EPOCHS if QUICK_MODE else 100,
                    "early_stopping_patience": 4 if QUICK_MODE else 15,
                    "ema_decay": 0.999,
                    "batch_size": 8,
                },
            )

        # Liberamos las arrays / datasets de secuencias — ya están persistidas y los
        # val_eval que necesita la sección 8 quedaron en memoria como objetos Python pequeños.
        del val_arr, train_ds, val_ds, train_scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"RAM tras liberar sequence datasets: {mem_gb():.2f} GB")
    """))

    # -------- 8 · Comparison + selection --------
    cells.append(md("""
        ## 8 · Comparativa final y selección del mejor modelo

        Recopilamos los resultados de las tres rondas, ordenamos por `val_weighted_pearson` y
        examinamos en detalle los dos mejores (scatter de predicción vs target y distribución
        de correlación por secuencia).
    """))

    cells.append(code("""
        results_df = pd.DataFrame(results).sort_values("val_weighted_pearson", ascending=False).reset_index(drop=True)
        results_df.style.format({
            "val_weighted_pearson": "{:+.4f}",
            "val_t0": "{:+.4f}",
            "val_t1": "{:+.4f}",
            "fit_seconds": "{:.1f}",
            "wallclock_seconds": "{:.1f}",
        }).background_gradient(subset=["val_weighted_pearson"], cmap="RdYlGn")
    """))

    cells.append(code("""
        # Barplot ordenado.
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = {"classical": "tab:blue", "sequence": "tab:orange"}
        colors = [palette[k] for k in results_df["kind"]]
        ax.barh(results_df["model"][::-1], results_df["val_weighted_pearson"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Weighted Pearson en val")
        ax.set_title("Ranking de modelos (val_weighted_pearson, ↑ mejor)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=palette[k], label=k) for k in palette])
        fig.tight_layout()
        plt.show()
    """))

    cells.append(code("""
        # Diagnóstico del mejor modelo: scatter pred vs true por target.
        # Los val_eval (objetos pequeños — solo predicciones + ids) los guardamos en este dict
        # para no tener que recomputar; los grandes (modelos, dataframes) ya están liberados o en disco.
        all_evals = {}
        candidates = [
            ("gru", gru_eval), ("lstm", lstm_eval), ("transformer", tfm_eval),
            ("deeplob", deeplob_eval), ("tcn", tcn_eval), ("mamba2", mamba_eval),
            ("tlob", tlob_eval),
        ]
        if not SKIP_CLASSICAL:
            candidates += [("linear", linear_eval), ("ridge", ridge_eval), ("lightgbm", lgb_eval)]
            if not SKIP_RF and 'rf_eval' in dir() and rf_eval is not None:
                candidates.append(("random_forest", rf_eval))
        for k, v in candidates:
            if v is not None:
                all_evals[k] = v

        best_row = results_df.iloc[0]
        best_name = best_row["model"]
        best_kind = best_row["kind"]
        best_eval = all_evals[best_name]
        print(f"Mejor modelo: {best_name}  (val_weighted_pearson = {best_row['val_weighted_pearson']:+.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, j, name in zip(axes, [0, 1], TARGET_COLS):
            ax.scatter(best_eval.targets[:, j], best_eval.predictions[:, j], s=4, alpha=0.2, color="steelblue")
            lim = max(np.abs(best_eval.targets[:, j]).max(), np.abs(best_eval.predictions[:, j]).max(), 1.0)
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"{name} (true)"); ax.set_ylabel(f"{name} (pred)")
            ax.set_title(f"{name}: ρ_w = {best_eval.metrics['per_target'][name]:+.4f}")
            ax.grid(alpha=0.2)
        fig.suptitle(f"Predicciones vs ground truth — modelo: {best_name}")
        fig.tight_layout()
        plt.show()
    """))

    cells.append(code("""
        # Distribución de la correlación weighted Pearson por secuencia (mide robustez).
        seq_corr = pd.Series(best_eval.per_sequence_corr)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(seq_corr, bins=40, ax=ax, color="tab:orange")
        ax.axvline(seq_corr.mean(),   color="black", linestyle="--", label=f"mean   {seq_corr.mean():+.3f}")
        ax.axvline(seq_corr.median(), color="gray",  linestyle=":",  label=f"median {seq_corr.median():+.3f}")
        ax.set_xlabel("weighted Pearson por seq_ix")
        ax.legend()
        ax.set_title(f"Distribución de correlación por secuencia — {best_name}")
        fig.tight_layout()
        plt.show()
    """))

    # -------- 8.bis · Ensemble --------
    cells.append(md("""
        ## 8.bis · Ensemble (promedio ponderado)

        Cada target tiene un **techo de señal por modelo** (~0.38 en `t0`, ~0.13 en `t1`):
        TCN y Mamba-2, arquitecturas muy distintas, llegan al *mismo* techo por separado —
        es un límite de la predictibilidad de los datos, no del modelo. La ganancia fiable
        restante viene de **promediar modelos decorrelacionados**: sus errores no están
        perfectamente correlacionados, así que la media reduce varianza y suele superar al
        mejor individual (sobre todo en `t1`).

        Incluimos **LightGBM** además de los modelos de secuencia: es el miembro con el sesgo
        inductivo más distinto (árboles vs redes), por lo que sus errores decorrelacionan más
        y aporta la mayor ganancia al promedio. Mezclar secuencia + clásico exige alinear las
        predicciones por `(seq_ix, step_in_seq)` (los dos evaluadores recorren las filas en
        orden distinto), lo cual hacemos explícitamente.

        Afinamos pesos convexos sobre las predicciones de **validación** (ya enmascaradas en
        `all_evals`) maximizando el weighted-Pearson del scorer, y si el ensemble supera al
        mejor individual lo empaquetamos como `solution.zip` final
        (`submission.package_ensemble`). El `EnsemblePredictionModel` avanza el estado de
        streaming de *cada* miembro en cada paso (el de LightGBM recomputa sus features
        ingenieradas incrementalmente) y promedia solo en las filas puntuadas.
    """))

    cells.append(code("""
        from submission.ensemble import tune_ensemble_weights
        from submission import package_ensemble

        # Candidatos: (nombre, modelo, kind). Los de secuencia comparten el scaler;
        # LightGBM es clásico (sesgo inductivo distinto -> más decorrelación -> más ganancia).
        _candidates = [
            ("tcn",      tcn_model       if 'tcn_model'     in dir() else None, "sequence"),
            ("mamba2",   mamba_model     if 'mamba_model'   in dir() else None, "sequence"),
            ("gru",      gru_model       if 'gru_model'     in dir() else None, "sequence"),
            ("lstm",     lstm_model      if 'lstm_model'    in dir() else None, "sequence"),
            ("deeplob",  deeplob_model   if 'deeplob_model' in dir() else None, "sequence"),
            ("tlob",     tlob_model      if 'tlob_model'    in dir() else None, "sequence"),
            ("lightgbm", lgb_model       if 'lgb_model'     in dir() else None, "classical"),
        ]
        _members = [(n, m, k) for n, m, k in _candidates if m is not None and n in all_evals]

        def _aligned(ev):
            # Ordena las filas puntuadas por (seq_ix, step_in_seq): mezclar miembros de
            # secuencia y clásicos exige alinear (cada evaluador las emite en distinto orden).
            key = ev.seq_ids.astype(np.int64) * 1000 + ev.step_ids.astype(np.int64)
            order = np.argsort(key, kind="stable")
            return key[order], ev.predictions[order], ev.targets[order]

        ensemble_zip = None
        tune = None
        if len(_members) >= 2:
            names = [n for n, _, _ in _members]
            keys0, _, y_ens = _aligned(all_evals[names[0]])
            preds = []
            for n in names:
                k, p, y = _aligned(all_evals[n])
                assert np.array_equal(k, keys0), f"{n}: filas puntuadas no coinciden con {names[0]}"
                assert np.allclose(y, y_ens, atol=1e-4), f"{n}: targets desalineados"
                preds.append(p)

            tune = tune_ensemble_weights(preds, y_ens, step=0.05)
            best_single = max(tune["per_member"])
            for n, s in sorted(zip(names, tune["per_member"]), key=lambda t: -t[1]):
                print(f"  {n:<10} val_wp = {s:+.4f}")
            print(f"\\nPesos óptimos: {{{', '.join(f'{n}:{w:.2f}' for n, w in zip(names, tune['weights']))}}}")
            print(f"ENSEMBLE val_wp = {tune['score']:+.4f}  "
                  f"(mejor individual = {best_single:+.4f}, ganancia = {tune['score'] - best_single:+.4f})")

            _blend = sum(w * p for w, p in zip(tune["weights"], preds))
            _ens_per = summary(y_ens, _blend)["per_target"]

            if tune["score"] > best_single:
                # Conservamos solo miembros con peso relevante (>=2%), renormalizando.
                kept = [(n, m, k, w) for (n, m, k), w in zip(_members, tune["weights"]) if w >= 0.02]
                wsum = sum(w for *_, w in kept)
                ens_dir = EXPERIMENTS_ROOT / "ensemble" / make_run_id("ensemble")
                ens_dir.mkdir(parents=True, exist_ok=True)
                members = []
                for n, m, k, w in kept:
                    if k == "sequence":
                        art = ens_dir / f"{n}.pt"; m.save(art)
                        members.append(dict(kind="sequence", model_name=n, model_artifact=art,
                                            scaler_state=scaler.state_dict()))
                    else:  # classical (LightGBM): recomputa features ingenieradas al vuelo
                        art = ens_dir / f"{n}.joblib"; m.save(art)
                        members.append(dict(kind="classical", model_name=n, model_artifact=art,
                                            feature_columns=list(m.feature_names_),
                                            with_engineered=True,
                                            rolling_windows=tuple(ROLLING_WINDOWS)))
                utils_py = Path("competition_package/utils.py")
                if utils_py.exists():
                    ensemble_zip = package_ensemble(
                        run_dir=ens_dir, src_root=Path("src"), utils_py=utils_py,
                        members=members, weights=[w / wsum for *_, w in kept])
                    print(f"\\n>>> El ENSEMBLE es el modelo final. solution.zip: {ensemble_zip}")
                    print(f"    miembros: {[(n, round(w / wsum, 2)) for n, _, _, w in kept]}")
            else:
                print("\\nEl ensemble no supera al mejor individual; se entrega el modelo único.")

            # Registrar el ensemble en la tabla acumulada de resultados (+ CSV).
            if 'results' in dir():
                results.append({
                    "model": "ensemble", "kind": "ensemble",
                    "val_weighted_pearson": float(tune["score"]),
                    "val_t0": float(_ens_per["t0"]), "val_t1": float(_ens_per["t1"]),
                    "members": ", ".join(f"{n}:{w:.2f}" for n, w in zip(names, tune["weights"]) if w >= 0.02),
                })
                if '_flush_results' in dir():
                    _flush_results()
                results_df = pd.DataFrame(results).sort_values(
                    "val_weighted_pearson", ascending=False).reset_index(drop=True)
                print("\\nTabla de resultados actualizada (ensemble incluido):")
                print(results_df[["model", "kind", "val_weighted_pearson", "val_t0", "val_t1"]].to_string(index=False))
        else:
            print("Menos de 2 miembros disponibles; se omite el ensemble.")
    """))

    # -------- 9 · Selección del mejor modelo --------
    cells.append(md("""
        ## 9 · Selección del modelo final

        El mejor modelo (mayor `val_weighted_pearson`) ya está persistido por
        `run_classical` / `run_sequence` en `experiments/<best>/<run_id>/`. Aquí solo
        imprimimos su ubicación y muestramos los artefactos generados para que sepas
        exactamente dónde encontrar `model.{pt,joblib}`, `metrics.json`, `predictions.parquet`
        y los plots.
    """))

    cells.append(code("""
        best_run_dir = Path(best_row["run_dir"])
        artefacts = sorted(p.relative_to(best_run_dir) for p in best_run_dir.rglob("*") if p.is_file())

        print(f"Modelo elegido:   {best_name}  ({best_kind})")
        print(f"Run directory:    {best_run_dir}")
        print(f"val_weighted_pearson = {best_row['val_weighted_pearson']:+.4f}")
        print()
        print("Artefactos persistidos:")
        for p in artefacts:
            print(f"  - {p}")
    """))

    cells.append(code("""
        # Snapshot de la tabla acumulada (también está en experiments/notebook_results.csv).
        print(f"results CSV: {NOTEBOOK_RESULTS_CSV}")
        print()
        results_df.to_string(index=False)
    """))

    # -------- 10 · Conclusions --------
    cells.append(md("""
        ## 10 · Conclusiones y próximos pasos

        ### Lecciones de las iteraciones

        * **Clásicos**: en problemas de microestructura una regresión lineal sobre features
          ingenieradas suele ser un piso fuerte. LightGBM con `sample_weight = |y|` y early stop
          en weighted Pearson aprovecha bien la no-linealidad.
        * **RNN básicos (GRU/LSTM)**: aprenden el régimen temporal sin necesidad de ventanas,
          pero son sensibles al learning rate y al warm-up.
        * **Transformer causal**: requiere más datos / epochs para empezar a generalizar; en
          modo rápido suele quedar por debajo de los RNN.
        * **DeepLOB**: arquitectura específica para LOB; en regresión hay que cuidar que la
          interpolación temporal no diluya la señal del último step.
        * **TCN**: stack de convoluciones causales con dilatación exponencial. Es el
          **modelo ganador** del proyecto (+0.2740 val_pearson), supera a LSTM/GRU/Transformer
          con menos parámetros (~330k) y entrenamiento paralelo. El sweep en
          `scripts/sweep_tcn.py` confirmó que la ganancia viene de combinar receptive field
          extra (L=6 → RF=253) con más canales (96), y que kernel=5 no mejora frente a kernel=3.
        * **Mamba-2**: con kernels nativos CUDA escala lineal en T = 1000 sin penalización
          de memoria, lo que lo hace especialmente atractivo para esta longitud.

        ### Próximos pasos

        * **TLOB / MLPLOB** (Berti & Kasneci 2025) — actual SOTA en FI-2010; arquitecturas
          pequeñas, fácil de portar al `models.sequence/` y registrar.
        * **Supervised-AE + Multi-task MLP** — receta ganadora de Jane Street 2021, ideal para
          ensemblar con LightGBM y los RNN.
        * **Ensembles** — rank-then-average de los top-3 suele aportar +5–15 % de correlación.
        * **Tuning**: una sweep ligera con Optuna sobre el LightGBM y un *learning-rate finder*
          para los DL es lo siguiente más rentable.

        ### Reproducibilidad

        Todos los runs producen un directorio en `experiments/<model>/<run_id>/` con
        `config.yaml`, `metrics.json`, `predictions.parquet`, `scatter.png`,
        `per_seq_corr.png` y (para los DL) `train_history.{csv,png}`. La tabla agregada
        de todos los runs se mantiene en `experiments/notebook_results.csv` y sobrevive
        a reinicios del kernel. Cualquier modelo nuevo (subclase con
        `@register_model("name", kind=…)` + un YAML) se integra al pipeline sin tocar
        nada más; ver `README.md` ➝ *Adding a new model*.
    """))

    return cells


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = build_cells()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (lob)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "version": "3.12",
        },
    }
    return nb


def main() -> None:
    nb = build_notebook()
    OUT_PATH.write_text(nbf.writes(nb, version=4), encoding="utf-8")
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"wrote {OUT_PATH}  ({n_md} markdown + {n_code} code cells)")


if __name__ == "__main__":
    main()
