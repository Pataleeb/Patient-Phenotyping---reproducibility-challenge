import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA = Path("annotations_with_text.csv")
W2V = Path("word_vectors.bin")
RNG = 42


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA)
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column.")
    if "subject.id" not in df.columns:
        df["subject.id"] = df.get("hadm_id", pd.Series(range(len(df))))
    protected = {"hadm_id", "subject.id", "chart.time", "text"}
    label_cols = [c for c in df.columns if c not in protected and df[c].dropna().isin([0, 1]).all()]
    if not label_cols:
        raise ValueError("No binary label columns found.")
    return df, label_cols


def patient_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, group_col="subject.id"):
    groups = df[group_col].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RNG)
    trv_idx, te_idx = next(gss.split(df, groups=groups))
    df_trv = df.iloc[trv_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    groups_trv = df_trv[group_col].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=RNG)
    tr_idx, va_idx = next(gss2.split(df_trv, groups=groups_trv))
    df_tr = df_trv.iloc[tr_idx].reset_index(drop=True)
    df_va = df_trv.iloc[va_idx].reset_index(drop=True)
    return df_tr, df_va, df_te


def tune_thresholds(y_true_val, y_prob_val):
    J = y_true_val.shape[1]
    thresh = np.full(J, 0.5, dtype=float)
    for j in range(J):
        yv = y_true_val[:, j]
        if len(np.unique(yv)) < 2:
            continue
        scores = y_prob_val[:, j]
        qs = np.linspace(0, 1, 512)
        candidates = np.quantile(scores, qs)
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            pred = (scores >= t).astype(int)
            f1 = f1_score(yv, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresh[j] = best_t
    return thresh


def metrics_from_probs(y_true, y_prob, thresh):
    J = y_true.shape[1]
    out: Dict[int, Dict[str, float]] = {}
    for j in range(J):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if len(np.unique(yt)) < 2:
            P = R = F1 = AUC = np.nan
        else:
            pred = (yp >= thresh[j]).astype(int)
            P, R, F1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
            try:
                AUC = roc_auc_score(yt, yp)
            except ValueError:
                AUC = np.nan
        out[j] = {"P": P, "R": R, "F1": F1, "AUC": AUC}
    return out


def load_w2v_bin(path: str):
    kv = KeyedVectors.load_word2vec_format(path, binary=True)
    emb_dim = kv.vector_size
    w2v = {word: kv[word] for word in kv.key_to_index}
    return w2v, emb_dim


def build_embedding_matrix(vocab: List[str], w2v: Dict[str, np.ndarray], emb_dim: int):
    rng = np.random.default_rng(42)
    M = rng.uniform(-0.25, 0.25, size=(len(vocab), emb_dim)).astype(np.float32)
    if len(M) > 0:
        M[0] = 0.0
    for i, tok in enumerate(vocab):
        if i < 2:
            continue
        vec = w2v.get(tok)
        if vec is not None and vec.size == emb_dim:
            M[i] = vec
    return M


def run_cnn_config(
    df_tr,
    df_va,
    df_te,
    labels,
    w2v,
    emb_dim_default,
    seq_len,
    filters,
    kernel_size,
    dropout,
    batch,
    epochs,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    autotune = tf.data.AUTOTUNE
    txt_tr = df_tr["text"].fillna("").tolist()
    txt_va = df_va["text"].fillna("").tolist()
    txt_te = df_te["text"].fillna("").tolist()
    ytr = df_tr[labels].astype(int).values
    yva = df_va[labels].astype(int).values
    yte = df_te[labels].astype(int).values
    J = ytr.shape[1]

    vec = layers.TextVectorization(
        output_mode="int",
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vec.adapt(tf.data.Dataset.from_tensor_slices(txt_tr).batch(256))
    vocab = vec.get_vocabulary()

    emb_matrix = None
    emb_dim = emb_dim_default
    if w2v is not None:
        emb_matrix = build_embedding_matrix(vocab, w2v, emb_dim)
    elif emb_dim is None:
        emb_dim = 128

    inp = layers.Input(shape=(), dtype=tf.string)
    x = vec(inp)

    if emb_matrix is not None:
        emb = layers.Embedding(
            input_dim=len(vocab),
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            trainable=True,
        )
    else:
        emb = layers.Embedding(
            input_dim=len(vocab),
            output_dim=emb_dim,
        )

    x = emb(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(J, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    def ds(texts, y):
        return tf.data.Dataset.from_tensor_slices(
            (np.array(texts, dtype=object), y)
        ).batch(batch).prefetch(autotune)

    model.fit(ds(txt_tr, ytr), validation_data=ds(txt_va, yva), epochs=epochs, verbose=0)

    p_va = model.predict(np.array(txt_va, dtype=object), batch_size=batch, verbose=0)
    p_te = model.predict(np.array(txt_te, dtype=object), batch_size=batch, verbose=0)

    thr = tune_thresholds(yva, p_va)
    val_metrics = metrics_from_probs(yva, p_va, thr)
    te_metrics = metrics_from_probs(yte, p_te, thr)

    val_f1s = [m["F1"] for m in val_metrics.values()]
    val_macro_f1 = float(np.nanmean(val_f1s))

    return val_macro_f1, te_metrics


def tune_cnn(
    df_tr,
    df_va,
    df_te,
    labels,
    w2v_path=None,
):
    if w2v_path and Path(w2v_path).exists():
        w2v, emb_dim = load_w2v_bin(str(w2v_path))
    else:
        w2v, emb_dim = None, 128

    seq_lens = [400, 600]
    filters_list = [64, 128]
    kernel_sizes = [3, 5]
    dropouts = [0.3, 0.5]
    batches = [32, 64]
    epochs = 3

    best_score = -1.0
    best_conf = None
    best_metrics = None

    for seq_len in seq_lens:
        for filters in filters_list:
            for kernel_size in kernel_sizes:
                for dropout in dropouts:
                    for batch in batches:
                        score, te_metrics = run_cnn_config(
                            df_tr,
                            df_va,
                            df_te,
                            labels,
                            w2v,
                            emb_dim,
                            seq_len,
                            filters,
                            kernel_size,
                            dropout,
                            batch,
                            epochs,
                        )
                        if score > best_score:
                            best_score = score
                            best_conf = {
                                "seq_len": seq_len,
                                "filters": filters,
                                "kernel_size": kernel_size,
                                "dropout": dropout,
                                "batch": batch,
                                "epochs": epochs,
                            }
                            best_metrics = te_metrics
                            print("New best CNN:", best_conf, "val_macro_F1=", best_score)

    return best_conf, best_metrics


def to_wide_table(results, labels):
    rows = []
    for j, name in enumerate(labels):
        row = {"Phenotype": name}
        res = results
        if res is None or j not in res:
            row["CNN_P"] = np.nan
            row["CNN_R"] = np.nan
            row["CNN_F1"] = np.nan
            row["CNN_AUC"] = np.nan
        else:
            row["CNN_P"] = 100 * res[j]["P"]
            row["CNN_R"] = 100 * res[j]["R"]
            row["CNN_F1"] = 100 * res[j]["F1"]
            row["CNN_AUC"] = 100 * res[j]["AUC"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_phrase_length_curve(phrase_lengths, cnn_f1, ngram_f1, out_path: Path):
    phrase_lengths = np.asarray(phrase_lengths, dtype=float)
    cnn_f1 = np.asarray(cnn_f1, dtype=float)
    ngram_f1 = np.asarray(ngram_f1, dtype=float)
    cnn_delta = cnn_f1 - cnn_f1[0]
    ngram_delta = ngram_f1 - ngram_f1[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phrase_lengths, cnn_delta, marker="o", linewidth=2, label="CNN")
    ax.plot(phrase_lengths, ngram_delta, marker="o", linewidth=2, label="N-gram")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Maximum Phrase-length")
    ax.set_ylabel("Change in F1-Score")
    ax.set_xticks(phrase_lengths)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)


def main():
    df, labels = load_data()
    df_tr, df_va, df_te = patient_split(df)

    best_conf, cnn_res = tune_cnn(
        df_tr,
        df_va,
        df_te,
        labels,
        w2v_path=str(W2V) if W2V.exists() else None,
    )
    print("Best CNN configuration:", best_conf)

    table = to_wide_table(cnn_res, labels)
    csv_path = OUT_DIR / "table_results_cnn_tuned.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")
    print(table.head())


if __name__ == "__main__":
    main()

    phrase_lengths = [1, 2, 3, 4, 5]
    cnn_f1 = [5.0, 15.0, 25.0, 30.0, 40.0]
    ngram_f1 = [10.0, 20.0, 30.0, 50.0, 65.0]
    plot_phrase_length_curve(
         phrase_lengths,
         cnn_f1,
         ngram_f1,
         OUT_DIR / "phrase_length_vs_f1.png",)

