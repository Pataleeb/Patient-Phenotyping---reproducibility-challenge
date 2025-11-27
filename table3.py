
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("annotations_with_text.csv")
RNG = 42

TARGET_LABELS = ["Advanced.Heart.Disease", "Alcohol.Abuse"]
TOP_K = 20
SEQ_LEN = 600
EPOCHS = 3
BATCH = 64
MAX_TOKS_PER_DOC = 300
N_DOCS_PER_LABEL = 60


from ctakes import build_ctakes_text_columns

def load_df() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA)
    if "text" not in df.columns: raise ValueError("missing 'text'")
    if "subject.id" not in df.columns:
        df["subject.id"] = df.get("hadm_id", pd.RangeIndex(len(df)))
    protected = {"hadm_id","subject.id","chart.time","text","clean_text"}
    labels = [c for c in df.columns if c not in protected and df[c].dropna().isin([0,1]).all()]
    missing = [t for t in TARGET_LABELS if t not in labels]
    if missing: raise ValueError(f"Missing targets: {missing}")
    return df, labels

def split_by_patient(df: pd.DataFrame, test_size=0.2, val_size=0.1, group_col="subject.id"):
    groups = df[group_col].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RNG)
    trv_idx, te_idx = next(gss.split(df, groups=groups))
    df_trv, df_te = df.iloc[trv_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    groups_trv = df_trv[group_col].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1.0 - test_size), random_state=RNG)
    tr_idx, va_idx = next(gss2.split(df_trv, groups=groups_trv))
    return df_trv.iloc[tr_idx].reset_index(drop=True), df_trv.iloc[va_idx].reset_index(drop=True), df_te

def build_cnn(vec, emb_dim=128, n_outputs=2, seq_len=600):
    inp = layers.Input(shape=(1,), dtype=tf.string)
    x = vec(inp)
    vocab_size = len(vec.get_vocabulary())
    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_outputs, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def fit_cnn(df_tr, df_va):
    vec = layers.TextVectorization(
        output_mode="int",
        output_sequence_length=600,
        standardize="lower_and_strip_punctuation",
        split="whitespace",

        # max_tokens=50000
    )
    txt_tr = df_tr["text"].fillna("").tolist()
    vec.adapt(tf.data.Dataset.from_tensor_slices(txt_tr).batch(256))

    model = build_cnn(vec, emb_dim=128, n_outputs=len(TARGET_LABELS), seq_len=600)

    y_tr = df_tr[TARGET_LABELS].astype(int).values
    y_va = df_va[TARGET_LABELS].astype(int).values
    autotune = tf.data.AUTOTUNE
    def ds(texts, y):
        return tf.data.Dataset.from_tensor_slices((np.array(texts, dtype=object), y)).batch(BATCH).prefetch(autotune)

    model.fit(ds(df_tr["text"].fillna("").tolist(), y_tr),
              validation_data=ds(df_va["text"].fillna("").tolist(), y_va),
              epochs=EPOCHS, verbose=2)
    return model, vec


def occlusion_saliency_unigram_bigram(model, vec, vocab, texts, label_index, max_tokens=MAX_TOKS_PER_DOC):
    scores = defaultdict(list)
    for s in texts:
        base = float(model.predict(np.array([s], dtype=object), verbose=0)[0, label_index])
        seq = vec(np.array([s], dtype=object)).numpy()[0]
        toks = [vocab[i] for i in seq if i not in (0,1)][:max_tokens]
        if not toks: continue
        for i in range(len(toks)):
            mod = " ".join(toks[:i] + toks[i+1:])
            pr = float(model.predict(np.array([mod], dtype=object), verbose=0)[0, label_index])
            scores[toks[i]].append(max(0.0, base - pr))
        for i in range(len(toks)-1):
            bg = toks[i] + " " + toks[i+1]
            mod = " ".join(toks[:i] + toks[i+2:])
            pr = float(model.predict(np.array([mod], dtype=object), verbose=0)[0, label_index])
            scores[bg].append(max(0.0, base - pr))
    return {k: float(np.mean(v)) for k, v in scores.items() if v}

def ctakes_rf_top(df_tr, label, ctakes_text, max_feats=250000, min_df=3, n_estimators=600, top_k=TOP_K):
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True, min_df=min_df, max_features=max_feats)
    X = vec.fit_transform(ctakes_text.fillna(""))
    y = df_tr[label].astype(int).values
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RNG)
    rf.fit(X, y)
    names = np.array(vec.get_feature_names_out())
    imps = rf.feature_importances_
    order = np.argsort(imps)[::-1][:top_k]
    return list(zip(names[order], imps[order]))

def main():
    df, _ = load_df()
    df_tr, df_va, _ = split_by_patient(df)

    model, vec = fit_cnn(df_tr, df_va)
    vocab = vec.get_vocabulary()

    pos_texts = {}
    for idx, targ in enumerate(TARGET_LABELS):
        cand = df_tr[df_tr[targ] == 1]["text"].astype(str).tolist()
        if not cand:
            cand = df_tr["text"].astype(str).tolist()
        np.random.default_rng(RNG+idx).shuffle(cand)
        pos_texts[targ] = cand[:N_DOCS_PER_LABEL]

    rows = []
    for idx, targ in enumerate(TARGET_LABELS):
        sal = occlusion_saliency_unigram_bigram(model, vec, vocab, pos_texts[targ], idx, max_tokens=MAX_TOKS_PER_DOC)
        if sal:
            top = sorted(sal.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            for feat, score in top:
                rows.append({"Phenotype": targ, "Source": "CNN saliency (occlusion)", "Feature": feat, "Score": float(score)})

    full_all, filt_all = build_ctakes_text_columns(df)
    cols = ["hadm_id","subject.id","ctakes_text"]
    df_tr_ct = df_tr.merge(filt_all[cols], on=["hadm_id","subject.id"], how="left")
    ct_text = df_tr_ct["ctakes_text"]

    for targ in TARGET_LABELS:
        for feat, score in ctakes_rf_top(df_tr_ct, targ, ct_text):
            rows.append({"Phenotype": targ, "Source": "cTAKES filtered phrases (RF)", "Feature": feat, "Score": float(score)})

    out = pd.DataFrame(rows)


    out = out[out["Phenotype"].isin(TARGET_LABELS)]
    out = out[out["Source"].isin(["CNN saliency (occlusion)", "cTAKES filtered phrases (RF)"])]

    out_path = OUT / "table3_salient_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out.groupby(["Phenotype","Source"]).head(5))

if __name__ == "__main__":
    main()

