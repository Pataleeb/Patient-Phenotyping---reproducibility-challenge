import re, os, math, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score

RNG = 42
OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("annotations_with_text.csv")


def load_df() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA)
    if "text" not in df.columns:
        raise ValueError("contain a 'text' column.")
    if "subject.id" not in df.columns:
        df["subject.id"] = df.get("hadm_id", pd.RangeIndex(len(df)))
    protected = {"hadm_id","subject.id","chart.time","text","clean_text"}
    label_cols = [c for c in df.columns
                  if c not in protected and df[c].dropna().isin([0,1]).all()]
    if not label_cols:
        raise ValueError("No binary label")
    return df, label_cols


def get_sci_pipeline():
    try:
        import spacy
        nlp = None
        try:
            import scispacy

            try:
                nlp = spacy.load("en_core_sci_sm")
            except Exception:
                nlp = spacy.load("en_core_web_sm")

            try:
                nlp.add_pipe("negex", last=True)
            except Exception:
                pass
        except Exception:
            nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception:
        return None

def extract_phrases_spacy(texts: List[str], nlp, filtered: bool) -> List[str]:
    out = []
    for doc in nlp.pipe(texts, batch_size=32, disable=[]):
        phrases = []

        if doc.ents:
            for ent in doc.ents:
                txt = ent.text.strip().lower()
                if filtered and hasattr(ent, "_.is_negated") and ent._.is_negated:
                    continue
                if len(txt) >= 3:
                    phrases.append(txt)
        for nc in getattr(doc, "noun_chunks", []):
            txt = nc.text.strip().lower()
            if len(txt) >= 3:
                phrases.append(txt)
        if not phrases:
            phrases = [t.text.lower() for t in doc if t.is_alpha]
        out.append(" ; ".join(phrases))
    return out


_rx_word = re.compile(r"[a-z]+")
def extract_phrases_fallback(texts: List[str], filtered: bool) -> List[str]:
    out = []
    for s in texts:
        s = str(s).lower()

        toks = _rx_word.findall(s)
        if filtered:
            keep = []
            neg = False
            for t in toks:
                if t in {"no","denies","without"}:
                    neg = True
                    continue
                if neg:

                    neg = False
                    continue
                keep.append(t)
            toks = keep
        out.append(" ; ".join(toks))
    return out

def build_ctakes_text_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nlp = get_sci_pipeline()
    texts = df["text"].astype(str).tolist()
    if nlp is not None:
        full  = extract_phrases_spacy(texts, nlp, filtered=False)
        filt  = extract_phrases_spacy(texts, nlp, filtered=True)
    else:
        full  = extract_phrases_fallback(texts, filtered=False)
        filt  = extract_phrases_fallback(texts, filtered=True)
    df_full = df.copy(); df_full["ctakes_text"] = full
    df_filt = df.copy(); df_filt["ctakes_text"] = filt
    return df_full, df_filt


def tune_thresholds(y_true_val: np.ndarray, y_prob_val: np.ndarray) -> np.ndarray:
    J = y_true_val.shape[1]
    thr = np.full(J, 0.5, dtype=float)
    for j in range(J):
        yv = y_true_val[:, j]
        if len(np.unique(yv)) < 2:
            thr[j] = 0.5
            continue
        scores = y_prob_val[:, j]
        grid = np.quantile(scores, np.linspace(0.05, 0.95, 31))
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (scores >= t).astype(int)
            f1 = f1_score(yv, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[j] = best_t
    return thr

def per_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: np.ndarray) -> Dict[int, Dict[str,float]]:
    out = {}
    J = y_true.shape[1]
    for j in range(J):
        yt, yp = y_true[:, j], y_prob[:, j]
        if len(np.unique(yt)) < 2:
            out[j] = {"P": np.nan, "R": np.nan, "F1": np.nan, "AUC": np.nan}
            continue
        pred = (yp >= thr[j]).astype(int)
        P,R,F1,_ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
        try: AUC = roc_auc_score(yt, yp)
        except ValueError: AUC = np.nan
        out[j] = {"P": P, "R": R, "F1": F1, "AUC": AUC}
    return out

def fit_eval_ctakes_cv(df: pd.DataFrame, label_cols: List[str],
                       max_features=300_000, min_df=3, C=1.0) -> Dict[int, Dict[str,float]]:
    groups = df["subject.id"].astype(str).values
    gkf = GroupKFold(n_splits=5)
    J = len(label_cols)
    scores = defaultdict(lambda: {"P":[], "R":[], "F1":[], "AUC":[]})

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        df_tr_all, df_te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RNG+fold)
        tr_i, va_i = next(gss.split(df_tr_all, groups=df_tr_all["subject.id"].astype(str).values))
        df_tr, df_va = df_tr_all.iloc[tr_i].reset_index(drop=True), df_tr_all.iloc[va_i].reset_index(drop=True)

        vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word",
                              lowercase=True, min_df=min_df, max_features=max_features)
        Xtr = vec.fit_transform(df_tr["ctakes_text"])
        Xva = vec.transform(df_va["ctakes_text"])
        Xte = vec.transform(df_te["ctakes_text"])

        ytr = df_tr[label_cols].astype(int).values
        yva = df_va[label_cols].astype(int).values
        yte = df_te[label_cols].astype(int).values

        base = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=C,
            tol=1e-3,
            max_iter=3000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG+fold
        )
        clf = OneVsRestClassifier(base, n_jobs=-1).fit(Xtr, ytr)

        try:
            p_va = clf.decision_function(Xva)
            p_te = clf.decision_function(Xte)
        except Exception:
            p_va = clf.predict_proba(Xva)
            p_te = clf.predict_proba(Xte)

        thr = tune_thresholds(yva, p_va)
        fold_metrics = per_label_metrics(yte, p_te, thr)
        for j in range(J):
            for k in ["P","R","F1","AUC"]:
                scores[j][k].append(fold_metrics[j][k])

    agg = {}
    for j in range(J):
        agg[j] = {k: np.nanmean(scores[j][k]) for k in ["P","R","F1","AUC"]}
    return agg

def dict_to_table_block(res: Dict[int, Dict[str,float]], label_cols: List[str], prefix: str) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(label_cols):
        rj = res.get(j, None)
        P=R=F=A=np.nan if rj is None else (100*rj["P"],100*rj["R"],100*rj["F1"],100*rj["AUC"])
        rows.append({"Phenotype": lab,
                     f"{prefix}_P":P, f"{prefix}_R":R, f"{prefix}_F1":F, f"{prefix}_AUC":A})
    return pd.DataFrame(rows)

def export_latex(df: pd.DataFrame, path: Path):
    models = ["CNN","BoW","n-gram","cTAKES full","cTAKES filter"]
    with open(path, "w") as f:
        f.write("\\begin{tabular}{l" + "rrrr"*len(models) + "}\n\\hline\n")
        f.write(" & " + " & ".join([f"\\multicolumn{{4}}{{c}}{{{m}}}" for m in models]) + " \\\\\n")
        f.write("Phenotype & " + " & ".join(["P & R & F1 & AUC"]*len(models)) + " \\\\\n\\hline\n")
        for _, r in df.iterrows():
            cells = []
            for m in models:
                for k in ["P","R","F1","AUC"]:
                    v = r.get(f"{m}_{k}", np.nan)
                    cells.append("--" if (v is None or np.isnan(v)) else f"{v:.0f}")
            f.write(f"{r['Phenotype']} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

def try_merge_existing(b: pd.DataFrame) -> pd.DataFrame:

    paths = [("BoW", OUT/"table2_cv_bow.csv"), ("n-gram", OUT/"table2_cv_ngram.csv")]
    M = b.copy()
    for name, p in paths:
        if p.exists():
            t = pd.read_csv(p)
            cols = [c for c in t.columns if c.startswith(name+"_")]
            M = M.merge(t[["Phenotype"]+cols], on="Phenotype", how="left")

    for k in ["P","R","F1","AUC"]:
        M[f"CNN_{k}"] = np.nan

    order = ["Phenotype"] + \
            [f"CNN_{k}" for k in ["P","R","F1","AUC"]] + \
            [f"BoW_{k}" for k in ["P","R","F1","AUC"]] + \
            [f"n-gram_{k}" for k in ["P","R","F1","AUC"]] + \
            [f"cTAKES full_{k}" for k in ["P","R","F1","AUC"]] + \
            [f"cTAKES filter_{k}" for k in ["P","R","F1","AUC"]]
    return M[order]

def main():
    df, labels = load_df()


    df_full, df_filt = build_ctakes_text_columns(df)

    # CV
    print("5-fold CV: cTAKES full")
    res_full = fit_eval_ctakes_cv(df_full, labels, max_features=300_000, min_df=3, C=1.0)
    print("5-fold CV: cTAKES filter (negations removed)")
    res_filt = fit_eval_ctakes_cv(df_filt, labels, max_features=300_000, min_df=3, C=1.0)

    t_full  = dict_to_table_block(res_full, labels, "cTAKES full")
    t_filt  = dict_to_table_block(res_filt, labels, "cTAKES filter")
    base = t_full.merge(t_filt, on="Phenotype", how="outer")

    base.to_csv(OUT/"table2_cv_ctakes_full.csv", index=False)
    pd.DataFrame(t_filt).to_csv(OUT/"table2_cv_ctakes_filter.csv", index=False)

    merged = try_merge_existing(base)
    merged.to_csv(OUT/"table2_cv_merge.csv", index=False)
    export_latex(merged, OUT/"table2_cv_merge.tex")

    print(f"Saved:\n- {OUT/'table2_cv_ctakes_full.csv'}\n- {OUT/'table2_cv_ctakes_filter.csv'}\n- {OUT/'table2_cv_merge.csv'}\n- {OUT/'table2_cv_merge.tex'}")
    print("\nPreview:\n", merged.head())

if __name__ == "__main__":
    main()
