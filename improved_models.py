import re, math, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score

RNG = 42
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA = Path("annotations_with_text.csv")



def load_df() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA)
    if "text" not in df.columns:
        raise ValueError("annotations_with_text.csv must contain a 'text' column.")
    if "subject.id" not in df.columns:
        df["subject.id"] = df.get("hadm_id", pd.RangeIndex(len(df)))
    protected = {"hadm_id","subject.id","chart.time","text"}
    label_cols = [c for c in df.columns
                  if c not in protected and df[c].dropna().isin([0,1]).all()]
    if not label_cols:
        raise ValueError("No binary label columns found.")
    return df, label_cols

_num_re = re.compile(r"\d+")
_space_re = re.compile(r"\s+")
def clean_text(s: str, max_tokens: int = 4000) -> str:
    if not isinstance(s, str): s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = _num_re.sub("0", s)
    s = s.replace("\x00"," ")
    s = _space_re.sub(" ", s).strip()
    if not s: return s
    toks = s.split(" ")
    if len(toks) > max_tokens: toks = toks[:max_tokens]
    return " ".join(toks)



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



def fit_eval_tfidf_cv(df: pd.DataFrame, labels: List[str], ngram=(1,1),
                      max_features=500_000, min_df=5, C=1.0) -> Dict[int, Dict[str,float]]:
    groups = df["subject.id"].astype(str).values
    gkf = GroupKFold(n_splits=5)
    J = len(labels)
    scores = defaultdict(lambda: {"P":[], "R":[], "F1":[], "AUC":[]})

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        df_tr_all, df_te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RNG+fold)
        tr_i, va_i = next(gss.split(df_tr_all, groups=df_tr_all["subject.id"].astype(str).values))
        df_tr, df_va = df_tr_all.iloc[tr_i].reset_index(drop=True), df_tr_all.iloc[va_i].reset_index(drop=True)

        vec = TfidfVectorizer(ngram_range=ngram, strip_accents="unicode",
                              lowercase=True, min_df=min_df, max_features=max_features)
        Xtr = vec.fit_transform(df_tr["clean_text"])
        Xva = vec.transform(df_va["clean_text"])
        Xte = vec.transform(df_te["clean_text"])

        ytr = df_tr[labels].astype(int).values
        yva = df_va[labels].astype(int).values
        yte = df_te[labels].astype(int).values

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

def to_table(models_results: Dict[str, Dict[int, Dict[str,float]]], label_cols: List[str]) -> pd.DataFrame:
    rows = []
    models = ["CNN","BoW","n-gram","cTAKES full","cTAKES filter"]
    for j, lab in enumerate(label_cols):
        row = {"Phenotype": lab}
        for m in models:
            res = models_results.get(m, {})
            rj = res.get(j)
            if rj is None:
                row[f"{m}_P"] = row[f"{m}_R"] = row[f"{m}_F1"] = row[f"{m}_AUC"] = np.nan
            else:
                row[f"{m}_P"]   = 100*rj["P"]
                row[f"{m}_R"]   = 100*rj["R"]
                row[f"{m}_F1"]  = 100*rj["F1"]
                row[f"{m}_AUC"] = 100*rj["AUC"]
        rows.append(row)
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
                    v = r[f"{m}_{k}"]
                    cells.append("--" if np.isnan(v) else f"{v:.0f}")
            f.write(f"{r['Phenotype']} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")



def main():
    df, label_cols = load_df()
    df["clean_text"] = df["text"].astype(str).map(lambda s: clean_text(s, max_tokens=4000))

    results = {}

    print("5-fold CV: BoW (1-gram)")
    bow = fit_eval_tfidf_cv(df, label_cols, ngram=(1,1), max_features=500_000, min_df=5, C=1.0)
    results["BoW"] = bow

    print("5-fold CV: n-gram (1–2)")
    ngram = fit_eval_tfidf_cv(df, label_cols, ngram=(1,2), max_features=500_000, min_df=5, C=1.0)
    results["n-gram"] = ngram


    results["CNN"] = {}
    results["cTAKES full"] = {}
    results["cTAKES filter"] = {}

    table_bow = to_table({"BoW": bow}, label_cols)
    table_ng  = to_table({"n-gram": ngram}, label_cols)
    table_all = to_table(results, label_cols)

    table_bow.to_csv(OUT_DIR/"table2_cv_bow.csv", index=False)
    table_ng.to_csv(OUT_DIR/"table2_cv_ngram.csv", index=False)
    table_all.to_csv(OUT_DIR/"table2_cv_partial.csv", index=False)

    export_latex(table_all, OUT_DIR/"table2_cv_partial.tex")

    print(f"Saved:\n- {OUT_DIR/'table2_cv_bow.csv'}\n- {OUT_DIR/'table2_cv_ngram.csv'}\n- {OUT_DIR/'table2_cv_partial.csv'}\n- {OUT_DIR/'table2_cv_partial.tex'}")
    print("\nPreview:\n", table_all.head())

if __name__ == "__main__":
    main()
