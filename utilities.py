"""Utilities for use with the 2023 NeuroHackademy data showcase.
"""

from io import StringIO
import numpy as np
import pandas as pd

def ls(path):
    "Lists the contents of the given path."
    # If path is not a directory, raise an error:
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory")
    else:
        return list(path.iterdir())


def crawl(path, indent=0):
    "Prints a nested tree of the contents of the given path."
    print((' '*indent) + path.name)
    if path.is_dir():
        for subpath in path.iterdir():
            crawl(subpath, indent=(indent + 3))
    else:
        pass


def load_aws_credentials(profile_name):
    "Returns (access_key, secred_key) from ~/.aws/credentials for the given profile."
    import boto3
    ses = boto3.Session(profile_name=profile_name)
    creds = ses.get_credentials()
    return (creds.access_key, creds.secret_key)




def dataframe_brief(
    df: pd.DataFrame,
    name: str = "df",
    sample_rows: int = 8,
    top_n_values: int = 5,
    corr_top_pairs: int = 10,
    max_text_len: int = 80,
    include_describe: bool = True,
) -> str:
    """
    Return a compact text summary of a dataframe for chat/LLM analysis.
    """
    out = StringIO()

    def w(line=""):
        out.write(str(line) + "\n")

    n_rows, n_cols = df.shape
    w(f"DATAFRAME: {name}")
    w(f"SHAPE: rows={n_rows}, cols={n_cols}")

    # Column dtype overview
    dtype_counts = df.dtypes.astype(str).value_counts()
    w("\nDTYPES (count):")
    for dt, ct in dtype_counts.items():
        w(f"- {dt}: {ct}")

    # Per-column profile
    w("\nCOLUMN PROFILE:")
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        nulls = int(s.isna().sum())
        null_pct = (nulls / n_rows * 100) if n_rows else 0.0
        nunique = int(s.nunique(dropna=True))
        nunique_pct = (nunique / n_rows * 100) if n_rows else 0.0

        vc = (
            s.dropna()
            .astype(str)
            .str.slice(0, max_text_len)
            .value_counts()
            .head(top_n_values)
        )
        top_vals = "; ".join([f"{k} ({v})" for k, v in vc.items()]) if len(vc) else "None"

        extra = ""
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            if len(vals):
                q1, q2, q3 = vals.quantile([0.25, 0.5, 0.75]).tolist()
                extra = (
                    f" | min={vals.min():.4g}, q1={q1:.4g}, med={q2:.4g}, "
                    f"q3={q3:.4g}, max={vals.max():.4g}, mean={vals.mean():.4g}, std={vals.std():.4g}"
                )

        w(
            f"- {c}: dtype={dtype}, nulls={nulls} ({null_pct:.1f}%), "
            f"unique={nunique} ({nunique_pct:.1f}%)"
            f"{extra}\n  top_values: {top_vals}"
        )

    dup = int(df.duplicated().sum())
    dup_pct = (dup / n_rows * 100) if n_rows else 0.0
    w(f"\nDUPLICATE ROWS: {dup} ({dup_pct:.1f}%)")

    candidate_keys = [c for c in df.columns if n_rows > 0 and df[c].nunique(dropna=False) == n_rows]
    w("POTENTIAL UNIQUE ID COLUMNS:")
    if candidate_keys:
        for c in candidate_keys:
            w(f"- {c}")
    else:
        w("- None found")

    w("\nDATE-LIKE COLUMNS:")
    date_like_found = False
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_like_found = True
            w(f"- {c}: datetime dtype, min={s.min()}, max={s.max()}")
        elif s.dtype == object:
            parsed = pd.to_datetime(s, errors="coerce")
            ok = parsed.notna().mean() if len(parsed) else 0
            if ok >= 0.8:
                date_like_found = True
                w(f"- {c}: parseable datetime-like ({ok:.0%}), min={parsed.min()}, max={parsed.max()}")
    if not date_like_found:
        w("- None obvious")

    num = df.select_dtypes(include=[np.number])
    w("\nNUMERIC CORRELATION HIGHLIGHTS:")
    if num.shape[1] >= 2 and len(num) >= 3:
        corr = num.corr(numeric_only=True)
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append((abs(v), v, cols[i], cols[j]))
        pairs.sort(reverse=True, key=lambda x: x[0])
        if pairs:
            for _, v, a, b in pairs[:corr_top_pairs]:
                w(f"- {a} vs {b}: r={v:.3f}")
        else:
            w("- No valid pairs")
    else:
        w("- Not enough numeric columns/rows")

    if include_describe and n_cols > 0 and n_rows > 0:
        w("\nDESCRIBE (numeric):")
        desc = df.describe(include=[np.number]).transpose()
        if len(desc):
            w(desc.to_string(max_rows=200, max_cols=20))
        else:
            w("No numeric columns")

        w("\nDESCRIBE (categorical/object):")
        desc_obj = df.describe(include=["object", "category", "bool"]).transpose()
        if len(desc_obj):
            w(desc_obj.to_string(max_rows=200, max_cols=20))
        else:
            w("No categorical/object/bool columns")

    w(f"\nHEAD SAMPLE ({sample_rows} rows):")
    if n_rows > 0:
        sample = df.head(sample_rows).copy()
        for c in sample.columns:
            if sample[c].dtype == object:
                sample[c] = sample[c].astype(str).str.slice(0, max_text_len)
        w(sample.to_string(index=False, max_rows=sample_rows))
    else:
        w("Dataframe is empty")

    return out.getvalue()

