from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ftfg.features.gap_builder import build_gap_features
from ftfg.features.liquidity import build_liquidity_controls
from ftfg.manifests import write_manifest


def build_real_analysis_panel(data_root: str | Path, manifest_dir: str | Path) -> Path:
    root = Path(data_root)
    out_path = root / "processed" / "analysis_panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    as_filed = _read_many(root / "raw" / "wrds" / "as_filed" / "funda_asfiled_*.parquet")
    sec_facts = _read_one(root / "raw" / "sec" / "sec_companyfacts_asfiled.parquet")
    if not sec_facts.empty:
        as_filed = pd.concat([as_filed, sec_facts], ignore_index=True, sort=False)
    structure = _read_many(root / "raw" / "wrds" / "as_filed" / "structure_asfiled_*.parquet")
    ccm = _read_one(root / "raw" / "wrds" / "ccm" / "ccm_linktable.parquet")
    crsp_m = _read_many(root / "raw" / "wrds" / "crsp" / "crsp_monthly_*.parquet")
    crsp_d = _read_many(root / "raw" / "wrds" / "crsp" / "crsp_daily_*.parquet")
    comp = _read_one(root / "raw" / "wrds" / "compustat" / "compustat_funda.parquet")
    ibes = _read_one(root / "raw" / "wrds" / "ibes" / "statsum_epsus.parquet")

    if as_filed.empty or ccm.empty or crsp_m.empty or comp.empty:
        missing = {
            "as_filed": as_filed.empty,
            "ccm": ccm.empty,
            "crsp_monthly": crsp_m.empty,
            "compustat": comp.empty,
        }
        raise FileNotFoundError(f"Required private extracts missing: {missing}")

    panel = _prepare_as_filed(as_filed)
    panel = _merge_structure(panel, structure)
    panel = _link_permno(panel, ccm)
    panel = _merge_stale_compustat(panel, comp)
    panel = _merge_monthly_returns(panel, crsp_m)
    if not crsp_d.empty:
        panel = _merge_event_returns(panel, crsp_d)
    if not ibes.empty:
        panel = _merge_ibes(panel, ibes)
    else:
        panel["analyst_coverage"] = np.nan

    panel = build_gap_features(panel)
    panel = build_liquidity_controls(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = _coerce_parquet_safe(panel)
    panel.to_parquet(out_path, index=False)
    write_manifest(
        Path(manifest_dir) / "real_analysis_panel_manifest.json",
        {
            "kind": "real_analysis_panel",
            "rows": int(len(panel)),
            "firms": int(panel["permno"].nunique()),
            "filings": int(panel["adsh"].nunique()) if "adsh" in panel else int(len(panel)),
            "columns": list(panel.columns),
            "date_min": str(panel["fdate"].min()),
            "date_max": str(panel["fdate"].max()),
        },
    )
    return out_path


def _read_many(pattern: Path) -> pd.DataFrame:
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _read_one(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _coerce_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    string_cols = {
        "adsh",
        "cik",
        "cusip",
        "cusip8",
        "form",
        "gvkey",
        "linkprim",
        "linktype",
        "name",
        "ncusip",
        "source",
        "ticker",
    }
    for col in out.select_dtypes(include=["object"]).columns:
        if col in string_cols or col.endswith("_concept"):
            out[col] = out[col].astype("string")
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        nonmissing = int(out[col].notna().sum())
        if nonmissing and int(numeric.notna().sum()) >= max(1, int(nonmissing * 0.95)):
            out[col] = numeric
        else:
            out[col] = out[col].astype("string")
    return out


def _prepare_as_filed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "at",
        "sale",
        "cogs",
        "oiadp",
        "ib",
        "oancf",
        "capx",
        "dlc",
        "dltt",
        "che",
        "xrd",
        "xsga",
        "sstk",
        "fincf",
        "sic",
    ]
    for col in required:
        if col not in out:
            out[col] = np.nan
    out["fdate"] = pd.to_datetime(out["fdate"])
    out["datadate"] = pd.to_datetime(out["datadate"])
    out["gvkey"] = out["gvkey"].astype(str).str.zfill(6)
    out = out.sort_values(["gvkey", "fdate", "datadate"])
    assets = out["at"].replace(0, np.nan)
    lag_assets = out.groupby("gvkey")["at"].shift(1).replace(0, np.nan)
    debt = out[["dlc", "dltt"]].fillna(0).sum(axis=1)
    out["profitability_af"] = out["oiadp"] / assets
    out["gross_margin_af"] = (out["sale"] - out["cogs"]) / assets
    out["cfo_assets_af"] = out["oancf"] / assets
    out["investment_af"] = (out["at"] - lag_assets) / lag_assets
    out["accruals_af"] = (out["ib"] - out["oancf"]) / assets
    out["capx_assets_af"] = out["capx"] / assets
    out["leverage_af"] = debt / assets
    out["cash_assets_af"] = out["che"] / assets
    out["rd_assets_af"] = out["xrd"] / assets
    out["sga_assets_af"] = out["xsga"] / assets
    out["equity_issuance_af"] = out["sstk"] / assets
    out["debt_issuance_af"] = out["fincf"] / assets
    out["sic2"] = pd.to_numeric(out["sic"], errors="coerce").floordiv(100)
    return out


def _merge_structure(panel: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    if structure.empty:
        panel["complexity_score"] = np.nan
        return panel
    s = structure.copy()
    s["gvkey"] = s["gvkey"].astype(str).str.zfill(6)
    s["fdate"] = pd.to_datetime(s["fdate"])
    keys = ["gvkey", "adsh"]
    use_cols = keys + [c for c in s.columns if c.startswith(("ntag_", "level_", "dq_")) or c == "d_amend"]
    out = panel.merge(s[use_cols].drop_duplicates(keys), on=keys, how="left")
    tag_cols = [c for c in out.columns if c.startswith("ntag_")]
    level_cols = [c for c in out.columns if c.startswith("level_")]
    total_tags = out[tag_cols].sum(axis=1, min_count=1)
    if "sec_concept_count" in out:
        total_tags = total_tags.fillna(out["sec_concept_count"])
    note_cols = [c for c in tag_cols if c.endswith("_note")]
    stmt_cols = [c for c in tag_cols if c.endswith("_stmt")]
    note_share = out[note_cols].sum(axis=1, min_count=1) / total_tags.replace(0, np.nan) if note_cols else pd.Series(np.nan, index=out.index)
    stmt_share = out[stmt_cols].sum(axis=1, min_count=1) / total_tags.replace(0, np.nan) if stmt_cols else pd.Series(np.nan, index=out.index)
    level_avg = out[level_cols].mean(axis=1) if level_cols else pd.Series(np.nan, index=out.index)
    dq_cols = [c for c in out.columns if c.startswith("dq_")]
    dq = out[dq_cols].mean(axis=1) if dq_cols else pd.Series(np.nan, index=out.index)
    out["custom_tag_share"] = 1 - stmt_share
    out["missingness_burden"] = out[["at", "sale", "cogs", "oancf", "capx", "ceq"]].isna().mean(axis=1)
    out["mapping_entropy"] = np.log1p(total_tags.fillna(0)) * (1 + note_share.fillna(0))
    out["articulation_error"] = (out["at"] - out[["lt", "ceq"]].sum(axis=1)).abs() / out["at"].abs().replace(0, np.nan)
    out["amendment_flag"] = out["form"].astype(str).str.endswith("/A") | out.get("d_amend", 0).fillna(0).astype(bool)
    raw = pd.concat(
        [
            out["custom_tag_share"].rank(pct=True),
            out["missingness_burden"].rank(pct=True),
            out["mapping_entropy"].rank(pct=True),
            out["articulation_error"].clip(upper=1).rank(pct=True),
            level_avg.rank(pct=True),
            (1 - dq).rank(pct=True),
            out["amendment_flag"].astype(float),
        ],
        axis=1,
    )
    out["complexity_score"] = raw.mean(axis=1)
    fallback = pd.concat(
        [
            out["missingness_burden"].rank(pct=True),
            pd.to_numeric(out.get("sec_concept_count", pd.Series(np.nan, index=out.index)), errors="coerce").rank(pct=True),
            out["amendment_flag"].astype(float),
        ],
        axis=1,
    ).mean(axis=1)
    out["complexity_score"] = out["complexity_score"].fillna(fallback)
    out["complexity_score"] = out["complexity_score"].fillna(out["missingness_burden"]).fillna(0.5)
    return out


def _link_permno(panel: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    links = ccm.copy()
    links["gvkey"] = links["gvkey"].astype(str).str.zfill(6)
    links["linkdt"] = pd.to_datetime(links["linkdt"])
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(pd.Timestamp("2100-01-01"))
    out = panel.merge(links, on="gvkey", how="left")
    out = out[(out["fdate"] >= out["linkdt"]) & (out["fdate"] <= out["linkenddt"])]
    out = out.rename(columns={"lpermno": "permno", "lpermco": "permco"})
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out.dropna(subset=["permno"]).copy()


def _merge_stale_compustat(panel: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    c = comp.copy()
    c["gvkey"] = c["gvkey"].astype(str).str.zfill(6)
    c["datadate"] = pd.to_datetime(c["datadate"])
    for col in ["fdate", "pdate"]:
        if col in c:
            c[col] = pd.to_datetime(c[col])
    c["public_date"] = c.get("pdate").fillna(c.get("fdate")).fillna(c["datadate"] + pd.Timedelta(days=180))
    c = c.sort_values(["gvkey", "public_date"])
    assets = c["at"].replace(0, np.nan)
    lag_assets = c.groupby("gvkey")["at"].shift(1).replace(0, np.nan)
    c["stale_profitability"] = c["oiadp"] / assets
    c["stale_investment"] = (c["at"] - lag_assets) / lag_assets
    c["stale_accruals"] = (c["ib"] - c["oancf"]) / assets
    c["stale_leverage"] = c[["dlc", "dltt"]].fillna(0).sum(axis=1) / assets
    keep = ["gvkey", "public_date", "stale_profitability", "stale_investment", "stale_accruals", "stale_leverage"]
    comp_by_gvkey = {gvkey: g[keep].sort_values("public_date") for gvkey, g in c.groupby("gvkey", sort=False)}
    pieces = []
    for gvkey, g in panel.sort_values(["gvkey", "fdate"]).groupby("gvkey", sort=False):
        cg = comp_by_gvkey.get(gvkey)
        if cg is None or cg.empty:
            pieces.append(g)
            continue
        merged = pd.merge_asof(
            g.sort_values("fdate"),
            cg.sort_values("public_date"),
            left_on="fdate",
            right_on="public_date",
            by="gvkey",
            direction="backward",
            allow_exact_matches=False,
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _merge_monthly_returns(panel: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    m = crsp.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["permno"] = pd.to_numeric(m["permno"], errors="coerce").astype("Int64")
    for col in ["ret", "prc", "shrout", "vol"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m["market_equity"] = m["prc"].abs() * m["shrout"] * 1000
    m["dollar_volume"] = m["prc"].abs() * m["vol"] * 100
    m["signal_month"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
    controls = m[["permno", "signal_month", "market_equity", "dollar_volume", "prc", "shrout", "vol", "ncusip", "ticker", "siccd"]].drop_duplicates(["permno", "signal_month"])
    returns = m[["permno", "signal_month", "ret"]].rename(columns={"signal_month": "return_month", "ret": "ret_fwd_1m"})
    out = panel.copy()
    out["signal_month"] = out["fdate"].dt.to_period("M").dt.to_timestamp("M")
    out["return_month"] = out["signal_month"] + pd.offsets.MonthEnd(1)
    out = out.merge(controls, on=["permno", "signal_month"], how="left")
    out = out.merge(returns, on=["permno", "return_month"], how="left")
    out["date"] = out["signal_month"]
    out["sic2"] = out["sic2"].fillna(pd.to_numeric(out["siccd"], errors="coerce").floordiv(100))
    return out


def _merge_event_returns(panel: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["permno"] = pd.to_numeric(d["permno"], errors="coerce").astype("Int64")
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0)
    d = d.sort_values(["permno", "date"])
    out = panel.copy()
    for horizon in [5, 20, 60]:
        out[f"ret_{horizon}d"] = np.nan
    daily_by_permno = {
        permno: (
            gd["date"].to_numpy(dtype="datetime64[ns]"),
            (1 + gd["ret"].to_numpy()).cumprod(),
        )
        for permno, gd in d.groupby("permno", sort=False)
    }
    for permno, idx in out.groupby("permno").groups.items():
        daily_tuple = daily_by_permno.get(permno)
        if daily_tuple is None:
            continue
        dates, gross = daily_tuple
        event_dates = out.loc[idx, "fdate"].to_numpy(dtype="datetime64[ns]")
        starts = np.searchsorted(dates, event_dates, side="right")
        for horizon in [5, 20, 60]:
            ends = starts + horizon - 1
            ok = ends < len(gross)
            vals = np.full(len(starts), np.nan)
            start_gross = np.where(starts > 0, gross[np.maximum(starts - 1, 0)], 1.0)
            vals[ok] = gross[ends[ok]] / start_gross[ok] - 1
            out.loc[idx, f"ret_{horizon}d"] = vals
    return out


def _merge_ibes(panel: pd.DataFrame, ibes: pd.DataFrame) -> pd.DataFrame:
    i = ibes.copy()
    i["statpers"] = pd.to_datetime(i["statpers"])
    i["cusip8"] = i["cusip"].astype(str).str[:8]
    i = i.sort_values(["cusip8", "statpers"])
    out = panel.copy()
    out["cusip8"] = out["ncusip"].astype(str).str[:8]
    pieces = []
    keep = ["cusip8", "statpers", "numest", "stdev", "meanest"]
    ibes_by_cusip = {cusip: g[keep].sort_values("statpers") for cusip, g in i.groupby("cusip8", sort=False)}
    for cusip, g in out.sort_values(["cusip8", "fdate"]).groupby("cusip8", sort=False):
        ig = ibes_by_cusip.get(cusip)
        if ig is None or ig.empty:
            pieces.append(g)
            continue
        merged = pd.merge_asof(
            g.sort_values("fdate"),
            ig.sort_values("statpers"),
            left_on="fdate",
            right_on="statpers",
            by="cusip8",
            direction="backward",
            allow_exact_matches=True,
        )
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True)
    out = out.rename(columns={"numest": "analyst_coverage", "stdev": "analyst_dispersion"})
    return out
