import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ftfg.manifests import write_manifest

SEC_BASE = "https://data.sec.gov"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CONCEPT_MAP = {
    "at": ["Assets"],
    "lt": ["Liabilities"],
    "ceq": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "sale": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "oiadp": ["OperatingIncomeLoss"],
    "ib": ["NetIncomeLoss", "ProfitLoss"],
    "oancf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capx": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "che": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsAndShortTermInvestments"],
    "xrd": ["ResearchAndDevelopmentExpense"],
    "xsga": ["SellingGeneralAndAdministrativeExpense"],
    "dlc": ["ShortTermBorrowings", "ShortTermDebtCurrent"],
    "dltt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}


def user_agent() -> str:
    return os.getenv("FTFG_SEC_USER_AGENT", "filing-to-factor-gap research contact@example.com")


def sec_get_json(url: str, sleep: float = 0.12) -> dict[str, Any]:
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"}
    response = requests.get(url, headers=headers, timeout=30)
    time.sleep(sleep)
    response.raise_for_status()
    return response.json()


def fetch_submission(cik: int | str) -> dict[str, Any]:
    cik10 = str(cik).lstrip("0").zfill(10)
    return sec_get_json(f"{SEC_BASE}/submissions/CIK{cik10}.json")


def fetch_companyfacts(cik: int | str) -> dict[str, Any]:
    cik10 = str(cik).lstrip("0").zfill(10)
    return sec_get_json(COMPANYFACTS.format(cik=cik10))


def submissions_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    return pd.DataFrame(recent)


def fetch_submissions_for_ciks(ciks: list[int | str], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for cik in ciks:
        payload = fetch_submission(cik)
        raw_path = out_dir / f"CIK{str(cik).lstrip('0').zfill(10)}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        frame = submissions_to_frame(payload)
        if not frame.empty:
            frame["cik"] = int(str(cik).lstrip("0"))
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result.to_parquet(out_dir / "sec_submissions.parquet", index=False)
    write_manifest(out_dir / "sec_submissions_manifest.json", {"kind": "sec_submissions", "n_ciks": len(ciks), "n_rows": int(len(result))})
    return result


def fetch_sec_companyfacts_for_companies(
    company_path: str | Path,
    out_dir: str | Path,
    start: str,
    end: str,
    max_ciks: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    company = pd.read_parquet(company_path)
    company = company.dropna(subset=["cik"]).copy()
    company = _restrict_to_linked_sample(company, Path(company_path).parents[1])
    company["cik"] = pd.to_numeric(company["cik"], errors="coerce").astype("Int64")
    company = company.dropna(subset=["cik"]).drop_duplicates("cik")
    if max_ciks:
        company = company.head(max_ciks)
    out = Path(out_dir)
    facts_dir = out / "companyfacts"
    sub_dir = out / "submissions"
    parsed_dir = out / "parsed"
    for path in [facts_dir, sub_dir, parsed_dir]:
        path.mkdir(parents=True, exist_ok=True)

    parsed_paths = []
    for row in company.itertuples(index=False):
        cik = int(row.cik)
        cik10 = str(cik).zfill(10)
        parsed_path = parsed_dir / f"CIK{cik10}.parquet"
        if parsed_path.exists() and not force:
            parsed_paths.append(parsed_path)
            continue
        try:
            facts_payload = fetch_companyfacts(cik)
            submissions_payload = fetch_submission(cik)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {401, 403, 429}:
                raise RuntimeError(f"SEC access blocked or rate limited at CIK {cik}; stop and retry later") from exc
            continue
        except requests.RequestException:
            continue
        (facts_dir / f"CIK{cik10}.json").write_text(json.dumps(facts_payload), encoding="utf-8")
        (sub_dir / f"CIK{cik10}.json").write_text(json.dumps(submissions_payload), encoding="utf-8")
        submissions = _parse_submissions_all(submissions_payload)
        parsed = companyfacts_to_asfiled_frame(facts_payload, submissions, row._asdict(), start, end)
        if not parsed.empty:
            parsed.to_parquet(parsed_path, index=False)
            parsed_paths.append(parsed_path)
    frames = [pd.read_parquet(path) for path in parsed_paths]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result.to_parquet(out / "sec_companyfacts_asfiled.parquet", index=False)
    write_manifest(
        out / "sec_companyfacts_manifest.json",
        {
            "kind": "sec_companyfacts",
            "n_ciks_requested": int(len(company)),
            "n_ciks_parsed": int(len(parsed_paths)),
            "n_rows": int(len(result)),
            "start": start,
            "end": end,
        },
    )
    return result


def _restrict_to_linked_sample(company: pd.DataFrame, wrds_root: Path) -> pd.DataFrame:
    ccm_path = wrds_root / "ccm" / "ccm_linktable.parquet"
    comp_path = wrds_root / "compustat" / "compustat_funda.parquet"
    crsp_dir = wrds_root / "crsp"
    eligible: set[str] = set()
    if ccm_path.exists():
        ccm = pd.read_parquet(ccm_path, columns=["gvkey"])
        eligible |= set(ccm["gvkey"].astype(str).str.zfill(6))
    if comp_path.exists():
        comp = pd.read_parquet(comp_path, columns=["gvkey"])
        eligible |= set(comp["gvkey"].astype(str).str.zfill(6))
    crsp_files = sorted(crsp_dir.glob("crsp_monthly_*.parquet"))
    if ccm_path.exists() and crsp_files:
        permnos: set[int] = set()
        for path in crsp_files:
            crsp = pd.read_parquet(path, columns=["permno"])
            permnos.update(pd.to_numeric(crsp["permno"], errors="coerce").dropna().astype(int).unique().tolist())
        ccm = pd.read_parquet(ccm_path, columns=["gvkey", "lpermno"]).dropna(subset=["lpermno"])
        ccm["lpermno"] = pd.to_numeric(ccm["lpermno"], errors="coerce")
        ccm = ccm[ccm["lpermno"].isin(permnos)]
        eligible = set(ccm["gvkey"].astype(str).str.zfill(6))
        if comp_path.exists():
            comp = pd.read_parquet(comp_path, columns=["gvkey"])
            eligible &= set(comp["gvkey"].astype(str).str.zfill(6))
    if not eligible:
        return company
    out = company.copy()
    out["gvkey"] = out["gvkey"].astype(str).str.zfill(6)
    return out[out["gvkey"].isin(eligible)]


def companyfacts_to_asfiled_frame(
    payload: dict[str, Any],
    submissions: pd.DataFrame,
    company_row: dict[str, Any],
    start: str,
    end: str,
) -> pd.DataFrame:
    concept_values: dict[tuple[str, str], dict[str, Any]] = {}
    facts = payload.get("facts", {}).get("us-gaap", {})
    cik = int(payload.get("cik") or company_row.get("cik"))
    for output_col, concepts in CONCEPT_MAP.items():
        for concept in concepts:
            node = facts.get(concept)
            if not node:
                continue
            units = node.get("units", {})
            unit_key = _choose_unit(units)
            if unit_key is None:
                continue
            for fact in units.get(unit_key, []):
                form = str(fact.get("form", ""))
                if form not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                    continue
                filed = fact.get("filed")
                end_date = fact.get("end")
                accn = fact.get("accn")
                if not filed or not end_date or not accn:
                    continue
                if filed < start or filed > end:
                    continue
                key = (accn, end_date)
                rec = concept_values.setdefault(
                    key,
                    {
                        "cik": cik,
                        "gvkey": str(company_row.get("gvkey", "")).zfill(6),
                        "sic": company_row.get("sic"),
                        "name": company_row.get("conm"),
                        "adsh": accn,
                        "form": form,
                        "fdate": filed,
                        "datadate": end_date,
                        "fy": fact.get("fy"),
                        "fp": fact.get("fp"),
                        "source": "sec_companyfacts",
                        "sec_concept_count": 0,
                    },
                )
                if pd.isna(rec.get(output_col)):
                    rec[output_col] = fact.get("val")
                    rec[f"{output_col}_concept"] = concept
                rec["sec_concept_count"] += 1
    if not concept_values:
        return pd.DataFrame()
    out = pd.DataFrame(concept_values.values())
    if not submissions.empty:
        out = out.merge(submissions, on="adsh", how="left")
    out["fdate"] = pd.to_datetime(out["fdate"])
    out["datadate"] = pd.to_datetime(out["datadate"])
    out = out[(out["fdate"] >= pd.Timestamp(start)) & (out["fdate"] <= pd.Timestamp(end))]
    numeric = [*CONCEPT_MAP.keys(), "sec_concept_count"]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["act", "ap", "ch", "dp", "invt", "lct", "ppent", "seq", "xint", "pi", "txt", "dv", "fincf", "prstkc", "sstk", "xad"]:
        if col not in out:
            out[col] = np.nan
    core_cols = [c for c in CONCEPT_MAP if c in out]
    out["nonmissing_core_facts"] = out[core_cols].notna().sum(axis=1)
    out = (
        out.sort_values(["adsh", "nonmissing_core_facts", "datadate"], ascending=[True, False, False])
        .drop_duplicates("adsh", keep="first")
        .drop(columns=["nonmissing_core_facts"])
    )
    return out


def _choose_unit(units: dict[str, list[dict[str, Any]]]) -> str | None:
    for candidate in ["USD", "shares", "pure"]:
        if candidate in units:
            return candidate
    return next(iter(units.keys()), None) if units else None


def _parse_submissions_all(payload: dict[str, Any]) -> pd.DataFrame:
    frames = [submissions_to_frame(payload)]
    for file_info in payload.get("filings", {}).get("files", []):
        name = file_info.get("name")
        if not name:
            continue
        try:
            frames.append(pd.DataFrame(sec_get_json(f"{SEC_BASE}/submissions/{name}")))
        except requests.RequestException:
            continue
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["adsh", "acceptance_datetime", "is_xbrl", "is_inline_xbrl"])
    sub = pd.concat(frames, ignore_index=True)
    rename = {
        "accessionNumber": "adsh",
        "acceptanceDateTime": "acceptance_datetime",
        "isXBRL": "is_xbrl",
        "isInlineXBRL": "is_inline_xbrl",
    }
    keep = [c for c in rename if c in sub.columns]
    sub = sub[keep].rename(columns=rename).drop_duplicates("adsh")
    return sub
