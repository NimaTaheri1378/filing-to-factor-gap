from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ftfg.manifests import write_manifest

LOG = logging.getLogger(__name__)


def _connect_wrds():
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the wrds extra to run WRDS extracts: pip install -e '.[wrds]'") from exc
    return wrds.Connection()


def wrds_smoke(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = _connect_wrds()
    libs = db.list_libraries()
    smoke = db.raw_sql("select current_date as query_date")
    path = out / "wrds_smoke.csv"
    smoke.to_csv(path, index=False)
    write_manifest(out / "wrds_smoke_manifest.json", {"kind": "wrds_smoke", "visible_libraries": len(libs)})
    db.close()
    LOG.info("WRDS smoke wrote %s", path)
    return path


def _valid_parquet(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def query_to_parquet(sql: str, out_path: str | Path, manifest_path: str | Path, params: dict | None = None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    db = _connect_wrds()
    df = db.raw_sql(sql, params=params)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    write_manifest(
        manifest_path,
        {
            "kind": "wrds_query",
            "out_path": str(out),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sql": sql,
            "params": params or {},
        },
    )
    db.close()
    return out


def query_to_parquet_if_missing(
    sql: str,
    out_path: str | Path,
    manifest_path: str | Path,
    params: dict | None = None,
    force: bool = False,
) -> Path:
    out = Path(out_path)
    if not force and _valid_parquet(out):
        LOG.info("Skipping existing shard %s", out)
        return out
    return query_to_parquet(sql, out, manifest_path, params=params)


def extract_minimal_crsp_monthly(out_dir: str | Path, start: str, end: str) -> Path:
    sql = """
        select permno, date, ret, prc, shrout, vol
        from crsp.msf
        where date between %(start)s and %(end)s
    """
    return query_to_parquet(
        sql,
        Path(out_dir) / "crsp_monthly_minimal.parquet",
        Path(out_dir) / "crsp_monthly_minimal_manifest.json",
        {"start": start, "end": end},
    )


AS_FILED_COLS = [
    "cik",
    "name",
    "adsh",
    "form",
    "fdate",
    "sic",
    "gvkey",
    "datadate",
    "at",
    "act",
    "ap",
    "ch",
    "che",
    "dlc",
    "dltt",
    "dp",
    "invt",
    "lct",
    "lt",
    "ppent",
    "seq",
    "ceq",
    "sale",
    "cogs",
    "xrd",
    "xad",
    "oiadp",
    "xint",
    "pi",
    "txt",
    "ib",
    "xsga",
    "oancf",
    "capx",
    "dv",
    "fincf",
    "prstkc",
    "sstk",
]

STRUCTURE_COLS = [
    "gvkey",
    "cik",
    "datadate",
    "adsh",
    "version_year",
    "dq_bs_xbrl",
    "dq_is_xbrl",
    "dq_xbrl",
    "sic",
    "form",
    "fdate",
    "d_amend",
    "ntag_bs",
    "level_bs",
    "ntag_bs_stmt",
    "level_bs_stmt",
    "ntag_bs_note",
    "level_bs_note",
    "ntag_is",
    "level_is",
    "ntag_is_stmt",
    "level_is_stmt",
    "ntag_is_note",
    "level_is_note",
    "ntag_cf",
    "level_cf",
    "ntag_cf_stmt",
    "level_cf_stmt",
    "ntag_cf_note",
    "level_cf_note",
]

COMPUSTAT_COLS = [
    "gvkey",
    "datadate",
    "fyear",
    "fyr",
    "indfmt",
    "consol",
    "popsrc",
    "datafmt",
    "tic",
    "cusip",
    "fdate",
    "pdate",
    "at",
    "sale",
    "cogs",
    "xrd",
    "xad",
    "oiadp",
    "ib",
    "oancf",
    "capx",
    "lt",
    "dlc",
    "dltt",
    "ceq",
    "seq",
    "che",
    "act",
    "lct",
    "invt",
    "ap",
]


def extract_wrds_core(
    out_dir: str | Path,
    start: str,
    end: str,
    years: list[int],
    smoke: bool = False,
    include_daily: bool = True,
    force: bool = False,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for year in years:
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        extract_as_filed_year(out, y_start, y_end, force=force)
        extract_structure_year(out, y_start, y_end, force=force)
        extract_crsp_monthly_year(out, y_start, y_end, force=force)
        if include_daily:
            extract_crsp_daily_year(out, y_start, y_end, force=force)
    extract_ccm(out, force=force)
    extract_compustat(out, start, end, force=force)
    extract_company(out, force=force)
    extract_ff_factors(out, start, end, force=force)
    extract_ibes(out, start, end, force=force and not smoke)
    write_manifest(
        out / "wrds_core_manifest.json",
        {
            "kind": "wrds_core",
            "start": start,
            "end": end,
            "years": years,
            "smoke": smoke,
            "include_daily": include_daily,
        },
    )


def extract_as_filed_year(out: Path, start: str, end: str, force: bool = False) -> Path:
    year = start[:4]
    cols = ", ".join(AS_FILED_COLS)
    sql = f"""
        select {cols}
        from contrib_as_filed_financials.funda_asfiled
        where fdate between %(start)s and %(end)s
          and form in ('10-K', '10-Q', '10-K/A', '10-Q/A')
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "as_filed" / f"funda_asfiled_{year}.parquet",
        out / "manifests" / f"funda_asfiled_{year}.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_structure_year(out: Path, start: str, end: str, force: bool = False) -> Path:
    year = start[:4]
    cols = ", ".join(STRUCTURE_COLS)
    sql = f"""
        select {cols}
        from contrib_as_filed_financials.structure_asfiled
        where fdate between %(start)s and %(end)s
          and form in ('10-K', '10-Q', '10-K/A', '10-Q/A')
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "as_filed" / f"structure_asfiled_{year}.parquet",
        out / "manifests" / f"structure_asfiled_{year}.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_crsp_monthly_year(out: Path, start: str, end: str, force: bool = False) -> Path:
    year = start[:4]
    sql = """
        select m.permno, m.permco, m.date, m.ret, m.retx, m.prc, m.shrout, m.vol,
               n.shrcd, n.exchcd, n.siccd, n.ncusip, n.ticker, n.comnam
        from crsp_a_stock.msf as m
        join crsp_a_stock.msenames as n
          on m.permno = n.permno
         and m.date between n.namedt and coalesce(n.nameendt, '2099-12-31')
        where m.date between %(start)s and %(end)s
          and n.shrcd in (10, 11)
          and n.exchcd in (1, 2, 3)
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "crsp" / f"crsp_monthly_{year}.parquet",
        out / "manifests" / f"crsp_monthly_{year}.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_crsp_daily_year(out: Path, start: str, end: str, force: bool = False) -> Path:
    year = start[:4]
    sql = """
        select d.permno, d.date, d.ret, d.retx, d.prc, d.vol, d.openprc,
               n.shrcd, n.exchcd, n.siccd
        from crsp_a_stock.dsf as d
        join crsp_a_stock.dsenames as n
          on d.permno = n.permno
         and d.date between n.namedt and coalesce(n.nameendt, '2099-12-31')
        where d.date between %(start)s and %(end)s
          and n.shrcd in (10, 11)
          and n.exchcd in (1, 2, 3)
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "crsp" / f"crsp_daily_{year}.parquet",
        out / "manifests" / f"crsp_daily_{year}.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_ccm(out: Path, force: bool = False) -> Path:
    sql = """
        select gvkey, linkprim, liid, linktype, lpermno, lpermco, usedflag, linkdt, linkenddt
        from crsp_a_ccm.ccmxpf_linktable
        where linktype in ('LC', 'LU', 'LS')
          and linkprim in ('P', 'C')
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "ccm" / "ccm_linktable.parquet",
        out / "manifests" / "ccm_linktable.json",
        force=force,
    )


def extract_compustat(out: Path, start: str, end: str, force: bool = False) -> Path:
    cols = ", ".join(COMPUSTAT_COLS)
    sql = f"""
        select {cols}
        from comp_na_daily_all.funda
        where datadate between %(start)s and %(end)s
          and indfmt = 'INDL'
          and consol = 'C'
          and popsrc = 'D'
          and datafmt = 'STD'
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "compustat" / "compustat_funda.parquet",
        out / "manifests" / "compustat_funda.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_company(out: Path, force: bool = False) -> Path:
    sql = """
        select gvkey, cik, conm, sic, gsector, ggroup, gind, gsubind, naics
        from comp_na_daily_all.company
        where cik is not null
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "compustat" / "company.parquet",
        out / "manifests" / "company.json",
        force=force,
    )


def extract_ff_factors(out: Path, start: str, end: str, force: bool = False) -> Path:
    sql = """
        select date, mktrf, smb, hml, rmw, cma, rf, umd, year, month, dateff
        from ff_all.fivefactors_monthly
        where date between %(start)s and %(end)s
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "ff" / "fivefactors_monthly.parquet",
        out / "manifests" / "fivefactors_monthly.json",
        {"start": start, "end": end},
        force=force,
    )


def extract_ibes(out: Path, start: str, end: str, force: bool = False) -> Path:
    sql = """
        select cusip, statpers, measure, fiscalp, fpi, numest, numup, numdown,
               meanest, stdev, actual, anndats_act
        from tr_ibes.statsum_epsus
        where statpers between %(start)s and %(end)s
          and measure = 'EPS'
          and usfirm = 1
    """
    return query_to_parquet_if_missing(
        sql,
        out / "raw" / "wrds" / "ibes" / "statsum_epsus.parquet",
        out / "manifests" / "statsum_epsus.json",
        {"start": start, "end": end},
        force=force,
    )


def read_parquet_glob(pattern: str | Path) -> pd.DataFrame:
    files = sorted(Path().glob(str(pattern))) if not isinstance(pattern, Path) else sorted(pattern.parent.glob(pattern.name))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
