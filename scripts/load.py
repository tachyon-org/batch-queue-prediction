"""Raw FIFE log scan and label derivation -- the single definition of `lf`.

Both `data-analysis.ipynb` and `feat-engineering.ipynb` build their frame from here.
It used to live only in the analysis notebook, so the feature notebook could not be
run on its own, and any copy of it would be a second fault taxonomy to keep in sync.

    from load import load_jobs
    globals().update(load_jobs())      # lf, have, qcol, to_sec, ftype, QMIN_S, ...
"""

import glob
import os
import time

import polars as pl

RAW_DIR = "/media/storage0/allison/BatchSystem-2025"

# Window start (2025-02-01 00:00 UTC). Jobs queued earlier have queue context
# that predates the logs. This is the ONLY definition -- the notebooks read it
# rather than recomputing a threshold of their own.
WINDOW_START = "2025-02-01"
WINDOW_START_S = int(
    __import__("datetime").datetime.fromisoformat(WINDOW_START)
    .replace(tzinfo=__import__("datetime").timezone.utc).timestamp())


def load_jobs(raw_dir=RAW_DIR, qmin_s=WINDOW_START_S):
    """Scans the raw partitions, derives fault labels, and returns the frame plus
    the names the notebooks build on.

    Returns a dict, so a notebook cell can `globals().update(load_jobs())` and get
    exactly the bindings the original scan cell left in scope.
    """
    # 1. Raw log location
    valid_paths = sorted(glob.glob(os.path.join(raw_dir, "batch_*.parquet")))
    print(f"Found {len(valid_paths)} Parquet partition files.")
    assert valid_paths, f"No files found in {raw_dir} matching 'batch_*.parquet'!"

    total_gb = sum(os.path.getsize(p) for p in valid_paths) / 1e9
    print(f"Total dataset size on disk: {total_gb:.2f} GB across {len(valid_paths)} files.")

    # 2. Columns to extract
    NEED = [
        "ExitCode","ExitSignal","JobStatus","RemoveReason","LastHoldReason","LastHoldReasonCode",
        "NumJobStarts","NumJobMatches",
        # Cmd identifies glidein pilots: a pilot is exactly Cmd == "./glidein_startup.sh".
        # Verified across all 71,158,721 raw rows -- 5,209,738 pilots, and not one of them
        # carries a site, so any site-sliced view excludes them by construction.
        "Cmd",
        # Lifecycle timestamps. EnteredCurrentStatus_ms is when the job entered its
        # terminal status: for a removed job that never ran, that is the only record of
        # when it left the queue. LastMatchTime_ms dates the match itself.
        "QDate_ms","JobStartDate_ms","JobCurrentStartDate_ms","JobCurrentStartExecutingDate_ms",
        "CompletionDate","CompletionDate_ms","EnteredCurrentStatus_ms","LastMatchTime_ms",
        "RemoteWallClockTime","x509UserProxyExpiration_ms","Owner","AccountingGroup",
        "Group","POMS4_CAMPAIGN_ID","POMS4_CAMPAIGN_NAME","POMS4_CAMPAIGN_STAGE_NAME",
        "POMS4_CAMPAIGN_STAGE_ID","POMS4_CAMPAIGN_TYPE","POMS4_TEST_LAUNCH","Jobsub_Group",
        "SingularityImage","Blacklist_Sites","RequestCpus","RequestDisk","RequestMemory","RequestSlots",
        "CpusProvisioned","DiskProvisioned","MemoryProvisioned","ExecutableSize","TransferInputSizeMB",
        "JOB_EXPECTED_MAX_LIFETIME","TotalSubmitProcs","MATCH_EXP_JOB_GLIDEIN_Site","MATCH_GLIDEIN_Entry_Name",
        "MATCH_GLIDEIN_SiteWMS_Queue","MachineAttrGLIDEIN_ResourceName0","MachineAttrCpus0","LastRemoteHost",
        "MATCH_EXP_JOB_Site","ClusterId",
    ]

    schema_sample = pl.read_parquet_schema(valid_paths[0])
    have = [c for c in NEED if c in schema_sample.names()]
    missing = [c for c in NEED if c not in schema_sample.names()]
    if missing:
        print("Columns in NEED not present in raw dataset:", missing)

    # 3. Create LazyFrame across all partitions
    try:
        lf = pl.scan_parquet(valid_paths, missing_columns="insert")
    except TypeError:
        lf = pl.scan_parquet(valid_paths, allow_missing_columns=True)

    lf = lf.select(have)

    if "AccountingGroup" in have:
        lf = lf.with_columns(pl.col("AccountingGroup").str.split(".").list.first().alias("Group"))
        if "Group" not in have:
            have.append("Group")

    # 4. Normalise the epoch-millisecond sentinels.
    _MS_COLS = [c for c in have if c.endswith("_ms")]
    if _MS_COLS:
        lf = lf.with_columns([
            pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None).alias(c) for c in _MS_COLS
        ])
        print(f"Normalised {len(_MS_COLS)} epoch-ms columns (0 -> null): {', '.join(_MS_COLS)}")

    schema = lf.collect_schema()

    # Timestamp normalization helper: Datetime -> epoch seconds, numeric -> rescaled
    # from ns/us/ms to seconds.
    def to_sec(col):
        if isinstance(schema[col], pl.Datetime):
            return pl.col(col).dt.epoch(time_unit="s").cast(pl.Float64)
        v = pl.col(col).cast(pl.Float64, strict=False)
        return (pl.when(v > 1e17).then(v / 1e9)
                  .when(v > 1e14).then(v / 1e6)
                  .when(v > 1e11).then(v / 1e3)
                  .otherwise(v))

    qcol = next((c for c in ["QDate_ms","QDate"] if c in have), None)
    scol = next((c for c in ["JobStartDate_ms","JobStartDate","JobCurrentStartDate_ms",
                             "JobCurrentStartExecutingDate_ms"] if c in have), None)
    ccol = next((c for c in ["CompletionDate","CompletionDate_ms"] if c in have), None)
    rcol = next((c for c in ["EnteredCurrentStatus_ms"] if c in have), None)
    print(f"Resolved: queue={qcol} start={scol} completion={ccol} terminal-status={rcol}")

    # ---- Labeling ----
    EXIT_SUB=[9,65,90,91,124,126,127,130,131,137]; EXIT_HW=[129,143]
    SIG_SUB=[2,3,9]; SIG_HW=[1,15]; HOLD_APP=[3,6,16]; HOLD_SUB=[1,4,7,8,12,13,26,32,33,34,35]

    ec, es, js = pl.col("ExitCode"), pl.col("ExitSignal"), pl.col("JobStatus")
    lhr = pl.col("LastHoldReasonCode") if "LastHoldReasonCode" in have else pl.lit(None, dtype=pl.Int64)
    rr  = pl.col("RemoveReason").fill_null("") if "RemoveReason" in have else pl.lit("")
    rr_lower = rr.str.to_lowercase()

    # A job the user removed. HTCondor writes the requesting user into RemoveReason.
    user_removed = (js == 3) & (
        rr_lower.str.contains("condor_rm")
        | rr_lower.str.contains("by user")
        | rr_lower.str.contains("user remove")
    )

    # condor_rm stops a job with SIGTERM and escalates to SIGKILL if it does not exit.
    # On a user-removed job those signals record HOW it was stopped, not that anything
    # went wrong, so they must not be read as fault evidence -- SIG_HW contains 15, so
    # treating them as evidence would file ~826k deliberate removals as hardware faults.
    # Any other signal (SIGSEGV, SIGABRT, SIGBUS ...) or a nonzero exit code is genuine
    # and the job is kept and classified on it.
    removal_signal = user_removed & es.is_in([9, 15])

    ftype = (
        # 1. CLEAN SUCCESS -> -1
        pl.when((ec == 0) & es.is_null() & (js != 3))
        .then(-1)
        # 2. USER CANCELLATION carrying no fault evidence -> -2 (excluded from the tasks)
        .when(user_removed & (es.is_null() | removal_signal) & (ec.is_null() | (ec == 0)))
        .then(-2)
        # 3. SIGNALS, ignoring the removal mechanism
        .when(es.is_not_null() & ~removal_signal)
        .then(
            pl.when(es.is_in(SIG_SUB)).then(2)
            .when(es.is_in(SIG_HW)).then(1)
            .otherwise(0)  # Unlisted signals -> Application Failure
        )
        # 4. NON-ZERO EXIT CODES
        .when(ec.is_not_null() & (ec != 0))
        .then(
            pl.when(ec.is_in(EXIT_SUB)).then(2)
            .when(ec.is_in(EXIT_HW)).then(1)
            .otherwise(0)  # Unlisted exit codes -> Application Failure
        )
        # 5. Any remaining user removal -> -2
        .when(user_removed)
        .then(-2)
        # 6. SYSTEM-ENFORCED REMOVALS (JobStatus == 3, not user-initiated)
        .when(js == 3)
        .then(
            pl.when(
                rr_lower.str.contains("exceeding job limits")
                | rr_lower.str.contains("opportunistic")
                # Sandbox transfer never completed, so the payload never ran. These
                # reached .otherwise(1) below and were filed as hardware: 227,795 of
                # them, 30% of the raw hardware tally, all on jobs that never started.
                | rr_lower.str.contains("staging of job files failed")
                | rr_lower.str.contains("spooling is taking too long")
            ).then(2)
            .when(
                rr_lower.str.contains("otherjobremoverequirements")
                | rr_lower.str.contains("node error: dag node")
                | rr_lower.str.contains("held 14 days")
                | rr_lower.str.contains("periodicremove")
            ).then(0)
            .otherwise(1)  # unrecognised system removal -> Hardware (see above)
        )
        # 7. HOLD REASONS
        .when(lhr.is_in(HOLD_SUB)).then(2)
        .when(lhr.is_in(HOLD_APP)).then(0)
        .otherwise(-1)
    ).cast(pl.Int8)

    wait = (to_sec(scol) - to_sec(qcol)) if (scol and qcol) else pl.lit(None, dtype=pl.Float64)

    lf = lf.with_columns([
        ftype.alias("fault_type"),
        (ftype >= 0).cast(pl.Int8).alias("Failed"),        # genuine technical failures
        (ftype == -2).cast(pl.Int8).alias("UserCancelled"),
        (ftype == -1).cast(pl.Int8).alias("Success"),
        (ftype == 1).cast(pl.Int8).alias("hw_fault"),
        pl.when(wait > 0).then(wait).otherwise(None).alias("wait_s"),
    ])

    # ---- Population ----
    # Drop jobs queued before the data window opens (2025-02-01 00:00 UTC): their queue
    # context predates the logs. This extraction reaches back to 2024-05, so the filter
    # now removes real rows rather than a handful of stragglers.
    QMIN_S = int(qmin_s)
    if qcol:
        lf = lf.filter(to_sec(qcol) >= QMIN_S)

    # A job that never started consumed no execution resources, so its outcome is a
    # queue-management event rather than a run-time failure and it is not a valid target
    # for failure prediction or fault attribution.
    #
    # It is NOT dropped, though: while queued it occupied a slot in the queue and so
    # contributed to the contention every other job's wait time depends on. Removing it
    # from the frame would silently bias the queue-depth feature. It is instead flagged
    # here and masked out of the E1/E3 target populations downstream, which keeps it
    # available as queue state. E2 excludes it automatically -- a job that never started
    # has no wait time to regress.
    lf = lf.with_columns((pl.col(scol).is_not_null()).cast(pl.Int8).alias("Ran"))

    # When each job left the queue: its start time if it ran, otherwise the moment it
    # entered its terminal status. The second case is what the previous extraction could
    # not express, and is what lets a never-ran job be counted in the queue for exactly
    # as long as it actually sat there.
    if rcol:
        lf = lf.with_columns(
            pl.when(pl.col(scol).is_not_null()).then(to_sec(scol))
              .otherwise(to_sec(rcol)).alias("t_queue_exit")
        )
    else:
        lf = lf.with_columns(to_sec(scol).alias("t_queue_exit"))

    print("Lazy execution plan initialized successfully!")

    return {
        "lf": lf,
        "have": have,
        "missing": missing,
        "NEED": NEED,
        "qcol": qcol,
        "scol": scol,
        "ccol": ccol,
        "rcol": rcol,
        "to_sec": to_sec,
        "ftype": ftype,
        "QMIN_S": QMIN_S,
        "WINDOW_START": WINDOW_START,
        "raw_dir": raw_dir,
        "valid_paths": valid_paths,
    }
