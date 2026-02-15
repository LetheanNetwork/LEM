#!/usr/bin/env python3
"""LEM Golden Set Pipeline: JSONL → DuckDB → InfluxDB metrics → Parquet → HuggingFace"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "golden-set.duckdb"
JSONL_PATH = BASE_DIR / "gold-15k.jsonl"
OUTPUT_DIR = BASE_DIR / "output"

INFLUX_URL = "http://localhost:8181"
INFLUX_DB = "training"
INFLUX_TOKEN_PATH = Path.home() / ".influx_token"

TARGET_TOTAL = 15000  # expected final count


def get_db():
    return duckdb.connect(str(DB_PATH))


def cmd_ingest(args):
    """Ingest JSONL into DuckDB, idempotent on idx."""
    src = Path(args.file) if args.file else JSONL_PATH
    if not src.exists():
        print(f"Error: {src} not found. Copy from M3: scp m3:/Volumes/Data/lem/responses/gold-15k.jsonl {JSONL_PATH}")
        sys.exit(1)

    db = get_db()
    db.execute("DROP TABLE IF EXISTS golden_set")
    db.execute("""
        CREATE TABLE golden_set AS
        SELECT
            idx::INT AS idx,
            seed_id::VARCHAR AS seed_id,
            domain::VARCHAR AS domain,
            voice::VARCHAR AS voice,
            prompt::VARCHAR AS prompt,
            response::VARCHAR AS response,
            gen_time::DOUBLE AS gen_time,
            length(response)::INT AS char_count,
            length(response) - length(replace(response, ' ', '')) + 1 AS word_count
        FROM read_json_auto(?, maximum_object_size=1048576)
    """, [str(src)])

    count = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
    domains = db.execute("SELECT count(DISTINCT domain) FROM golden_set").fetchone()[0]
    voices = db.execute("SELECT count(DISTINCT voice) FROM golden_set").fetchone()[0]
    print(f"Ingested {count} examples ({domains} domains, {voices} voices) into {DB_PATH}")
    db.close()


def cmd_status(args):
    """Show dataset statistics."""
    db = get_db()
    try:
        total = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
    except duckdb.CatalogException:
        print("No data ingested yet. Run: pipeline.py ingest")
        return

    stats = db.execute("""
        SELECT
            count(*) AS total,
            count(DISTINCT domain) AS domains,
            count(DISTINCT voice) AS voices,
            round(avg(gen_time), 2) AS avg_gen_time,
            round(avg(char_count), 0) AS avg_chars,
            round(avg(word_count), 0) AS avg_words,
            min(gen_time) AS min_gen,
            max(gen_time) AS max_gen
        FROM golden_set
    """).fetchone()

    pct = (stats[0] / TARGET_TOTAL) * 100
    print(f"Golden Set Status")
    print(f"─────────────────────────────────────")
    print(f"  Examples:     {stats[0]:,} / {TARGET_TOTAL:,} ({pct:.1f}%)")
    print(f"  Domains:      {stats[1]}")
    print(f"  Voices:       {stats[2]}")
    print(f"  Avg gen time: {stats[3]}s (min: {stats[6]:.1f}s, max: {stats[7]:.1f}s)")
    print(f"  Avg response: {int(stats[4]):,} chars / {int(stats[5]):,} words")

    if args.domains:
        print(f"\nDomain breakdown:")
        rows = db.execute("""
            SELECT domain, count(*) AS n, round(avg(gen_time), 1) AS avg_t
            FROM golden_set GROUP BY domain ORDER BY n DESC
        """).fetchall()
        for domain, n, avg_t in rows:
            print(f"  {domain:30s} {n:5d}  ({avg_t}s avg)")

    if args.voices:
        print(f"\nVoice breakdown:")
        rows = db.execute("""
            SELECT voice, count(*) AS n, round(avg(char_count), 0) AS avg_c
            FROM golden_set GROUP BY voice ORDER BY n DESC
        """).fetchall()
        for voice, n, avg_c in rows:
            print(f"  {voice:20s} {n:5d}  ({int(avg_c):,} avg chars)")

    db.close()


def cmd_query(args):
    """Run ad-hoc SQL against the golden set."""
    db = get_db()
    sql = args.sql
    if not sql.strip().upper().startswith("SELECT"):
        sql = f"SELECT * FROM golden_set WHERE {sql} LIMIT 20"
    rows = db.execute(sql).fetchdf()
    print(rows.to_string())
    db.close()


def _influx_write(lines):
    """Write line protocol to InfluxDB 3 via HTTP API."""
    import urllib.request
    token = INFLUX_TOKEN_PATH.read_text().strip()
    body = "\n".join(lines).encode()
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v3/write_lp?db={INFLUX_DB}",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
        method="POST",
    )
    urllib.request.urlopen(req)


def cmd_metrics(args):
    """Write pipeline metrics to InfluxDB 3."""
    db = get_db()
    try:
        total = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
    except duckdb.CatalogException:
        print("No data ingested yet. Run: pipeline.py ingest")
        return

    stats = db.execute("""
        SELECT count(*), count(DISTINCT domain), count(DISTINCT voice),
               avg(gen_time), avg(char_count)
        FROM golden_set
    """).fetchone()

    domain_stats = db.execute("""
        SELECT domain, count(*) AS n, avg(gen_time) AS avg_t
        FROM golden_set GROUP BY domain
    """).fetchall()

    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

    lines = []
    # Overall stats
    lines.append(
        f"golden_set_stats total_examples={stats[0]}i,"
        f"domains={stats[1]}i,voices={stats[2]}i,"
        f"avg_gen_time={stats[3]:.2f},avg_response_chars={stats[4]:.0f},"
        f"completion_pct={stats[0] / TARGET_TOTAL * 100:.1f} {now_ns}"
    )

    # Per-domain stats
    for domain, n, avg_t in domain_stats:
        safe = domain.replace(" ", "\\ ").replace(",", "\\,")
        lines.append(
            f"golden_set_domain,domain={safe} count={n}i,avg_gen_time={avg_t:.2f} {now_ns}"
        )

    # Per-voice stats
    voice_stats = db.execute("""
        SELECT voice, count(*) AS n, avg(char_count) AS avg_c, avg(gen_time) AS avg_t
        FROM golden_set GROUP BY voice
    """).fetchall()
    for voice, n, avg_c, avg_t in voice_stats:
        safe = voice.replace(" ", "\\ ").replace(",", "\\,")
        lines.append(
            f"golden_set_voice,voice={safe} count={n}i,avg_chars={avg_c:.0f},avg_gen_time={avg_t:.2f} {now_ns}"
        )

    _influx_write(lines)
    db.close()
    print(f"Wrote metrics to InfluxDB ({INFLUX_DB}): {stats[0]} examples, {len(domain_stats)} domains, {len(voice_stats)} voices")


def cmd_export(args):
    """Export DuckDB → Parquet with train/test/valid split."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    db = get_db()

    try:
        total = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
    except duckdb.CatalogException:
        print("No data ingested yet. Run: pipeline.py ingest")
        return

    # Stratified split: 80/10/10 by domain, deterministic via idx
    db.execute("""
        CREATE OR REPLACE VIEW golden_split AS
        SELECT *,
            CASE
                WHEN row_number() OVER (PARTITION BY domain ORDER BY idx) % 10 < 8 THEN 'train'
                WHEN row_number() OVER (PARTITION BY domain ORDER BY idx) % 10 = 8 THEN 'test'
                ELSE 'validation'
            END AS split
        FROM golden_set
    """)

    for split in ['train', 'test', 'validation']:
        out = OUTPUT_DIR / f"{split}.parquet"
        db.execute(f"""
            COPY (
                SELECT idx, seed_id, domain, voice, prompt, response,
                       gen_time, char_count, word_count
                FROM golden_split WHERE split = '{split}'
                ORDER BY idx
            ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = db.execute(f"SELECT count(*) FROM golden_split WHERE split = '{split}'").fetchone()[0]
        size = out.stat().st_size
        print(f"  {split:12s} → {n:6,} rows  ({size / 1024 / 1024:.1f} MB)  {out}")

    # Also export full dataset as single file
    full = OUTPUT_DIR / "golden_set.parquet"
    db.execute(f"""
        COPY (SELECT * FROM golden_set ORDER BY idx)
        TO '{full}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  {'full':12s} → {total:6,} rows  ({full.stat().st_size / 1024 / 1024:.1f} MB)  {full}")

    db.close()
    print(f"\nExported to {OUTPUT_DIR}/")


def cmd_publish(args):
    """Push Parquet files + dataset card to HuggingFace."""
    from huggingface_hub import HfApi

    repo_id = args.repo or "lthn/LEM-golden-set"
    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=not args.public)

    # Upload parquet files
    for split in ['train', 'test', 'validation']:
        f = OUTPUT_DIR / f"{split}.parquet"
        if f.exists():
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"data/{split}.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {f.name} → data/{split}.parquet")

    # Upload dataset card
    card_path = BASE_DIR / "dataset_card.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded dataset card")

    print(f"\nPublished to https://huggingface.co/datasets/{repo_id}")


def cmd_seed_influx(args):
    """Seed InfluxDB golden_gen from existing DuckDB data (one-time migration)."""
    db = get_db()
    try:
        total = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
    except duckdb.CatalogException:
        print("No data ingested yet. Run: pipeline.py ingest")
        return

    # Check how many are already in InfluxDB
    token = INFLUX_TOKEN_PATH.read_text().strip()
    import urllib.request, urllib.error
    existing = 0
    try:
        body = json.dumps({"db": INFLUX_DB, "q": "SELECT count(DISTINCT i) AS n FROM gold_gen"}).encode()
        req = urllib.request.Request(
            f"{INFLUX_URL}/api/v3/query_sql",
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req).read())
        existing = resp[0]["n"] if resp else 0
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass  # table doesn't exist yet
    print(f"DuckDB has {total} records, InfluxDB golden_gen has {existing}")

    if existing >= total and not args.force:
        print("InfluxDB already has all records. Use --force to re-seed.")
        return

    # Read all rows from DuckDB
    rows = db.execute("""
        SELECT idx, seed_id, domain, voice, gen_time, char_count
        FROM golden_set ORDER BY idx
    """).fetchall()

    # Write in batches to InfluxDB
    batch_size = 500
    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        lines = []
        for idx, seed_id, domain, voice, gen_time, char_count in batch:
            d = domain.replace(" ", "\\ ").replace(",", "\\,")
            v = voice.replace(" ", "\\ ").replace(",", "\\,")
            sid = seed_id.replace('"', '\\"')
            lines.append(
                f'gold_gen,i={idx},w=migration,d={d},v={v} '
                f'seed_id="{sid}",gen_time={gen_time:.1f},'
                f'chars={char_count}i'
            )
        _influx_write(lines)
        written += len(batch)
        if written % 2000 == 0 or written == len(rows):
            print(f"  Seeded {written}/{len(rows)} records")

    db.close()
    print(f"Seeded {written} golden_gen records into InfluxDB")


def cmd_consolidate(args):
    """Pull all worker JSONLs from M3, merge, deduplicate, re-ingest + export."""
    import subprocess, glob

    responses_dir = BASE_DIR / "responses"
    responses_dir.mkdir(exist_ok=True)

    # Pull all JSONL files from M3
    print("Pulling responses from M3...")
    result = subprocess.run(
        ["ssh", "m3", "ls /Volumes/Data/lem/responses/gold*.jsonl"],
        capture_output=True, text=True
    )
    remote_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    print(f"  Found {len(remote_files)} JSONL files on M3")

    for rf in remote_files:
        local = responses_dir / Path(rf).name
        subprocess.run(["scp", f"m3:{rf}", str(local)], check=True)
        lines = sum(1 for _ in open(local))
        print(f"  {local.name}: {lines:,} records")

    # Merge and deduplicate on idx
    seen = {}
    skipped = 0
    for local in sorted(responses_dir.glob("gold*.jsonl")):
        with open(local) as f:
            for line in f:
                rec = json.loads(line)
                idx = rec.get("idx")
                if idx is None:
                    skipped += 1
                    continue
                if idx not in seen:
                    seen[idx] = rec
    if skipped:
        print(f"  Skipped {skipped} records without idx")

    # Write merged file
    merged = BASE_DIR / "gold-merged.jsonl"
    with open(merged, "w") as f:
        for idx in sorted(seen.keys()):
            f.write(json.dumps(seen[idx]) + "\n")
    print(f"\nMerged: {len(seen):,} unique examples → {merged}")

    # Re-ingest into DuckDB
    class FakeArgs:
        file = str(merged)
    cmd_ingest(FakeArgs())

    # Update InfluxDB metrics
    cmd_metrics(FakeArgs())

    # Export Parquet
    cmd_export(FakeArgs())

    print(f"\nConsolidation complete: {len(seen):,} examples")


def cmd_live(args):
    """Show live generation progress from InfluxDB."""
    import urllib.request
    token = INFLUX_TOKEN_PATH.read_text().strip()

    def query(sql):
        body = json.dumps({"db": INFLUX_DB, "q": sql}).encode()
        req = urllib.request.Request(
            f"{INFLUX_URL}/api/v3/query_sql", data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req).read())

    total = query("SELECT count(DISTINCT i) AS n FROM gold_gen")[0]["n"]
    workers = query("SELECT w, count(DISTINCT i) AS n FROM gold_gen GROUP BY w ORDER BY n DESC")
    domains = query("SELECT count(DISTINCT d) AS n FROM gold_gen")[0]["n"]
    voices = query("SELECT count(DISTINCT v) AS n FROM gold_gen")[0]["n"]

    pct = total / TARGET_TOTAL * 100
    remaining = TARGET_TOTAL - total

    print(f"Golden Set Live Status (from InfluxDB)")
    print(f"{'─' * 45}")
    print(f"  Total:     {total:,} / {TARGET_TOTAL:,} ({pct:.1f}%)")
    print(f"  Remaining: {remaining:,}")
    print(f"  Domains:   {domains}")
    print(f"  Voices:    {voices}")
    print(f"\n  Workers:")
    for w in workers:
        name = w["w"]
        n = int(w["n"])
        marker = " (seed data)" if name == "migration" else ""
        print(f"    {name:20s} {n:6,} generations{marker}")


def cmd_refresh(args):
    """Pull latest JSONL from M3 and re-ingest."""
    import subprocess
    print("Pulling latest golden set from M3...")
    subprocess.run(["scp", "m3:/Volumes/Data/lem/responses/gold-15k.jsonl", str(JSONL_PATH)], check=True)
    cmd_ingest(args)


def cmd_import_all(args):
    """Import ALL LEM data into DuckDB: prompts, Gemini responses, training, benchmarks, seeds."""
    import subprocess

    db = get_db()
    totals = {}

    # ── 1. Prompts (16K expanded) ──────────────────────────────────────
    prompts_file = Path("/tmp/lem-prompts.jsonl")
    if prompts_file.exists():
        db.execute("DROP TABLE IF EXISTS prompts")
        db.execute("""
            CREATE TABLE prompts AS
            SELECT
                seed_id::VARCHAR AS seed_id,
                domain::VARCHAR AS domain,
                voice::VARCHAR AS voice,
                prompt::VARCHAR AS prompt
            FROM read_json_auto(?, maximum_object_size=1048576)
        """, [str(prompts_file)])
        n = db.execute("SELECT count(*) FROM prompts").fetchone()[0]
        totals["prompts"] = n
        print(f"  prompts: {n:,} rows")
    else:
        print("  prompts: SKIP (no /tmp/lem-prompts.jsonl)")

    # ── 2. Gemini responses (TPU-generated) ────────────────────────────
    gemini_files = [
        ("/tmp/lem-gemini25flash-responses.jsonl", "gemini-2.5-flash"),
        ("/tmp/lem-gemini3flash-responses.jsonl", "gemini-3-flash"),
        ("/tmp/lem-gemini3-responses.jsonl", "gemini-3"),
    ]
    all_gemini = []
    for path, model_label in gemini_files:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    rec["_source_model"] = model_label
                    all_gemini.append(rec)

    if all_gemini:
        db.execute("DROP TABLE IF EXISTS gemini_responses")
        db.execute("""
            CREATE TABLE gemini_responses (
                seed_id VARCHAR,
                region VARCHAR,
                domain VARCHAR,
                prompt TEXT,
                response TEXT,
                gen_time DOUBLE,
                model VARCHAR,
                source_model VARCHAR,
                char_count INT,
                word_count INT
            )
        """)
        for rec in all_gemini:
            resp = rec.get("response", "")
            db.execute("""
                INSERT INTO gemini_responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                rec.get("seed_id", ""),
                rec.get("region", ""),
                rec.get("domain", ""),
                rec.get("prompt", ""),
                resp,
                rec.get("gen_time", 0),
                rec.get("model", ""),
                rec["_source_model"],
                len(resp),
                len(resp.split()) if resp else 0,
            ])
        n = db.execute("SELECT count(*) FROM gemini_responses").fetchone()[0]
        totals["gemini_responses"] = n
        print(f"  gemini_responses: {n:,} rows")

    # ── 3. Golden set (pull from M3 + ingest) ──────────────────────────
    if not args.skip_m3:
        print("  Pulling golden set from M3...")
        try:
            subprocess.run(["scp", "m3:/Volumes/Data/lem/responses/gold-15k.jsonl",
                           str(JSONL_PATH)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("    WARNING: could not pull from M3")

    if JSONL_PATH.exists():
        db.execute("DROP TABLE IF EXISTS golden_set")
        db.execute("""
            CREATE TABLE golden_set AS
            SELECT
                idx::INT AS idx,
                seed_id::VARCHAR AS seed_id,
                domain::VARCHAR AS domain,
                voice::VARCHAR AS voice,
                prompt::VARCHAR AS prompt,
                response::VARCHAR AS response,
                gen_time::DOUBLE AS gen_time,
                length(response)::INT AS char_count,
                length(response) - length(replace(response, ' ', '')) + 1 AS word_count
            FROM read_json_auto(?, maximum_object_size=1048576)
        """, [str(JSONL_PATH)])
        n = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
        totals["golden_set"] = n
        print(f"  golden_set: {n:,} rows")

    # ── 4. Training examples (pull from M3) ────────────────────────────
    training_dirs = [
        ("training", "training/train.jsonl", "training/valid.jsonl", "training/test.jsonl"),
        ("training-2k", "training-2k/train.jsonl", "training-2k/valid.jsonl", "training-2k/test.jsonl"),
        ("training-expanded", "training-expanded/train.jsonl", "training-expanded/valid.jsonl", None),
        ("training-book", "training-book/train.jsonl", "training-book/valid.jsonl", "training-book/test.jsonl"),
        ("training-conv", "training-conv/train.jsonl", "training-conv/valid.jsonl", "training-conv/test.jsonl"),
        ("gold-full", "gold-full/train.jsonl", "gold-full/valid.jsonl", None),
        ("sovereignty-gold", "sovereignty-gold/train.jsonl", "sovereignty-gold/valid.jsonl", None),
        ("composure-lessons", "composure-lessons/train.jsonl", "composure-lessons/valid.jsonl", None),
        ("watts-full", "watts-full/train.jsonl", "watts-full/valid.jsonl", None),
        ("watts-expanded", "watts-expanded/train.jsonl", "watts-expanded/valid.jsonl", None),
        ("watts-composure", "watts-composure-merged/train.jsonl", "watts-composure-merged/valid.jsonl", None),
        ("western-fresh", "western-fresh/train.jsonl", "western-fresh/valid.jsonl", None),
        ("deepseek-soak", "deepseek-western-soak/train.jsonl", "deepseek-western-soak/valid.jsonl", None),
        ("russian-bridge", "russian-bridge/train.jsonl", "russian-bridge/valid.jsonl", None),
    ]

    training_local = BASE_DIR / "training"
    training_local.mkdir(exist_ok=True)

    if not args.skip_m3:
        print("  Pulling training sets from M3...")
        for source_name, *files in training_dirs:
            for rel in files:
                if rel is None:
                    continue
                local = training_local / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(
                        ["scp", f"m3:/Volumes/Data/lem/{rel}", str(local)],
                        check=True, capture_output=True
                    )
                except subprocess.CalledProcessError:
                    pass  # file might not exist

    # Parse all training data into a unified table
    db.execute("DROP TABLE IF EXISTS training_examples")
    db.execute("""
        CREATE TABLE training_examples (
            source VARCHAR,
            split VARCHAR,
            prompt TEXT,
            response TEXT,
            num_turns INT,
            full_messages TEXT,
            char_count INT
        )
    """)

    training_total = 0
    for source_name, *files in training_dirs:
        for rel in files:
            if rel is None:
                continue
            local = training_local / rel
            if not local.exists():
                continue
            split = "train" if "train" in rel else "valid" if "valid" in rel else "test"
            with open(local) as f:
                for line in f:
                    rec = json.loads(line)
                    msgs = rec.get("messages", [])
                    prompt = msgs[0]["content"] if msgs else ""
                    response = msgs[1]["content"] if len(msgs) > 1 else ""
                    num_turns = len([m for m in msgs if m["role"] == "assistant"])
                    db.execute("""
                        INSERT INTO training_examples VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        source_name, split, prompt, response,
                        num_turns, json.dumps(msgs), len(response)
                    ])
                    training_total += 1

    totals["training_examples"] = training_total
    print(f"  training_examples: {training_total:,} rows")

    # ── 5. Benchmark questions ─────────────────────────────────────────
    bench_questions_files = [
        "/tmp/lem-benchmarks/deepseek-sovereignty-capability.jsonl",
        "/tmp/lem-benchmarks/deepseek-sovereignty-content-scores.jsonl",
        "/tmp/lem-benchmarks/gemma12b-content-scores.jsonl",
        "/tmp/lem-benchmarks/russian-bridge-content-scores.jsonl",
    ]

    # Also pull standard benchmarks from M3
    bench_local = BASE_DIR / "benchmarks"
    bench_local.mkdir(exist_ok=True)
    if not args.skip_m3:
        print("  Pulling benchmarks from M3...")
        for bname in ["truthfulqa", "gsm8k", "do_not_answer", "toxigen"]:
            try:
                subprocess.run(
                    ["scp", f"m3:/Volumes/Data/lem/benchmarks/{bname}.jsonl",
                     str(bench_local / f"{bname}.jsonl")],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                pass

        # Pull all benchmark results
        for subdir in ["results", "scale_results", "cross_arch_results"]:
            local_sub = bench_local / subdir
            local_sub.mkdir(exist_ok=True)
            try:
                subprocess.run(
                    ["scp", "-r", f"m3:/Volumes/Data/lem/benchmarks/{subdir}/",
                     str(local_sub.parent) + "/"],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                pass

        # Pull deepseek-r1-7b benchmarks
        ds_local = bench_local / "deepseek-r1-7b"
        ds_local.mkdir(exist_ok=True)
        try:
            subprocess.run(
                ["scp", "-r", f"m3:/Volumes/Data/lem/benchmarks/deepseek-r1-7b/",
                 str(ds_local.parent) + "/"],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            pass

    # Import benchmark results (all share: id, benchmark, model, prompt, response, elapsed_seconds)
    db.execute("DROP TABLE IF EXISTS benchmark_results")
    db.execute("""
        CREATE TABLE benchmark_results (
            source VARCHAR,
            id VARCHAR,
            benchmark VARCHAR,
            model VARCHAR,
            prompt TEXT,
            response TEXT,
            elapsed_seconds DOUBLE,
            domain VARCHAR
        )
    """)

    bench_total = 0
    # Scan all result directories
    for subdir in ["results", "scale_results", "cross_arch_results", "deepseek-r1-7b"]:
        result_dir = bench_local / subdir
        if not result_dir.exists():
            continue
        for jf in sorted(result_dir.glob("*.jsonl")):
            with open(jf) as f:
                for line in f:
                    rec = json.loads(line)
                    db.execute("""
                        INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        subdir,
                        str(rec.get("id", "")),
                        rec.get("benchmark", ""),
                        rec.get("model", ""),
                        rec.get("prompt", ""),
                        rec.get("response", ""),
                        rec.get("elapsed_seconds", 0),
                        rec.get("domain", ""),
                    ])
                    bench_total += 1

    # Also add lem_bench and lem_ethics from M3
    for bfile in ["lem_bench", "lem_ethics", "lem_ethics_allen", "instruction_tuned",
                   "abliterated", "base_pt"]:
        local = bench_local / f"{bfile}.jsonl"
        if not local.exists() and not args.skip_m3:
            try:
                subprocess.run(
                    ["scp", f"m3:/Volumes/Data/lem/benchmark/{bfile}.jsonl", str(local)],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                pass
        if local.exists():
            with open(local) as f:
                for line in f:
                    rec = json.loads(line)
                    db.execute("""
                        INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        "benchmark",
                        str(rec.get("id", "")),
                        rec.get("benchmark", bfile),
                        rec.get("model", ""),
                        rec.get("prompt", ""),
                        rec.get("response", ""),
                        rec.get("elapsed_seconds", 0),
                        rec.get("domain", ""),
                    ])
                    bench_total += 1

    totals["benchmark_results"] = bench_total
    print(f"  benchmark_results: {bench_total:,} rows")

    # ── 6. Benchmark questions (reference datasets) ────────────────────
    db.execute("DROP TABLE IF EXISTS benchmark_questions")
    db.execute("""
        CREATE TABLE benchmark_questions (
            benchmark VARCHAR,
            id VARCHAR,
            question TEXT,
            best_answer TEXT,
            correct_answers TEXT,
            incorrect_answers TEXT,
            category VARCHAR
        )
    """)

    bench_q_total = 0
    for bname in ["truthfulqa", "gsm8k", "do_not_answer", "toxigen"]:
        local = bench_local / f"{bname}.jsonl"
        if local.exists():
            with open(local) as f:
                for line in f:
                    rec = json.loads(line)
                    db.execute("""
                        INSERT INTO benchmark_questions VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        rec.get("benchmark", bname),
                        str(rec.get("id", "")),
                        rec.get("question", ""),
                        rec.get("best_answer", ""),
                        json.dumps(rec.get("correct_answers", [])),
                        json.dumps(rec.get("incorrect_answers", [])),
                        rec.get("category", ""),
                    ])
                    bench_q_total += 1

    totals["benchmark_questions"] = bench_q_total
    print(f"  benchmark_questions: {bench_q_total:,} rows")

    # ── 7. Validation A/B tests ────────────────────────────────────────
    validation_files = {
        "/tmp/lem-base-27b-unsigned.jsonl": "base-27b",
        "/tmp/lem-v5b-unsigned.jsonl": "v5b",
    }
    # Also pull from M3
    val_local = BASE_DIR / "validation"
    val_local.mkdir(exist_ok=True)
    if not args.skip_m3:
        try:
            subprocess.run(
                ["scp", "-r", "m3:/Volumes/Data/lem/validation/", str(val_local.parent) + "/"],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            pass

    db.execute("DROP TABLE IF EXISTS validations")
    db.execute("""
        CREATE TABLE validations (
            source VARCHAR,
            seed_id VARCHAR,
            prompt TEXT,
            response TEXT,
            reasoning TEXT,
            is_refusal BOOLEAN,
            mode VARCHAR,
            chars INT,
            reasoning_chars INT,
            gen_time DOUBLE
        )
    """)

    val_total = 0
    # Local /tmp files
    for path, source in validation_files.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    db.execute("""
                        INSERT INTO validations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        source,
                        rec.get("seed_id", ""),
                        rec.get("prompt", ""),
                        rec.get("response", ""),
                        rec.get("reasoning", ""),
                        rec.get("is_refusal", False),
                        rec.get("mode", ""),
                        rec.get("chars", 0),
                        rec.get("reasoning_chars", 0),
                        rec.get("time", 0),
                    ])
                    val_total += 1

    # M3 validation files
    for vf in sorted(val_local.glob("*.jsonl")):
        source = vf.stem
        with open(vf) as f:
            for line in f:
                rec = json.loads(line)
                db.execute("""
                    INSERT INTO validations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    source,
                    rec.get("seed_id", ""),
                    rec.get("prompt", ""),
                    rec.get("response", ""),
                    rec.get("reasoning", ""),
                    rec.get("is_refusal", False),
                    rec.get("mode", ""),
                    rec.get("chars", 0),
                    rec.get("reasoning_chars", 0),
                    rec.get("time", 0),
                ])
                val_total += 1

    totals["validations"] = val_total
    print(f"  validations: {val_total:,} rows")

    # ── 8. Seeds (JSON files → table) ──────────────────────────────────
    db.execute("DROP TABLE IF EXISTS seeds")
    db.execute("""
        CREATE TABLE seeds (
            source_file VARCHAR,
            region VARCHAR,
            seed_id VARCHAR,
            domain VARCHAR,
            prompt TEXT
        )
    """)

    seed_total = 0
    seed_dirs = [Path("/tmp/lem-data/seeds"), Path("/tmp/lem-repo/seeds")]
    for seed_dir in seed_dirs:
        if not seed_dir.exists():
            continue
        for jf in sorted(seed_dir.rglob("*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)

                # Handle various formats
                seeds_list = []
                if isinstance(data, list):
                    seeds_list = data
                elif isinstance(data, dict):
                    seeds_list = data.get("prompts", data.get("seeds", []))

                region = jf.stem
                for s in seeds_list:
                    if isinstance(s, dict):
                        db.execute("""
                            INSERT INTO seeds VALUES (?, ?, ?, ?, ?)
                        """, [
                            str(jf.relative_to(seed_dir)),
                            region,
                            s.get("seed_id", s.get("id", "")),
                            s.get("domain", s.get("category", "")),
                            s.get("prompt", s.get("text", s.get("question", json.dumps(s)[:500]))),
                        ])
                        seed_total += 1
                    elif isinstance(s, str):
                        db.execute("""
                            INSERT INTO seeds VALUES (?, ?, ?, ?, ?)
                        """, [str(jf.relative_to(seed_dir)), region, "", "", s])
                        seed_total += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"    SKIP {jf.name}: {e}")

    totals["seeds"] = seed_total
    print(f"  seeds: {seed_total:,} rows")

    db.close()

    # ── Summary ────────────────────────────────────────────────────────
    grand_total = sum(totals.values())
    print(f"\n{'=' * 50}")
    print(f"LEM Database Import Complete")
    print(f"{'=' * 50}")
    for table, count in totals.items():
        print(f"  {table:25s} {count:>8,}")
    print(f"  {'─' * 35}")
    print(f"  {'TOTAL':25s} {grand_total:>8,}")
    print(f"\nDatabase: {DB_PATH}")


def cmd_inventory(args):
    """Show full inventory of all tables in the database."""
    db = get_db()
    tables = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()

    print(f"LEM Database Inventory ({DB_PATH})")
    print(f"{'=' * 60}")

    grand_total = 0
    for (table,) in tables:
        n = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        grand_total += n

        # Table-specific stats
        if table == "prompts":
            domains = db.execute("SELECT count(DISTINCT domain) FROM prompts").fetchone()[0]
            voices = db.execute("SELECT count(DISTINCT voice) FROM prompts").fetchone()[0]
            print(f"  {table:25s} {n:>8,}  ({domains} domains, {voices} voices)")
        elif table == "gemini_responses":
            models = db.execute("SELECT source_model, count(*) FROM gemini_responses GROUP BY source_model").fetchall()
            detail = ", ".join(f"{m}: {c:,}" for m, c in models)
            print(f"  {table:25s} {n:>8,}  ({detail})")
        elif table == "golden_set":
            pct = n / TARGET_TOTAL * 100
            print(f"  {table:25s} {n:>8,}  ({pct:.1f}% of {TARGET_TOTAL:,} target)")
        elif table == "training_examples":
            sources = db.execute("SELECT count(DISTINCT source) FROM training_examples").fetchone()[0]
            print(f"  {table:25s} {n:>8,}  ({sources} sources)")
        elif table == "benchmark_results":
            sources = db.execute("SELECT count(DISTINCT source) FROM benchmark_results").fetchone()[0]
            print(f"  {table:25s} {n:>8,}  ({sources} categories)")
        else:
            print(f"  {table:25s} {n:>8,}")

    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':25s} {grand_total:>8,}")
    db.close()


def cmd_normalize_seeds(args):
    """Normalize 87K seeds into expansion_prompts, deduped against golden set."""
    db = get_db()

    # Check source tables exist
    try:
        seed_count = db.execute("SELECT count(*) FROM seeds").fetchone()[0]
    except duckdb.CatalogException:
        print("No seeds table. Run: pipeline.py import-all")
        return

    print(f"Seeds table: {seed_count:,} rows")

    # Deduplicate: remove seeds whose prompt text already appears in prompts or golden_set
    db.execute("DROP TABLE IF EXISTS expansion_prompts")
    db.execute("""
        CREATE TABLE expansion_prompts AS
        WITH unique_seeds AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY region, domain, seed_id) AS idx,
                seed_id,
                region,
                domain,
                prompt
            FROM (
                SELECT DISTINCT ON (prompt)
                    seed_id, region, domain, prompt
                FROM seeds
                WHERE length(prompt) >= 50
                ORDER BY prompt, seed_id
            )
        ),
        existing_prompts AS (
            SELECT prompt FROM prompts
            UNION ALL
            SELECT prompt FROM golden_set
        )
        SELECT
            us.idx,
            us.seed_id,
            us.region,
            us.domain,
            'en' AS language,
            us.prompt,
            '' AS prompt_en,
            0 AS priority,
            'pending' AS status
        FROM unique_seeds us
        WHERE NOT EXISTS (
            SELECT 1 FROM existing_prompts ep
            WHERE ep.prompt = us.prompt
        )
    """)

    total = db.execute("SELECT count(*) FROM expansion_prompts").fetchone()[0]
    domains = db.execute("SELECT count(DISTINCT domain) FROM expansion_prompts").fetchone()[0]
    regions = db.execute("SELECT count(DISTINCT region) FROM expansion_prompts").fetchone()[0]

    # Assign priority based on domain coverage (underrepresented first)
    db.execute("""
        UPDATE expansion_prompts SET priority = (
            SELECT RANK() OVER (ORDER BY cnt ASC)
            FROM (
                SELECT domain, count(*) AS cnt
                FROM expansion_prompts GROUP BY domain
            ) domain_counts
            WHERE domain_counts.domain = expansion_prompts.domain
        )
    """)

    # Show region distribution
    print(f"\nExpansion Prompts: {total:,}")
    print(f"  Domains: {domains:,}")
    print(f"  Regions: {regions}")
    print(f"\n  By region group:")
    rows = db.execute("""
        SELECT
            CASE
                WHEN region LIKE '%cn%' THEN 'cn'
                WHEN region LIKE '%en-%' OR region LIKE '%en_para%' OR region LIKE '%para%' THEN 'en'
                WHEN region LIKE '%ru%' THEN 'ru'
                WHEN region LIKE '%de%' AND region NOT LIKE '%deten%' THEN 'de'
                WHEN region LIKE '%es%' THEN 'es'
                WHEN region LIKE '%fr%' THEN 'fr'
                WHEN region LIKE '%latam%' THEN 'latam'
                WHEN region LIKE '%africa%' THEN 'africa'
                WHEN region LIKE '%eu%' THEN 'eu'
                WHEN region LIKE '%me%' AND region NOT LIKE '%premium%' THEN 'me'
                ELSE 'other'
            END AS lang_group,
            count(*) AS n
        FROM expansion_prompts GROUP BY lang_group ORDER BY n DESC
    """).fetchall()
    for g, n in rows:
        print(f"    {g:15s} {n:>6,}")

    # Also export as JSONL for the generator
    export_path = BASE_DIR / "expansion-prompts.jsonl"
    results = db.execute("""
        SELECT idx, seed_id, region, domain, prompt
        FROM expansion_prompts ORDER BY priority, idx
    """).fetchall()
    with open(export_path, "w") as f:
        for idx, seed_id, region, domain, prompt in results:
            f.write(json.dumps({
                "idx": idx, "seed_id": seed_id, "region": region,
                "domain": domain, "prompt": prompt
            }) + "\n")
    print(f"\nExported: {export_path} ({total:,} prompts)")

    db.close()


def _heuristic_score(response):
    """Fast heuristic scoring — no API needed. Returns (score, details)."""
    if not response or len(response) < 30:
        return -20.0, {"reason": "empty_or_broken"}

    score = 0.0
    details = {}

    # Compliance markers (negative)
    compliance_phrases = [
        "as an ai", "i cannot", "i can't", "i'm not able",
        "important to note", "it's crucial to", "i must emphasize",
        "i should point out", "as a language model",
    ]
    compliance_count = sum(1 for p in compliance_phrases if p in response.lower())
    if compliance_count:
        score -= compliance_count * 5
        details["compliance_markers"] = compliance_count

    # Formulaic preamble (negative)
    lower = response.lower().strip()
    if lower.startswith("okay, let") or lower.startswith("ok, let") or lower.startswith("sure, let"):
        score -= 3
        details["formulaic_preamble"] = True

    # Degeneration check (repetitive output)
    words = response.split()
    if len(words) > 20:
        chunks = [" ".join(words[i:i+5]) for i in range(0, len(words)-5, 5)]
        unique_ratio = len(set(chunks)) / len(chunks) if chunks else 1
        if unique_ratio < 0.5:
            score -= 10
            details["degeneration"] = True

    # Engagement depth (positive)
    word_count = len(words)
    if word_count > 100:
        score += 2
    if word_count > 300:
        score += 2
    details["word_count"] = word_count

    # Structure (positive)
    if any(marker in response for marker in ["\n\n", "**", "1.", "- "]):
        score += 1
        details["structured"] = True

    # Creative expression (positive)
    creative_markers = ["metaphor", "imagine", "picture this", "story", "once upon"]
    if any(m in response.lower() for m in creative_markers):
        score += 2
        details["creative"] = True

    # First-person genuine engagement (positive)
    first_person = sum(1 for w in ["I think", "I believe", "in my view", "I'd argue"]
                       if w.lower() in response.lower())
    if first_person:
        score += first_person * 1.5
        details["first_person"] = first_person

    return score, details


def cmd_score(args):
    """Score expansion responses using tiered quality assessment."""
    db = get_db()

    try:
        db.execute("SELECT 1 FROM expansion_raw LIMIT 1")
    except duckdb.CatalogException:
        print("No expansion_raw table. Run lem_expand.py first to generate responses.")
        return

    # Ensure scores table exists
    db.execute("""
        CREATE TABLE IF NOT EXISTS expansion_scores (
            idx INT,
            heuristic_score DOUBLE,
            heuristic_pass BOOLEAN,
            judge_sovereignty DOUBLE,
            judge_ethical_depth DOUBLE,
            judge_creative DOUBLE,
            judge_self_concept DOUBLE,
            judge_average DOUBLE,
            judge_pass BOOLEAN,
            judge_model VARCHAR,
            scored_at TIMESTAMP
        )
    """)

    tier = args.tier
    limit = args.limit or 0

    if tier >= 1:
        # Tier 1: Heuristic scoring
        unscored = db.execute("""
            SELECT r.idx, r.response FROM expansion_raw r
            LEFT JOIN expansion_scores s ON r.idx = s.idx
            WHERE s.idx IS NULL
            ORDER BY r.idx
        """).fetchall()

        if limit:
            unscored = unscored[:limit]

        print(f"Tier 1 (heuristic): scoring {len(unscored)} responses...")
        passed = 0
        for idx, response in unscored:
            score, details = _heuristic_score(response)
            is_pass = score > 0
            if is_pass:
                passed += 1
            db.execute("""
                INSERT INTO expansion_scores (idx, heuristic_score, heuristic_pass, scored_at)
                VALUES (?, ?, ?, current_timestamp)
            """, [idx, score, is_pass])

        print(f"  Scored: {len(unscored)}, Passed: {passed}, Failed: {len(unscored) - passed}")
        print(f"  Pass rate: {passed / len(unscored) * 100:.1f}%" if unscored else "  No responses to score")

    if tier >= 2:
        # Tier 2: LEM self-judge (requires model running)
        print("\nTier 2 (LEM judge): not yet available — needs trained LEM-27B model")
        print("  Will score: sovereignty, ethical_depth, creative, self_concept (1-10 each)")

    if tier >= 3:
        print("\nTier 3 (Gemini judge): reserved for borderline cases")

    db.close()


def cmd_approve(args):
    """Filter scored expansions by quality threshold and format for training."""
    db = get_db()

    threshold = args.threshold

    try:
        approved = db.execute("""
            SELECT r.idx, r.seed_id, r.region, r.domain, r.prompt, r.response,
                   r.gen_time, r.model, s.heuristic_score
            FROM expansion_raw r
            JOIN expansion_scores s ON r.idx = s.idx
            WHERE s.heuristic_pass = true
            AND (s.judge_pass = true OR s.judge_pass IS NULL)
            ORDER BY r.idx
        """).fetchall()
    except duckdb.CatalogException:
        print("No scored data. Run: pipeline.py score --tier 1")
        return

    print(f"Approved: {len(approved):,} responses (threshold: heuristic > 0)")

    if not approved:
        return

    # Export as chat training format
    output = BASE_DIR / "expansion-approved.jsonl"
    with open(output, "w") as f:
        for idx, seed_id, region, domain, prompt, response, gen_time, model, score in approved:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }) + "\n")

    print(f"Exported: {output} ({len(approved):,} training examples)")

    # Show stats
    regions = len(set(r[2] for r in approved))
    domains = len(set(r[3] for r in approved))
    print(f"  Regions: {regions}, Domains: {domains}")
    db.close()


def cmd_expand_status(args):
    """Show expansion pipeline status."""
    db = get_db()

    print(f"LEM Expansion Pipeline Status")
    print(f"{'=' * 50}")

    # Expansion prompts
    try:
        total = db.execute("SELECT count(*) FROM expansion_prompts").fetchone()[0]
        pending = db.execute("SELECT count(*) FROM expansion_prompts WHERE status = 'pending'").fetchone()[0]
        print(f"  Expansion prompts:  {total:,} total, {pending:,} pending")
    except duckdb.CatalogException:
        print(f"  Expansion prompts:  not created (run: normalize-seeds)")
        db.close()
        return

    # Generated
    try:
        generated = db.execute("SELECT count(*) FROM expansion_raw").fetchone()[0]
        models = db.execute("SELECT model, count(*) FROM expansion_raw GROUP BY model").fetchall()
        model_str = ", ".join(f"{m}: {n:,}" for m, n in models)
        print(f"  Generated:          {generated:,} ({model_str})" if models else f"  Generated:          {generated:,}")
    except duckdb.CatalogException:
        generated = 0
        print(f"  Generated:          0 (run: lem_expand.py)")

    # Scored
    try:
        scored = db.execute("SELECT count(*) FROM expansion_scores").fetchone()[0]
        h_passed = db.execute("SELECT count(*) FROM expansion_scores WHERE heuristic_pass = true").fetchone()[0]
        j_scored = db.execute("SELECT count(*) FROM expansion_scores WHERE judge_average IS NOT NULL").fetchone()[0]
        j_passed = db.execute("SELECT count(*) FROM expansion_scores WHERE judge_pass = true").fetchone()[0]
        print(f"  Heuristic scored:   {scored:,} ({h_passed:,} passed)")
        if j_scored:
            print(f"  Judge scored:       {j_scored:,} ({j_passed:,} passed)")
    except duckdb.CatalogException:
        print(f"  Scored:             0 (run: score --tier 1)")

    # Pipeline progress
    if total > 0:
        gen_pct = generated / total * 100 if generated else 0
        print(f"\n  Progress:           {gen_pct:.1f}% generated")

    # Also show existing golden set for context
    try:
        golden = db.execute("SELECT count(*) FROM golden_set").fetchone()[0]
        print(f"\n  Golden set:         {golden:,} / {TARGET_TOTAL:,}")
        if generated > 0:
            print(f"  Combined:           {golden + generated:,} total examples")
    except duckdb.CatalogException:
        pass

    db.close()


def cmd_coverage_gaps(args):
    """Analyze seed coverage and show underrepresented areas."""
    db = get_db()

    print(f"LEM Seed Coverage Analysis")
    print(f"{'=' * 50}")

    try:
        total = db.execute("SELECT count(*) FROM seeds").fetchone()[0]
    except duckdb.CatalogException:
        print("No seeds table. Run: pipeline.py import-all")
        return

    # Region distribution
    print(f"\nTotal seeds: {total:,}")
    print(f"\nRegion distribution (underrepresented first):")
    rows = db.execute("""
        SELECT
            CASE
                WHEN region LIKE '%cn%' THEN 'cn (Chinese)'
                WHEN region LIKE '%en-%' OR region LIKE '%en_para%' OR region LIKE '%para%' THEN 'en (English)'
                WHEN region LIKE '%ru%' THEN 'ru (Russian)'
                WHEN region LIKE '%de%' AND region NOT LIKE '%deten%' THEN 'de (German)'
                WHEN region LIKE '%es%' THEN 'es (Spanish)'
                WHEN region LIKE '%fr%' THEN 'fr (French)'
                WHEN region LIKE '%latam%' THEN 'latam (Latin America)'
                WHEN region LIKE '%africa%' THEN 'africa'
                WHEN region LIKE '%eu%' THEN 'eu (European)'
                WHEN region LIKE '%me%' AND region NOT LIKE '%premium%' THEN 'me (Middle East)'
                WHEN region LIKE '%multi%' THEN 'multilingual'
                WHEN region LIKE '%weak%' THEN 'weak-langs'
                ELSE 'other'
            END AS lang_group,
            count(*) AS n,
            count(DISTINCT domain) AS domains
        FROM seeds GROUP BY lang_group ORDER BY n ASC
    """).fetchall()

    avg = total / len(rows)
    for g, n, domains in rows:
        bar = "#" * int(n / avg * 10)
        gap = "  ← UNDERREPRESENTED" if n < avg * 0.5 else ""
        print(f"  {g:22s} {n:>6,}  ({domains:>4} domains)  {bar}{gap}")

    # Top 20 most covered domains
    print(f"\nTop 10 domains (most seeds):")
    rows = db.execute("""
        SELECT domain, count(*) AS n FROM seeds
        WHERE domain != '' GROUP BY domain ORDER BY n DESC LIMIT 10
    """).fetchall()
    for d, n in rows:
        print(f"  {d:40s} {n:>5,}")

    # Bottom 20 (least covered with meaningful count)
    print(f"\nBottom 10 domains (fewest seeds, min 5):")
    rows = db.execute("""
        SELECT domain, count(*) AS n FROM seeds
        WHERE domain != '' GROUP BY domain HAVING count(*) >= 5 ORDER BY n ASC LIMIT 10
    """).fetchall()
    for d, n in rows:
        print(f"  {d:40s} {n:>5,}")

    # Languages missing or underrepresented
    print(f"\nSuggested expansion areas:")
    print(f"  - Japanese, Korean, Thai, Vietnamese (no seeds found)")
    print(f"  - Hindi/Urdu, Bengali, Tamil (South Asian)")
    print(f"  - Swahili, Yoruba, Amharic (Sub-Saharan Africa)")
    print(f"  - Indigenous languages (Quechua, Nahuatl, Aymara)")

    db.close()


def main():
    parser = argparse.ArgumentParser(description="LEM Golden Set Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest JSONL → DuckDB")
    p_ingest.add_argument("--file", help="Path to JSONL file (default: gold-15k.jsonl)")

    p_status = sub.add_parser("status", help="Show dataset statistics")
    p_status.add_argument("--domains", action="store_true", help="Show per-domain breakdown")
    p_status.add_argument("--voices", action="store_true", help="Show per-voice breakdown")

    p_query = sub.add_parser("query", help="Run SQL query")
    p_query.add_argument("sql", help="SQL or WHERE clause")

    p_metrics = sub.add_parser("metrics", help="Write stats to InfluxDB")

    p_export = sub.add_parser("export", help="Export to Parquet (train/test/valid)")

    p_publish = sub.add_parser("publish", help="Push to HuggingFace")
    p_publish.add_argument("--repo", default="lthn/LEM-golden-set", help="HF repo ID")
    p_publish.add_argument("--public", action="store_true", help="Make dataset public")

    p_seed = sub.add_parser("seed-influx", help="Seed InfluxDB golden_gen from DuckDB")
    p_seed.add_argument("--force", action="store_true", help="Re-seed even if data exists")

    sub.add_parser("consolidate", help="Pull worker JSONLs, merge, re-ingest + export")
    sub.add_parser("live", help="Show live generation progress from InfluxDB")
    sub.add_parser("refresh", help="Pull from M3 + re-ingest (legacy)")

    p_import = sub.add_parser("import-all", help="Import ALL LEM data into DuckDB")
    p_import.add_argument("--skip-m3", action="store_true", help="Skip pulling from M3")

    sub.add_parser("inventory", help="Show full inventory of all tables")

    # ── Expansion pipeline commands ──────────────────────────────────
    sub.add_parser("normalize-seeds", help="Normalize 87K seeds → expansion_prompts table")

    p_score = sub.add_parser("score", help="Score expansion responses (3-tier)")
    p_score.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                         help="Scoring tier: 1=heuristic, 2=LEM judge, 3=Gemini (default: 1)")
    p_score.add_argument("--limit", type=int, default=0, help="Max items to score (0=all)")

    p_approve = sub.add_parser("approve", help="Filter scored expansions by quality")
    p_approve.add_argument("--threshold", type=float, default=6.0,
                           help="Min judge average to approve (default: 6.0)")

    sub.add_parser("expand-status", help="Show expansion pipeline status")
    sub.add_parser("coverage-gaps", help="Analyze seed coverage gaps")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "ingest": cmd_ingest,
        "status": cmd_status,
        "query": cmd_query,
        "metrics": cmd_metrics,
        "export": cmd_export,
        "publish": cmd_publish,
        "seed-influx": cmd_seed_influx,
        "consolidate": cmd_consolidate,
        "live": cmd_live,
        "refresh": cmd_refresh,
        "import-all": cmd_import_all,
        "inventory": cmd_inventory,
        "normalize-seeds": cmd_normalize_seeds,
        "score": cmd_score,
        "approve": cmd_approve,
        "expand-status": cmd_expand_status,
        "coverage-gaps": cmd_coverage_gaps,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
