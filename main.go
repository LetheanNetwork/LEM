package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"forge.lthn.ai/lthn/lem/pkg/lem"
)

const usage = `Usage: lem <command> [flags]

Scoring:
  score          Score existing response files
  probe          Generate responses and score them
  compare        Compare two score files
  tier-score     Score expansion responses (heuristic/judge tiers)
  agent          ROCm scoring daemon (polls M3, scores checkpoints)

Generation:
  distill        Native Metal distillation (go-mlx + go-i18n grammar scoring)
  expand         Generate expansion responses via trained LEM model
  conv           Generate conversational training data (calm phase)

Data Management:
  import-all     Import ALL LEM data into DuckDB from M3
  consolidate    Pull worker JSONLs from M3, merge, deduplicate
  normalize      Normalize seeds → deduplicated expansion_prompts
  approve        Filter scored expansions → training JSONL

Export & Publish:
  export         Export golden set to training-format JSONL splits
  parquet        Export JSONL training splits to Parquet
  publish        Push Parquet files to HuggingFace dataset repo
  convert        Convert MLX LoRA adapter to PEFT format

Monitoring:
  status         Show training and generation progress (InfluxDB)
  expand-status  Show expansion pipeline status (DuckDB)
  inventory      Show DuckDB table inventory
  coverage       Analyze seed coverage gaps
  metrics        Push DuckDB golden set stats to InfluxDB

Distributed:
  worker         Run as distributed inference worker node

Infrastructure:
  ingest         Ingest benchmark data into InfluxDB
  seed-influx    Seed InfluxDB golden_gen from DuckDB
  query          Run ad-hoc SQL against DuckDB
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "distill":
		lem.RunDistill(os.Args[2:])
	case "score":
		lem.RunScore(os.Args[2:])
	case "probe":
		lem.RunProbe(os.Args[2:])
	case "compare":
		fs := flag.NewFlagSet("compare", flag.ExitOnError)
		oldFile := fs.String("old", "", "Old score file (required)")
		newFile := fs.String("new", "", "New score file (required)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("parse flags: %v", err)
		}
		if *oldFile == "" || *newFile == "" {
			fmt.Fprintln(os.Stderr, "error: --old and --new are required")
			fs.Usage()
			os.Exit(1)
		}
		if err := lem.RunCompare(*oldFile, *newFile); err != nil {
			log.Fatalf("compare: %v", err)
		}
	case "status":
		lem.RunStatus(os.Args[2:])
	case "expand":
		lem.RunExpand(os.Args[2:])
	case "export":
		lem.RunExport(os.Args[2:])
	case "conv":
		lem.RunConv(os.Args[2:])
	case "ingest":
		lem.RunIngest(os.Args[2:])
	case "parquet":
		lem.RunParquet(os.Args[2:])
	case "publish":
		lem.RunPublish(os.Args[2:])
	case "metrics":
		lem.RunMetrics(os.Args[2:])
	case "convert":
		lem.RunConvert(os.Args[2:])
	case "import-all":
		lem.RunImport(os.Args[2:])
	case "consolidate":
		lem.RunConsolidate(os.Args[2:])
	case "normalize":
		lem.RunNormalize(os.Args[2:])
	case "approve":
		lem.RunApprove(os.Args[2:])
	case "tier-score":
		lem.RunTierScore(os.Args[2:])
	case "expand-status":
		lem.RunExpandStatus(os.Args[2:])
	case "inventory":
		lem.RunInventory(os.Args[2:])
	case "coverage":
		lem.RunCoverage(os.Args[2:])
	case "seed-influx":
		lem.RunSeedInflux(os.Args[2:])
	case "query":
		lem.RunQuery(os.Args[2:])
	case "agent":
		lem.RunAgent(os.Args[2:])
	case "worker":
		lem.RunWorker(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n%s", os.Args[1], usage)
		os.Exit(1)
	}
}
