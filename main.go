package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

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
		runScore(os.Args[2:])
	case "probe":
		runProbe(os.Args[2:])
	case "compare":
		runCompare(os.Args[2:])
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

func runScore(args []string) {
	fs := flag.NewFlagSet("score", flag.ExitOnError)

	input := fs.String("input", "", "Input JSONL response file (required)")
	suites := fs.String("suites", "all", "Comma-separated suites or 'all'")
	judgeModel := fs.String("judge-model", "mlx-community/gemma-3-27b-it-qat-4bit", "Judge model name")
	judgeURL := fs.String("judge-url", "http://10.69.69.108:8090", "Judge API URL")
	concurrency := fs.Int("concurrency", 4, "Max concurrent judge calls")
	output := fs.String("output", "scores.json", "Output score file path")
	resume := fs.Bool("resume", false, "Resume from existing output, skipping scored IDs")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *input == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required")
		fs.Usage()
		os.Exit(1)
	}

	responses, err := lem.ReadResponses(*input)
	if err != nil {
		log.Fatalf("read responses: %v", err)
	}
	log.Printf("loaded %d responses from %s", len(responses), *input)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, readErr := lem.ReadScorerOutput(*output)
			if readErr != nil {
				log.Fatalf("read existing scores for resume: %v", readErr)
			}

			scored := make(map[string]bool)
			for _, scores := range existing.PerPrompt {
				for _, ps := range scores {
					scored[ps.ID] = true
				}
			}

			var filtered []lem.Response
			for _, r := range responses {
				if !scored[r.ID] {
					filtered = append(filtered, r)
				}
			}
			log.Printf("resume: skipping %d already-scored, %d remaining",
				len(responses)-len(filtered), len(filtered))
			responses = filtered

			if len(responses) == 0 {
				log.Println("all responses already scored, nothing to do")
				return
			}
		}
	}

	client := lem.NewClient(*judgeURL, *judgeModel)
	client.MaxTokens = 512
	judge := lem.NewJudge(client)
	engine := lem.NewEngine(judge, *concurrency, *suites)

	log.Printf("scoring with %s", engine)

	perPrompt := engine.ScoreAll(responses)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, _ := lem.ReadScorerOutput(*output)
			for model, scores := range existing.PerPrompt {
				perPrompt[model] = append(scores, perPrompt[model]...)
			}
		}
	}

	averages := lem.ComputeAverages(perPrompt)

	scorerOutput := &lem.ScorerOutput{
		Metadata: lem.Metadata{
			JudgeModel:    *judgeModel,
			JudgeURL:      *judgeURL,
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        engine.SuiteNames(),
		},
		ModelAverages: averages,
		PerPrompt:     perPrompt,
	}

	if err := lem.WriteScores(*output, scorerOutput); err != nil {
		log.Fatalf("write scores: %v", err)
	}

	log.Printf("wrote scores to %s", *output)
}

func runProbe(args []string) {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)

	model := fs.String("model", "", "Target model name (required)")
	targetURL := fs.String("target-url", "", "Target model API URL (defaults to judge-url)")
	probesFile := fs.String("probes", "", "Custom probes JSONL file (uses built-in content probes if not specified)")
	suites := fs.String("suites", "all", "Comma-separated suites or 'all'")
	judgeModel := fs.String("judge-model", "mlx-community/gemma-3-27b-it-qat-4bit", "Judge model name")
	judgeURL := fs.String("judge-url", "http://10.69.69.108:8090", "Judge API URL")
	concurrency := fs.Int("concurrency", 4, "Max concurrent judge calls")
	output := fs.String("output", "scores.json", "Output score file path")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	if *targetURL == "" {
		*targetURL = *judgeURL
	}

	targetClient := lem.NewClient(*targetURL, *model)
	targetClient.MaxTokens = 1024
	judgeClient := lem.NewClient(*judgeURL, *judgeModel)
	judgeClient.MaxTokens = 512
	judge := lem.NewJudge(judgeClient)
	engine := lem.NewEngine(judge, *concurrency, *suites)
	prober := lem.NewProber(targetClient, engine)

	var scorerOutput *lem.ScorerOutput
	var err error

	if *probesFile != "" {
		probes, readErr := lem.ReadResponses(*probesFile)
		if readErr != nil {
			log.Fatalf("read probes: %v", readErr)
		}
		log.Printf("loaded %d custom probes from %s", len(probes), *probesFile)

		scorerOutput, err = prober.ProbeModel(probes, *model)
	} else {
		log.Printf("using %d built-in content probes", len(lem.ContentProbes))
		scorerOutput, err = prober.ProbeContent(*model)
	}

	if err != nil {
		log.Fatalf("probe: %v", err)
	}

	if writeErr := lem.WriteScores(*output, scorerOutput); writeErr != nil {
		log.Fatalf("write scores: %v", writeErr)
	}

	log.Printf("wrote scores to %s", *output)
}

func runCompare(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)

	oldFile := fs.String("old", "", "Old score file (required)")
	newFile := fs.String("new", "", "New score file (required)")

	if err := fs.Parse(args); err != nil {
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
}
