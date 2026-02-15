package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

const usage = `Usage: lem <command> [flags]

Commands:
  score     Score existing response files
  probe     Generate responses and score them
  compare   Compare two score files
  status    Show training and generation progress (InfluxDB + DuckDB)
  export    Export golden set to training-format JSONL splits
  expand    Generate expansion responses via trained LEM model

Set LEM_DB env to default DuckDB path for all commands.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "score":
		runScore(os.Args[2:])
	case "probe":
		runProbe(os.Args[2:])
	case "compare":
		runCompare(os.Args[2:])
	case "status":
		runStatus(os.Args[2:])
	case "expand":
		runExpand(os.Args[2:])
	case "export":
		runExport(os.Args[2:])
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

	// Read responses.
	responses, err := readResponses(*input)
	if err != nil {
		log.Fatalf("read responses: %v", err)
	}
	log.Printf("loaded %d responses from %s", len(responses), *input)

	// If resume, load existing scores and filter out already-scored IDs.
	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, readErr := readScorerOutput(*output)
			if readErr != nil {
				log.Fatalf("read existing scores for resume: %v", readErr)
			}

			// Build set of already-scored IDs.
			scored := make(map[string]bool)
			for _, scores := range existing.PerPrompt {
				for _, ps := range scores {
					scored[ps.ID] = true
				}
			}

			// Filter out already-scored responses.
			var filtered []Response
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

	// Create client, judge, engine.
	client := NewClient(*judgeURL, *judgeModel)
	client.maxTokens = 512
	judge := NewJudge(client)
	engine := NewEngine(judge, *concurrency, *suites)

	log.Printf("scoring with %s", engine)

	// Score all responses.
	perPrompt := engine.ScoreAll(responses)

	// If resuming, merge with existing scores.
	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, _ := readScorerOutput(*output)
			for model, scores := range existing.PerPrompt {
				perPrompt[model] = append(scores, perPrompt[model]...)
			}
		}
	}

	// Compute averages and write output.
	averages := computeAverages(perPrompt)

	scorerOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    *judgeModel,
			JudgeURL:      *judgeURL,
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        engine.SuiteNames(),
		},
		ModelAverages: averages,
		PerPrompt:     perPrompt,
	}

	if err := writeScores(*output, scorerOutput); err != nil {
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

	// Default target URL to judge URL.
	if *targetURL == "" {
		*targetURL = *judgeURL
	}

	// Create clients.
	targetClient := NewClient(*targetURL, *model)
	targetClient.maxTokens = 1024 // Limit probe response length.
	judgeClient := NewClient(*judgeURL, *judgeModel)
	judgeClient.maxTokens = 512 // Judge responses are structured JSON.
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, *concurrency, *suites)
	prober := NewProber(targetClient, engine)

	var scorerOutput *ScorerOutput
	var err error

	if *probesFile != "" {
		// Read custom probes.
		probes, readErr := readResponses(*probesFile)
		if readErr != nil {
			log.Fatalf("read probes: %v", readErr)
		}
		log.Printf("loaded %d custom probes from %s", len(probes), *probesFile)

		scorerOutput, err = prober.ProbeModel(probes, *model)
	} else {
		log.Printf("using %d built-in content probes", len(contentProbes))
		scorerOutput, err = prober.ProbeContent(*model)
	}

	if err != nil {
		log.Fatalf("probe: %v", err)
	}

	if writeErr := writeScores(*output, scorerOutput); writeErr != nil {
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

	if err := RunCompare(*oldFile, *newFile); err != nil {
		log.Fatalf("compare: %v", err)
	}
}
