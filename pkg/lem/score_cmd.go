package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

// RunScore scores existing response files using a judge model.
func RunScore(args []string) {
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

	responses, err := ReadResponses(*input)
	if err != nil {
		log.Fatalf("read responses: %v", err)
	}
	log.Printf("loaded %d responses from %s", len(responses), *input)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, readErr := ReadScorerOutput(*output)
			if readErr != nil {
				log.Fatalf("read existing scores for resume: %v", readErr)
			}

			scored := make(map[string]bool)
			for _, scores := range existing.PerPrompt {
				for _, ps := range scores {
					scored[ps.ID] = true
				}
			}

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

	client := NewClient(*judgeURL, *judgeModel)
	client.MaxTokens = 512
	judge := NewJudge(client)
	engine := NewEngine(judge, *concurrency, *suites)

	log.Printf("scoring with %s", engine)

	perPrompt := engine.ScoreAll(responses)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, readErr := ReadScorerOutput(*output)
			if readErr != nil {
				log.Fatalf("re-read scores for merge: %v", readErr)
			}
			for model, scores := range existing.PerPrompt {
				perPrompt[model] = append(scores, perPrompt[model]...)
			}
		}
	}

	averages := ComputeAverages(perPrompt)

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

	if err := WriteScores(*output, scorerOutput); err != nil {
		log.Fatalf("write scores: %v", err)
	}

	log.Printf("wrote scores to %s", *output)
}

// RunProbe generates responses from a target model and scores them.
func RunProbe(args []string) {
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

	targetClient := NewClient(*targetURL, *model)
	targetClient.MaxTokens = 1024
	judgeClient := NewClient(*judgeURL, *judgeModel)
	judgeClient.MaxTokens = 512
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, *concurrency, *suites)
	prober := NewProber(targetClient, engine)

	var scorerOutput *ScorerOutput
	var err error

	if *probesFile != "" {
		probes, readErr := ReadResponses(*probesFile)
		if readErr != nil {
			log.Fatalf("read probes: %v", readErr)
		}
		log.Printf("loaded %d custom probes from %s", len(probes), *probesFile)

		scorerOutput, err = prober.ProbeModel(probes, *model)
	} else {
		log.Printf("using %d built-in content probes", len(ContentProbes))
		scorerOutput, err = prober.ProbeContent(*model)
	}

	if err != nil {
		log.Fatalf("probe: %v", err)
	}

	if writeErr := WriteScores(*output, scorerOutput); writeErr != nil {
		log.Fatalf("write scores: %v", writeErr)
	}

	log.Printf("wrote scores to %s", *output)
}
