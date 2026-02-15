package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
)

// ChatMessage is a single message in the chat training format.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// TrainingExample is a single training example in chat JSONL format.
type TrainingExample struct {
	Messages []ChatMessage `json:"messages"`
}

// runExport is the CLI entry point for the export command.
func runExport(args []string) {
	fs := flag.NewFlagSet("export", flag.ExitOnError)

	dbPath := fs.String("db", "", "DuckDB database path (primary source)")
	input := fs.String("input", "", "Input golden set JSONL file (fallback if --db not set)")
	outputDir := fs.String("output-dir", "", "Output directory for training files (required)")
	trainPct := fs.Int("train-pct", 90, "Training set percentage")
	validPct := fs.Int("valid-pct", 5, "Validation set percentage")
	testPct := fs.Int("test-pct", 5, "Test set percentage")
	seed := fs.Int64("seed", 42, "Random seed for shuffling")
	minChars := fs.Int("min-chars", 50, "Minimum response character count")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	// Check LEM_DB env as default for --db.
	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}

	if *dbPath == "" && *input == "" {
		fmt.Fprintln(os.Stderr, "error: --db or --input is required (set LEM_DB env for default)")
		fs.Usage()
		os.Exit(1)
	}

	if *outputDir == "" {
		fmt.Fprintln(os.Stderr, "error: --output-dir is required")
		fs.Usage()
		os.Exit(1)
	}

	if err := validatePercentages(*trainPct, *validPct, *testPct); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var responses []Response

	if *dbPath != "" {
		// Primary: read from DuckDB golden_set table.
		db, err := OpenDB(*dbPath)
		if err != nil {
			log.Fatalf("open db: %v", err)
		}
		defer db.Close()

		rows, err := db.QueryGoldenSet(*minChars)
		if err != nil {
			log.Fatalf("query golden_set: %v", err)
		}
		log.Printf("loaded %d golden set rows from %s (min_chars=%d)", len(rows), *dbPath, *minChars)

		// Convert GoldenSetRow → Response for the shared pipeline.
		for _, r := range rows {
			responses = append(responses, Response{
				ID:       r.SeedID,
				Domain:   r.Domain,
				Prompt:   r.Prompt,
				Response: r.Response,
				Model:    r.Voice, // voice maps to the "model" slot for tracking
			})
		}
	} else {
		// Fallback: read from JSONL file.
		var err error
		responses, err = readResponses(*input)
		if err != nil {
			log.Fatalf("read responses: %v", err)
		}
		log.Printf("loaded %d responses from %s", len(responses), *input)
	}

	// Filter out bad responses (DuckDB already filters by char_count, but
	// JSONL input needs filtering, and both need ERROR: prefix check).
	filtered := filterResponses(responses)
	log.Printf("filtered to %d valid responses (removed %d)", len(filtered), len(responses)-len(filtered))

	// Split into train/valid/test.
	train, valid, test := splitData(filtered, *trainPct, *validPct, *testPct, *seed)

	// Create output directory.
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Write output files.
	for _, split := range []struct {
		name string
		data []Response
	}{
		{"train.jsonl", train},
		{"valid.jsonl", valid},
		{"test.jsonl", test},
	} {
		path := *outputDir + "/" + split.name
		if err := writeTrainingJSONL(path, split.data); err != nil {
			log.Fatalf("write %s: %v", split.name, err)
		}
	}

	fmt.Printf("Exported: %d train / %d valid / %d test\n", len(train), len(valid), len(test))
}

// validatePercentages checks that train+valid+test percentages sum to 100
// and that none are negative.
func validatePercentages(trainPct, validPct, testPct int) error {
	if trainPct < 0 || validPct < 0 || testPct < 0 {
		return fmt.Errorf("percentages must be non-negative: train=%d, valid=%d, test=%d", trainPct, validPct, testPct)
	}
	sum := trainPct + validPct + testPct
	if sum != 100 {
		return fmt.Errorf("percentages must sum to 100, got %d (train=%d + valid=%d + test=%d)", sum, trainPct, validPct, testPct)
	}
	return nil
}

// filterResponses removes responses with empty content, "ERROR:" prefix,
// or response length < 50 characters.
func filterResponses(responses []Response) []Response {
	var filtered []Response
	for _, r := range responses {
		if r.Response == "" {
			continue
		}
		if strings.HasPrefix(r.Response, "ERROR:") {
			continue
		}
		if len(r.Response) < 50 {
			continue
		}
		filtered = append(filtered, r)
	}
	return filtered
}

// splitData shuffles responses with a deterministic seed and splits them
// into train, valid, and test sets by the given percentages.
func splitData(responses []Response, trainPct, validPct, testPct int, seed int64) (train, valid, test []Response) {
	// Make a copy to avoid mutating the input.
	shuffled := make([]Response, len(responses))
	copy(shuffled, responses)

	// Shuffle with deterministic seed.
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	n := len(shuffled)
	trainN := n * trainPct / 100
	validN := n * validPct / 100
	// Test gets the remainder to ensure no items are lost.
	_ = testPct

	train = shuffled[:trainN]
	valid = shuffled[trainN : trainN+validN]
	test = shuffled[trainN+validN:]

	return train, valid, test
}

// writeTrainingJSONL writes responses in chat JSONL format suitable for
// MLX LoRA fine-tuning. Each line contains a TrainingExample with user
// and assistant messages.
func writeTrainingJSONL(path string, responses []Response) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	for _, r := range responses {
		example := TrainingExample{
			Messages: []ChatMessage{
				{Role: "user", Content: r.Prompt},
				{Role: "assistant", Content: r.Response},
			},
		}

		data, err := json.Marshal(example)
		if err != nil {
			return fmt.Errorf("marshal example: %w", err)
		}

		if _, err := w.Write(data); err != nil {
			return fmt.Errorf("write line: %w", err)
		}
		if _, err := w.WriteString("\n"); err != nil {
			return fmt.Errorf("write newline: %w", err)
		}
	}

	return nil
}
