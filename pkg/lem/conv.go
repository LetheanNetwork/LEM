package lem

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

// RunConv is the CLI entry point for the conv command.
// It generates multi-turn conversational training data from built-in
// seed conversations plus optional extra files and golden set data.
func RunConv(args []string) {
	fs := flag.NewFlagSet("conv", flag.ExitOnError)

	outputDir := fs.String("output-dir", "", "Output directory for training files (required)")
	extra := fs.String("extra", "", "Additional conversations JSONL file (multi-turn format)")
	golden := fs.String("golden", "", "Golden set JSONL to convert to single-turn conversations")
	dbPath := fs.String("db", "", "DuckDB database path for golden set (alternative to --golden)")
	trainPct := fs.Int("train-pct", 80, "Training set percentage")
	validPct := fs.Int("valid-pct", 10, "Validation set percentage")
	testPct := fs.Int("test-pct", 10, "Test set percentage")
	seed := fs.Int64("seed", 42, "Random seed for shuffling")
	minChars := fs.Int("min-chars", 50, "Minimum response chars for golden set conversion")
	noBuiltin := fs.Bool("no-builtin", false, "Exclude built-in seed conversations")
	influxURL := fs.String("influx", "", "InfluxDB URL for progress reporting")
	influxDB := fs.String("influx-db", "", "InfluxDB database name")
	worker := fs.String("worker", "", "Worker hostname for InfluxDB reporting")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
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

	// Check LEM_DB env as default for --db.
	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}

	// Default worker to hostname.
	if *worker == "" {
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "unknown"
		}
		*worker = hostname
	}

	// Collect all conversations.
	var conversations []TrainingExample

	// 1. Built-in seed conversations.
	if !*noBuiltin {
		conversations = append(conversations, SeedConversations...)
		log.Printf("loaded %d built-in seed conversations", len(SeedConversations))
	}

	// 2. Extra conversations from file.
	if *extra != "" {
		extras, err := readConversations(*extra)
		if err != nil {
			log.Fatalf("read extra conversations: %v", err)
		}
		conversations = append(conversations, extras...)
		log.Printf("loaded %d extra conversations from %s", len(extras), *extra)
	}

	// 3. Golden set responses converted to single-turn format.
	var goldenResponses []Response
	if *dbPath != "" && *golden == "" {
		db, err := OpenDB(*dbPath)
		if err != nil {
			log.Fatalf("open db: %v", err)
		}
		defer db.Close()

		rows, err := db.QueryGoldenSet(*minChars)
		if err != nil {
			log.Fatalf("query golden_set: %v", err)
		}
		for _, r := range rows {
			goldenResponses = append(goldenResponses, Response{
				ID:       r.SeedID,
				Domain:   r.Domain,
				Prompt:   r.Prompt,
				Response: r.Response,
				Model:    r.Voice,
			})
		}
		log.Printf("loaded %d golden set rows from %s", len(goldenResponses), *dbPath)
	} else if *golden != "" {
		var err error
		goldenResponses, err = ReadResponses(*golden)
		if err != nil {
			log.Fatalf("read golden set: %v", err)
		}
		log.Printf("loaded %d golden set responses from %s", len(goldenResponses), *golden)
	}

	if len(goldenResponses) > 0 {
		converted := convertToConversations(goldenResponses, *minChars)
		conversations = append(conversations, converted...)
		log.Printf("converted %d golden set responses to single-turn conversations", len(converted))
	}

	if len(conversations) == 0 {
		log.Fatal("no conversations to process — use built-in seeds, --extra, --golden, or --db")
	}

	// Split into train/valid/test.
	train, valid, test := splitConversations(conversations, *trainPct, *validPct, *testPct, *seed)

	// Create output directory.
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Write output files.
	for _, split := range []struct {
		name string
		data []TrainingExample
	}{
		{"train.jsonl", train},
		{"valid.jsonl", valid},
		{"test.jsonl", test},
	} {
		path := *outputDir + "/" + split.name
		if err := writeConversationJSONL(path, split.data); err != nil {
			log.Fatalf("write %s: %v", split.name, err)
		}
	}

	// Stats.
	totalTurns := 0
	totalAssistantWords := 0
	assistantMsgCount := 0
	for _, c := range conversations {
		totalTurns += len(c.Messages)
		for _, m := range c.Messages {
			if m.Role == "assistant" {
				totalAssistantWords += len(strings.Fields(m.Content))
				assistantMsgCount++
			}
		}
	}
	avgTurns := float64(totalTurns) / float64(len(conversations))
	avgWords := 0.0
	if assistantMsgCount > 0 {
		avgWords = float64(totalAssistantWords) / float64(assistantMsgCount)
	}

	fmt.Printf("Conversational training data generated:\n")
	fmt.Printf("  %d train / %d valid / %d test\n", len(train), len(valid), len(test))
	fmt.Printf("  %d total conversations\n", len(conversations))
	fmt.Printf("  %d total turns (%.1f avg per conversation)\n", totalTurns, avgTurns)
	fmt.Printf("  %.0f words avg per assistant response\n", avgWords)
	fmt.Printf("  Output: %s/\n", *outputDir)

	// Report to InfluxDB if configured.
	influx := NewInfluxClient(*influxURL, *influxDB)
	line := fmt.Sprintf("conv_export,worker=%s total=%di,train=%di,valid=%di,test=%di,turns=%di,avg_turns=%f,avg_words=%f",
		escapeLp(*worker), len(conversations), len(train), len(valid), len(test),
		totalTurns, avgTurns, avgWords)
	if err := influx.WriteLp([]string{line}); err != nil {
		log.Printf("influx write (best-effort): %v", err)
	}
}

// readConversations reads multi-turn conversations from a JSONL file.
// Each line must be a TrainingExample with a messages array.
func readConversations(path string) ([]TrainingExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var conversations []TrainingExample
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var te TrainingExample
		if err := json.Unmarshal([]byte(line), &te); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if len(te.Messages) >= 2 {
			conversations = append(conversations, te)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}

	return conversations, nil
}

// convertToConversations converts golden set prompt/response pairs into
// single-turn TrainingExample conversations (user → assistant).
func convertToConversations(responses []Response, minChars int) []TrainingExample {
	var conversations []TrainingExample
	for _, r := range responses {
		if r.Response == "" || len(r.Response) < minChars {
			continue
		}
		if strings.HasPrefix(r.Response, "ERROR:") {
			continue
		}
		conversations = append(conversations, TrainingExample{
			Messages: []ChatMessage{
				{Role: "user", Content: r.Prompt},
				{Role: "assistant", Content: r.Response},
			},
		})
	}
	return conversations
}

// splitConversations shuffles conversations with a deterministic seed and
// splits them into train, valid, and test sets by percentage.
func splitConversations(conversations []TrainingExample, trainPct, validPct, testPct int, seed int64) (train, valid, test []TrainingExample) {
	shuffled := make([]TrainingExample, len(conversations))
	copy(shuffled, conversations)

	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	n := len(shuffled)
	trainN := n * trainPct / 100
	validN := n * validPct / 100
	_ = testPct

	train = shuffled[:trainN]
	valid = shuffled[trainN : trainN+validN]
	test = shuffled[trainN+validN:]

	// Ensure at least 1 in each split when we have enough data.
	if len(valid) == 0 && len(train) > 1 {
		valid = train[len(train)-1:]
		train = train[:len(train)-1]
	}
	if len(test) == 0 && len(train) > 1 {
		test = train[len(train)-1:]
		train = train[:len(train)-1]
	}

	return train, valid, test
}

// writeConversationJSONL writes TrainingExample conversations to a JSONL file.
func writeConversationJSONL(path string, conversations []TrainingExample) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	for _, c := range conversations {
		data, err := json.Marshal(c)
		if err != nil {
			return fmt.Errorf("marshal conversation: %w", err)
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
