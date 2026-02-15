package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// expandOutput is the JSONL output structure for expansion generation.
// It extends the core Response fields with a chars count.
type expandOutput struct {
	ID             string  `json:"id"`
	Domain         string  `json:"domain,omitempty"`
	Prompt         string  `json:"prompt"`
	Response       string  `json:"response"`
	Model          string  `json:"model"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
	Chars          int     `json:"chars"`
}

// runExpand parses CLI flags and runs the expand command.
func runExpand(args []string) {
	fs := flag.NewFlagSet("expand", flag.ExitOnError)

	model := fs.String("model", "", "Model name for generation (required)")
	dbPath := fs.String("db", "", "DuckDB database path (primary prompt source)")
	prompts := fs.String("prompts", "", "Input JSONL file with expansion prompts (fallback)")
	apiURL := fs.String("api-url", "http://10.69.69.108:8090", "OpenAI-compatible API URL")
	worker := fs.String("worker", "", "Worker hostname (defaults to os.Hostname())")
	limit := fs.Int("limit", 0, "Max prompts to process (0 = all)")
	output := fs.String("output", ".", "Output directory for JSONL files")
	influxURL := fs.String("influx", "", "InfluxDB URL (default http://10.69.69.165:8181)")
	influxDB := fs.String("influx-db", "", "InfluxDB database name (default training)")
	dryRun := fs.Bool("dry-run", false, "Print plan and exit without generating")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	// Check LEM_DB env as default for --db.
	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}

	if *dbPath == "" && *prompts == "" {
		fmt.Fprintln(os.Stderr, "error: --db or --prompts is required (set LEM_DB env for default)")
		fs.Usage()
		os.Exit(1)
	}

	// Default worker to hostname.
	if *worker == "" {
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "unknown"
		}
		*worker = hostname
	}

	// Load prompts from DuckDB or JSONL.
	var promptList []Response
	var duckDB *DB

	if *dbPath != "" {
		var err error
		duckDB, err = OpenDBReadWrite(*dbPath)
		if err != nil {
			log.Fatalf("open db: %v", err)
		}
		defer duckDB.Close()

		rows, err := duckDB.QueryExpansionPrompts("pending", *limit)
		if err != nil {
			log.Fatalf("query expansion_prompts: %v", err)
		}
		log.Printf("loaded %d pending prompts from %s", len(rows), *dbPath)

		for _, r := range rows {
			prompt := r.Prompt
			if prompt == "" && r.PromptEn != "" {
				prompt = r.PromptEn // Use English translation if primary is empty.
			}
			promptList = append(promptList, Response{
				ID:     r.SeedID,
				Domain: r.Domain,
				Prompt: prompt,
			})
		}
	} else {
		var err error
		promptList, err = readResponses(*prompts)
		if err != nil {
			log.Fatalf("read prompts: %v", err)
		}
		log.Printf("loaded %d prompts from %s", len(promptList), *prompts)
	}

	// Create clients.
	client := NewClient(*apiURL, *model)
	client.maxTokens = 2048
	influx := NewInfluxClient(*influxURL, *influxDB)

	if err := expandPrompts(client, influx, duckDB, promptList, *model, *worker, *output, *dryRun, *limit); err != nil {
		log.Fatalf("expand: %v", err)
	}
}

// getCompletedIDs queries InfluxDB for prompt IDs that have already been
// processed in the expansion_gen measurement. Returns a set of completed IDs.
func getCompletedIDs(influx *InfluxClient) (map[string]bool, error) {
	rows, err := influx.QuerySQL("SELECT DISTINCT seed_id FROM expansion_gen")
	if err != nil {
		return nil, fmt.Errorf("query expansion_gen: %w", err)
	}

	ids := make(map[string]bool, len(rows))
	for _, row := range rows {
		id := strVal(row, "seed_id")
		if id != "" {
			ids[id] = true
		}
	}

	return ids, nil
}

// expandPrompts generates responses for expansion prompts using the given
// client and reports progress to InfluxDB. Already-completed prompts (per
// InfluxDB) are skipped. API errors for individual prompts are logged and
// skipped. InfluxDB reporting is best-effort. If duckDB is non-nil, prompt
// status is updated in DuckDB after each successful generation.
func expandPrompts(client *Client, influx *InfluxClient, duckDB *DB, prompts []Response,
	modelName, worker, outputDir string, dryRun bool, limits ...int) error {

	// When reading from DuckDB, prompts are already filtered to 'pending'.
	// When reading from JSONL, check InfluxDB for already-completed IDs.
	remaining := prompts
	if duckDB == nil {
		completed, err := getCompletedIDs(influx)
		if err != nil {
			return fmt.Errorf("get completed IDs: %w", err)
		}

		remaining = nil
		for _, p := range prompts {
			if !completed[p.ID] {
				remaining = append(remaining, p)
			}
		}

		skipped := len(prompts) - len(remaining)
		if skipped > 0 {
			log.Printf("skipping %d already-completed prompts, %d remaining", skipped, len(remaining))
		}
	}

	// Apply limit if provided (only for JSONL mode; DuckDB already limited in query).
	if duckDB == nil {
		limit := 0
		if len(limits) > 0 {
			limit = limits[0]
		}
		if limit > 0 && limit < len(remaining) {
			remaining = remaining[:limit]
		}
	}

	if len(remaining) == 0 {
		log.Println("all prompts already completed, nothing to do")
		return nil
	}

	// Dry-run: print plan and exit.
	if dryRun {
		log.Printf("dry-run: would process %d prompts with model %s (worker: %s)", len(remaining), modelName, worker)
		for i, p := range remaining {
			if i >= 10 {
				log.Printf("  ... and %d more", len(remaining)-10)
				break
			}
			log.Printf("  %s (domain: %s)", p.ID, p.Domain)
		}
		return nil
	}

	// Open output file in append mode.
	outputPath := filepath.Join(outputDir, fmt.Sprintf("expand-%s.jsonl", worker))
	f, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer f.Close()

	total := len(remaining)
	completedCount := 0

	for idx, p := range remaining {
		// Generate response.
		start := time.Now()
		response, err := client.ChatWithTemp(p.Prompt, 0.7)
		elapsed := time.Since(start).Seconds()

		if err != nil {
			log.Printf("[%d/%d] id=%s ERROR: %v", idx+1, total, p.ID, err)
			continue
		}

		chars := len(response)
		completedCount++

		// Write JSONL output.
		out := expandOutput{
			ID:             p.ID,
			Domain:         p.Domain,
			Prompt:         p.Prompt,
			Response:       response,
			Model:          modelName,
			ElapsedSeconds: elapsed,
			Chars:          chars,
		}

		line, err := json.Marshal(out)
		if err != nil {
			log.Printf("[%d/%d] id=%s marshal error: %v", idx+1, total, p.ID, err)
			continue
		}

		if _, err := f.Write(append(line, '\n')); err != nil {
			log.Printf("[%d/%d] id=%s write error: %v", idx+1, total, p.ID, err)
			continue
		}

		// Report to InfluxDB (best-effort).
		genLine := fmt.Sprintf("expansion_gen,i=%d,w=%s,d=%s seed_id=\"%s\",gen_time=%f,chars=%di,model=\"%s\"",
			idx, escapeLp(worker), escapeLp(p.Domain),
			p.ID, elapsed, chars, modelName)

		pct := float64(completedCount) / float64(total) * 100.0
		progressLine := fmt.Sprintf("expansion_progress,worker=%s completed=%di,target=%di,pct=%f",
			escapeLp(worker), completedCount, total, pct)

		if writeErr := influx.WriteLp([]string{genLine, progressLine}); writeErr != nil {
			log.Printf("[%d/%d] id=%s influx write error: %v", idx+1, total, p.ID, writeErr)
		}

		// Update DuckDB status if available (best-effort).
		if duckDB != nil {
			if dbErr := duckDB.UpdateExpansionStatus(int64(idx), "completed"); dbErr != nil {
				log.Printf("[%d/%d] id=%s db update error: %v", idx+1, total, p.ID, dbErr)
			}
		}

		// Log progress.
		log.Printf("[%d/%d] id=%s chars=%d time=%.1fs", idx+1, total, p.ID, chars, elapsed)
	}

	log.Printf("expand complete: %d/%d prompts generated, output: %s", completedCount, total, outputPath)

	return nil
}
