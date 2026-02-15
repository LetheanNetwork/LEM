package lem

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
)

// runStatus parses CLI flags and prints training/generation status from InfluxDB.
func RunStatus(args []string) {
	fs := flag.NewFlagSet("status", flag.ExitOnError)

	influxURL := fs.String("influx", "", "InfluxDB URL (default http://10.69.69.165:8181)")
	influxDB := fs.String("influx-db", "", "InfluxDB database name (default training)")
	dbPath := fs.String("db", "", "DuckDB database path (shows table counts)")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	// Check LEM_DB env as default for --db.
	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}

	influx := NewInfluxClient(*influxURL, *influxDB)

	if err := printStatus(influx, os.Stdout); err != nil {
		log.Fatalf("status: %v", err)
	}

	// If DuckDB path provided, show table counts.
	if *dbPath != "" {
		db, err := OpenDB(*dbPath)
		if err != nil {
			log.Fatalf("open db: %v", err)
		}
		defer db.Close()

		counts, err := db.TableCounts()
		if err != nil {
			log.Fatalf("table counts: %v", err)
		}

		fmt.Fprintln(os.Stdout)
		fmt.Fprintln(os.Stdout, "DuckDB:")
		order := []string{"golden_set", "expansion_prompts", "seeds", "training_examples",
			"prompts", "gemini_responses", "benchmark_questions", "benchmark_results", "validations"}
		for _, table := range order {
			if count, ok := counts[table]; ok {
				fmt.Fprintf(os.Stdout, "  %-22s %6d rows\n", table, count)
			}
		}
	}
}

// trainingRow holds deduplicated training status + loss for a single model.
type trainingRow struct {
	model      string
	status     string
	iteration  int
	totalIters int
	pct        float64
	loss       float64
	hasLoss    bool
}

// genRow holds deduplicated generation progress for a single worker.
type genRow struct {
	worker    string
	completed int
	target    int
	pct       float64
}

// printStatus queries InfluxDB for training and generation progress and writes
// a formatted summary to w. The function is separated from runStatus so tests
// can capture output via an io.Writer.
func printStatus(influx *InfluxClient, w io.Writer) error {
	// Query training status (may not exist yet).
	statusRows, err := influx.QuerySQL(
		"SELECT model, run_id, status, iteration, total_iters, pct FROM training_status ORDER BY time DESC LIMIT 10",
	)
	if err != nil {
		statusRows = nil
	}

	// Query training loss (may not exist yet).
	lossRows, err := influx.QuerySQL(
		"SELECT model, loss_type, loss, iteration, tokens_per_sec FROM training_loss WHERE loss_type = 'train' ORDER BY time DESC LIMIT 10",
	)
	if err != nil {
		lossRows = nil
	}

	// Query golden generation progress (may not exist yet).
	goldenRows, err := influx.QuerySQL(
		"SELECT worker, completed, target, pct FROM golden_gen_progress ORDER BY time DESC LIMIT 5",
	)
	if err != nil {
		goldenRows = nil // table may not exist yet
	}

	// Query expansion progress (may not exist yet).
	expansionRows, err := influx.QuerySQL(
		"SELECT worker, completed, target, pct FROM expansion_progress ORDER BY time DESC LIMIT 5",
	)
	if err != nil {
		expansionRows = nil // table may not exist yet
	}

	// Deduplicate training status by model (keep first = latest).
	training := dedupeTraining(statusRows, lossRows)

	// Deduplicate generation progress by worker.
	golden := dedupeGeneration(goldenRows)
	expansion := dedupeGeneration(expansionRows)

	// Print training section.
	fmt.Fprintln(w, "Training:")
	if len(training) == 0 {
		fmt.Fprintln(w, "  (no data)")
	} else {
		for _, tr := range training {
			progress := fmt.Sprintf("%d/%d", tr.iteration, tr.totalIters)
			pct := fmt.Sprintf("%.1f%%", tr.pct)
			if tr.hasLoss {
				fmt.Fprintf(w, "  %-13s %-9s %9s %7s  loss=%.3f\n",
					tr.model, tr.status, progress, pct, tr.loss)
			} else {
				fmt.Fprintf(w, "  %-13s %-9s %9s %7s\n",
					tr.model, tr.status, progress, pct)
			}
		}
	}

	// Print generation section.
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Generation:")

	hasGenData := false

	if len(golden) > 0 {
		hasGenData = true
		for _, g := range golden {
			progress := fmt.Sprintf("%d/%d", g.completed, g.target)
			pct := fmt.Sprintf("%.1f%%", g.pct)
			fmt.Fprintf(w, "  %-13s %11s %7s  (%s)\n", "golden", progress, pct, g.worker)
		}
	}

	if len(expansion) > 0 {
		hasGenData = true
		for _, g := range expansion {
			progress := fmt.Sprintf("%d/%d", g.completed, g.target)
			pct := fmt.Sprintf("%.1f%%", g.pct)
			fmt.Fprintf(w, "  %-13s %11s %7s  (%s)\n", "expansion", progress, pct, g.worker)
		}
	}

	if !hasGenData {
		fmt.Fprintln(w, "  (no data)")
	}

	return nil
}

// dedupeTraining merges training status and loss rows, keeping only the first
// (latest) row per model. Returns sorted by model name.
func dedupeTraining(statusRows, lossRows []map[string]interface{}) []trainingRow {
	// Build loss lookup: model -> loss value.
	lossMap := make(map[string]float64)
	lossSeenMap := make(map[string]bool)
	for _, row := range lossRows {
		model := strVal(row, "model")
		if model == "" {
			continue
		}
		if lossSeenMap[model] {
			continue // keep first (latest)
		}
		lossSeenMap[model] = true
		lossMap[model] = floatVal(row, "loss")
	}

	// Build training rows, deduplicating by model.
	seen := make(map[string]bool)
	var rows []trainingRow
	for _, row := range statusRows {
		model := strVal(row, "model")
		if model == "" {
			continue
		}
		if seen[model] {
			continue // keep first (latest)
		}
		seen[model] = true

		tr := trainingRow{
			model:      model,
			status:     strVal(row, "status"),
			iteration:  intVal(row, "iteration"),
			totalIters: intVal(row, "total_iters"),
			pct:        floatVal(row, "pct"),
		}

		if loss, ok := lossMap[model]; ok {
			tr.loss = loss
			tr.hasLoss = true
		}

		rows = append(rows, tr)
	}

	// Sort by model name for deterministic output.
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].model < rows[j].model
	})

	return rows
}

// dedupeGeneration deduplicates generation progress rows by worker, keeping
// only the first (latest) row per worker. Returns sorted by worker name.
func dedupeGeneration(rows []map[string]interface{}) []genRow {
	seen := make(map[string]bool)
	var result []genRow
	for _, row := range rows {
		worker := strVal(row, "worker")
		if worker == "" {
			continue
		}
		if seen[worker] {
			continue // keep first (latest)
		}
		seen[worker] = true

		result = append(result, genRow{
			worker:    worker,
			completed: intVal(row, "completed"),
			target:    intVal(row, "target"),
			pct:       floatVal(row, "pct"),
		})
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].worker < result[j].worker
	})

	return result
}

// strVal extracts a string value from a row map, returning "" if missing or
// not a string.
func strVal(row map[string]interface{}, key string) string {
	v, ok := row[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// floatVal extracts a float64 value from a row map, returning 0 if missing or
// not a float64.
func floatVal(row map[string]interface{}, key string) float64 {
	v, ok := row[key]
	if !ok {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return f
}

// intVal extracts an integer value from a row map. InfluxDB JSON returns all
// numbers as float64, so this truncates to int.
func intVal(row map[string]interface{}, key string) int {
	return int(floatVal(row, key))
}
