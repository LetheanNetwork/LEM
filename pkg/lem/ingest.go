package lem

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// RunIngest is the CLI entry point for the ingest command.
// It reads benchmark JSONL files and training logs, then pushes
// the data into InfluxDB as line protocol for the lab dashboard.
func RunIngest(args []string) {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)

	contentFile := fs.String("content", "", "Content scores JSONL file")
	capabilityFile := fs.String("capability", "", "Capability scores JSONL file")
	trainingLog := fs.String("training-log", "", "MLX LoRA training log file")
	model := fs.String("model", "", "Model name tag (required)")
	runID := fs.String("run-id", "", "Run ID tag (defaults to model name)")
	influxURL := fs.String("influx", "", "InfluxDB URL")
	influxDB := fs.String("influx-db", "", "InfluxDB database name")
	batchSize := fs.Int("batch-size", 100, "Lines per InfluxDB write batch")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	if *contentFile == "" && *capabilityFile == "" && *trainingLog == "" {
		fmt.Fprintln(os.Stderr, "error: at least one of --content, --capability, or --training-log is required")
		fs.Usage()
		os.Exit(1)
	}

	if *runID == "" {
		*runID = *model
	}

	influx := NewInfluxClient(*influxURL, *influxDB)
	total := 0

	if *contentFile != "" {
		n, err := ingestContentScores(influx, *contentFile, *model, *runID, *batchSize)
		if err != nil {
			log.Fatalf("ingest content scores: %v", err)
		}
		fmt.Printf("  Content scores: %d points\n", n)
		total += n
	}

	if *capabilityFile != "" {
		n, err := ingestCapabilityScores(influx, *capabilityFile, *model, *runID, *batchSize)
		if err != nil {
			log.Fatalf("ingest capability scores: %v", err)
		}
		fmt.Printf("  Capability scores: %d points\n", n)
		total += n
	}

	if *trainingLog != "" {
		n, err := ingestTrainingCurve(influx, *trainingLog, *model, *runID, *batchSize)
		if err != nil {
			log.Fatalf("ingest training curve: %v", err)
		}
		fmt.Printf("  Training curve: %d points\n", n)
		total += n
	}

	fmt.Printf("\nTotal: %d points ingested\n", total)
}

var iterRe = regexp.MustCompile(`@(\d+)`)

// extractIteration pulls the iteration number from a label like "model@200".
func extractIteration(label string) int {
	m := iterRe.FindStringSubmatch(label)
	if m == nil {
		return 0
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// contentScoreEntry is one line from a content scores JSONL file.
type contentScoreEntry struct {
	Label      string                       `json:"label"`
	Aggregates map[string]float64           `json:"aggregates"`
	Probes     map[string]contentProbeEntry `json:"probes"`
}

type contentProbeEntry struct {
	Scores map[string]interface{} `json:"scores"`
}

// ingestContentScores reads a content scores JSONL file and writes
// content_score and probe_score measurements to InfluxDB.
func ingestContentScores(influx *InfluxClient, filepath, model, runID string, batchSize int) (int, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", filepath, err)
	}
	defer f.Close()

	var lines []string
	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		var entry contentScoreEntry
		if err := json.Unmarshal([]byte(text), &entry); err != nil {
			continue
		}

		label := entry.Label
		if label == "" {
			label = "unknown"
		}
		iteration := extractIteration(label)
		hasKernel := strings.Contains(strings.ToLower(label), "kernel")

		// Aggregate scores.
		for dim, val := range entry.Aggregates {
			lp := fmt.Sprintf("content_score,model=%s,run_id=%s,label=%s,dimension=%s,has_kernel=%t score=%f,iteration=%di",
				escapeLp(model), escapeLp(runID), escapeLp(label), escapeLp(dim), hasKernel, val, iteration)
			lines = append(lines, lp)
			count++
		}

		// Per-probe scores.
		for probeID, probeData := range entry.Probes {
			for dim, val := range probeData.Scores {
				if dim == "notes" {
					continue
				}
				fval, ok := toFloat64(val)
				if !ok {
					continue
				}
				lp := fmt.Sprintf("probe_score,model=%s,run_id=%s,label=%s,probe=%s,dimension=%s,has_kernel=%t score=%f,iteration=%di",
					escapeLp(model), escapeLp(runID), escapeLp(label), escapeLp(probeID), escapeLp(dim), hasKernel, fval, iteration)
				lines = append(lines, lp)
				count++
			}
		}

		if len(lines) >= batchSize {
			if err := influx.WriteLp(lines); err != nil {
				return count, fmt.Errorf("write content scores: %w", err)
			}
			lines = lines[:0]
		}
	}

	if len(lines) > 0 {
		if err := influx.WriteLp(lines); err != nil {
			return count, fmt.Errorf("flush content scores: %w", err)
		}
	}

	return count, scanner.Err()
}

// capabilityScoreEntry is one line from a capability scores JSONL file.
type capabilityScoreEntry struct {
	Label      string                        `json:"label"`
	Accuracy   float64                       `json:"accuracy"`
	Correct    int                           `json:"correct"`
	Total      int                           `json:"total"`
	ByCategory map[string]capabilityCatEntry `json:"by_category"`
}

type capabilityCatEntry struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

// ingestCapabilityScores reads a capability scores JSONL file and writes
// capability_score measurements to InfluxDB.
func ingestCapabilityScores(influx *InfluxClient, filepath, model, runID string, batchSize int) (int, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", filepath, err)
	}
	defer f.Close()

	var lines []string
	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		var entry capabilityScoreEntry
		if err := json.Unmarshal([]byte(text), &entry); err != nil {
			continue
		}

		label := entry.Label
		if label == "" {
			label = "unknown"
		}
		iteration := extractIteration(label)

		// Overall score.
		lp := fmt.Sprintf("capability_score,model=%s,run_id=%s,label=%s,category=overall accuracy=%f,correct=%di,total=%di,iteration=%di",
			escapeLp(model), escapeLp(runID), escapeLp(label), entry.Accuracy, entry.Correct, entry.Total, iteration)
		lines = append(lines, lp)
		count++

		// Per-category scores.
		for cat, catData := range entry.ByCategory {
			if catData.Total > 0 {
				pct := float64(catData.Correct) / float64(catData.Total) * 100.0
				lp := fmt.Sprintf("capability_score,model=%s,run_id=%s,label=%s,category=%s accuracy=%f,correct=%di,total=%di,iteration=%di",
					escapeLp(model), escapeLp(runID), escapeLp(label), escapeLp(cat), pct, catData.Correct, catData.Total, iteration)
				lines = append(lines, lp)
				count++
			}
		}

		if len(lines) >= batchSize {
			if err := influx.WriteLp(lines); err != nil {
				return count, fmt.Errorf("write capability scores: %w", err)
			}
			lines = lines[:0]
		}
	}

	if len(lines) > 0 {
		if err := influx.WriteLp(lines); err != nil {
			return count, fmt.Errorf("flush capability scores: %w", err)
		}
	}

	return count, scanner.Err()
}

var (
	valLossRe   = regexp.MustCompile(`Iter (\d+): Val loss ([\d.]+)`)
	trainLossRe = regexp.MustCompile(`Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.eE+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+)`)
)

// ingestTrainingCurve parses an mlx_lm training log and writes
// training_loss measurements to InfluxDB.
func ingestTrainingCurve(influx *InfluxClient, filepath, model, runID string, batchSize int) (int, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", filepath, err)
	}
	defer f.Close()

	var lines []string
	count := 0
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		text := scanner.Text()

		if m := valLossRe.FindStringSubmatch(text); m != nil {
			iteration, _ := strconv.Atoi(m[1])
			valLoss, _ := strconv.ParseFloat(m[2], 64)
			lp := fmt.Sprintf("training_loss,model=%s,run_id=%s,loss_type=val loss=%f,iteration=%di",
				escapeLp(model), escapeLp(runID), valLoss, iteration)
			lines = append(lines, lp)
			count++
		}

		if m := trainLossRe.FindStringSubmatch(text); m != nil {
			iteration, _ := strconv.Atoi(m[1])
			trainLoss, _ := strconv.ParseFloat(m[2], 64)
			lr, _ := strconv.ParseFloat(m[3], 64)
			itSec, _ := strconv.ParseFloat(m[4], 64)
			tokSec, _ := strconv.ParseFloat(m[5], 64)
			lp := fmt.Sprintf("training_loss,model=%s,run_id=%s,loss_type=train loss=%f,learning_rate=%f,iterations_per_sec=%f,tokens_per_sec=%f,iteration=%di",
				escapeLp(model), escapeLp(runID), trainLoss, lr, itSec, tokSec, iteration)
			lines = append(lines, lp)
			count++
		}

		if len(lines) >= batchSize {
			if err := influx.WriteLp(lines); err != nil {
				return count, fmt.Errorf("write training curve: %w", err)
			}
			lines = lines[:0]
		}
	}

	if len(lines) > 0 {
		if err := influx.WriteLp(lines); err != nil {
			return count, fmt.Errorf("flush training curve: %w", err)
		}
	}

	return count, scanner.Err()
}

// toFloat64 converts an interface{} to float64 if possible.
func toFloat64(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
