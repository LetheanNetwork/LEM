package lem

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/parquet-go/parquet-go"
)

// ParquetRow is the schema for exported Parquet files.
type ParquetRow struct {
	Prompt   string `parquet:"prompt"`
	Response string `parquet:"response"`
	System   string `parquet:"system"`
	Messages string `parquet:"messages"`
}

// RunParquet is the CLI entry point for the parquet command.
// Reads JSONL training splits (train.jsonl, valid.jsonl, test.jsonl) and
// writes Parquet files with snappy compression for HuggingFace datasets.
func RunParquet(args []string) {
	fs := flag.NewFlagSet("parquet", flag.ExitOnError)

	trainingDir := fs.String("input", "", "Directory containing train.jsonl, valid.jsonl, test.jsonl (required)")
	outputDir := fs.String("output", "", "Output directory for Parquet files (defaults to input/parquet)")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *trainingDir == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required (directory with JSONL splits)")
		fs.Usage()
		os.Exit(1)
	}

	if *outputDir == "" {
		*outputDir = filepath.Join(*trainingDir, "parquet")
	}

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	fmt.Printf("Exporting Parquet from %s → %s\n", *trainingDir, *outputDir)

	total := 0
	for _, split := range []string{"train", "valid", "test"} {
		jsonlPath := filepath.Join(*trainingDir, split+".jsonl")
		if _, err := os.Stat(jsonlPath); os.IsNotExist(err) {
			fmt.Printf("  Skip: %s.jsonl not found\n", split)
			continue
		}

		n, err := exportSplitParquet(jsonlPath, *outputDir, split)
		if err != nil {
			log.Fatalf("export %s: %v", split, err)
		}
		total += n
	}

	fmt.Printf("\nTotal: %d rows exported\n", total)
}

// exportSplitParquet reads a JSONL file and writes a Parquet file for the split.
func exportSplitParquet(jsonlPath, outputDir, split string) (int, error) {
	f, err := os.Open(jsonlPath)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", jsonlPath, err)
	}
	defer f.Close()

	var rows []ParquetRow
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		var data struct {
			Messages []ChatMessage `json:"messages"`
		}
		if err := json.Unmarshal([]byte(text), &data); err != nil {
			continue
		}

		var prompt, response, system string
		for _, m := range data.Messages {
			switch m.Role {
			case "user":
				if prompt == "" {
					prompt = m.Content
				}
			case "assistant":
				if response == "" {
					response = m.Content
				}
			case "system":
				if system == "" {
					system = m.Content
				}
			}
		}

		msgsJSON, _ := json.Marshal(data.Messages)
		rows = append(rows, ParquetRow{
			Prompt:   prompt,
			Response: response,
			System:   system,
			Messages: string(msgsJSON),
		})
	}

	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("scan %s: %w", jsonlPath, err)
	}

	if len(rows) == 0 {
		fmt.Printf("  Skip: %s — no data\n", split)
		return 0, nil
	}

	outPath := filepath.Join(outputDir, split+".parquet")

	out, err := os.Create(outPath)
	if err != nil {
		return 0, fmt.Errorf("create %s: %w", outPath, err)
	}

	writer := parquet.NewGenericWriter[ParquetRow](out,
		parquet.Compression(&parquet.Snappy),
	)

	if _, err := writer.Write(rows); err != nil {
		out.Close()
		return 0, fmt.Errorf("write parquet rows: %w", err)
	}

	if err := writer.Close(); err != nil {
		out.Close()
		return 0, fmt.Errorf("close parquet writer: %w", err)
	}

	if err := out.Close(); err != nil {
		return 0, fmt.Errorf("close file: %w", err)
	}

	info, _ := os.Stat(outPath)
	sizeMB := float64(info.Size()) / 1024 / 1024
	fmt.Printf("  %s.parquet: %d rows (%.1f MB)\n", split, len(rows), sizeMB)

	return len(rows), nil
}
