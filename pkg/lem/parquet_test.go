package lem

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/parquet-go/parquet-go"
)

func TestExportSplitParquet(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "train.jsonl")
	outputDir := filepath.Join(dir, "output")
	os.MkdirAll(outputDir, 0755)

	// Write test JSONL.
	convs := []TrainingExample{
		{Messages: []ChatMessage{
			{Role: "user", Content: "What is wisdom?"},
			{Role: "assistant", Content: "The application of understanding."},
		}},
		{Messages: []ChatMessage{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Tell me about ethics."},
			{Role: "assistant", Content: "Ethics concerns right action."},
		}},
	}

	f, _ := os.Create(inputPath)
	for _, c := range convs {
		data, _ := json.Marshal(c)
		f.Write(data)
		f.WriteString("\n")
	}
	f.Close()

	n, err := exportSplitParquet(inputPath, outputDir, "train")
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if n != 2 {
		t.Errorf("expected 2 rows, got %d", n)
	}

	// Verify Parquet file exists and is readable.
	outPath := filepath.Join(outputDir, "train.parquet")
	pf, err := os.Open(outPath)
	if err != nil {
		t.Fatalf("open parquet: %v", err)
	}
	defer pf.Close()

	info, _ := pf.Stat()
	reader := parquet.NewGenericReader[ParquetRow](pf)
	defer reader.Close()

	rows := make([]ParquetRow, 10)
	read, err := reader.Read(rows)
	if err != nil && err != io.EOF {
		t.Fatalf("read parquet: %v", err)
	}
	if read != 2 {
		t.Errorf("expected 2 rows in parquet, got %d", read)
	}

	if rows[0].Prompt != "What is wisdom?" {
		t.Errorf("unexpected prompt: %s", rows[0].Prompt)
	}
	if rows[0].Response != "The application of understanding." {
		t.Errorf("unexpected response: %s", rows[0].Response)
	}
	if rows[1].System != "You are helpful." {
		t.Errorf("expected system message, got: %s", rows[1].System)
	}

	if info.Size() == 0 {
		t.Error("parquet file is empty")
	}
}

func TestExportSplitParquetEmpty(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "empty.jsonl")
	outputDir := filepath.Join(dir, "output")
	os.MkdirAll(outputDir, 0755)

	// Write empty JSONL.
	os.WriteFile(inputPath, []byte("\n\n"), 0644)

	n, err := exportSplitParquet(inputPath, outputDir, "test")
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 rows for empty file, got %d", n)
	}
}

func TestExportSplitParquetMessages(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "valid.jsonl")
	outputDir := filepath.Join(dir, "output")
	os.MkdirAll(outputDir, 0755)

	conv := TrainingExample{Messages: []ChatMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}

	f, _ := os.Create(inputPath)
	data, _ := json.Marshal(conv)
	f.Write(data)
	f.WriteString("\n")
	f.Close()

	n, err := exportSplitParquet(inputPath, outputDir, "valid")
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if n != 1 {
		t.Errorf("expected 1 row, got %d", n)
	}

	// Verify messages field contains valid JSON.
	pf, _ := os.Open(filepath.Join(outputDir, "valid.parquet"))
	defer pf.Close()
	reader := parquet.NewGenericReader[ParquetRow](pf)
	defer reader.Close()

	rows := make([]ParquetRow, 1)
	reader.Read(rows)

	var msgs []ChatMessage
	if err := json.Unmarshal([]byte(rows[0].Messages), &msgs); err != nil {
		t.Fatalf("parse messages JSON: %v", err)
	}
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages in JSON, got %d", len(msgs))
	}
}
