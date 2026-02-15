package lem

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractIteration(t *testing.T) {
	tests := []struct {
		label string
		want  int
	}{
		{"deepseek-r1@200", 200},
		{"gemma12b@1600", 1600},
		{"model@0", 0},
		{"no-iteration", 0},
		{"base", 0},
		{"@50+kernel", 50},
	}

	for _, tt := range tests {
		got := extractIteration(tt.label)
		if got != tt.want {
			t.Errorf("extractIteration(%q) = %d, want %d", tt.label, got, tt.want)
		}
	}
}

func TestIngestContentScores(t *testing.T) {
	var receivedLines []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		receivedLines = append(receivedLines, strings.Split(string(body), "\n")...)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	// Create test JSONL.
	dir := t.TempDir()
	path := filepath.Join(dir, "content.jsonl")

	entries := []contentScoreEntry{
		{
			Label:      "gemma12b@200",
			Aggregates: map[string]float64{"sovereignty": 7.5, "ethical_depth": 8.0},
			Probes: map[string]contentProbeEntry{
				"p01": {Scores: map[string]interface{}{"sovereignty": 8.0, "notes": "good"}},
			},
		},
		{
			Label:      "gemma12b@400+kernel",
			Aggregates: map[string]float64{"sovereignty": 9.0},
		},
	}

	f, _ := os.Create(path)
	for _, e := range entries {
		data, _ := json.Marshal(e)
		f.Write(data)
		f.WriteString("\n")
	}
	f.Close()

	influx := &InfluxClient{url: ts.URL, db: "test", token: "test"}
	n, err := ingestContentScores(influx, path, "gemma3-12b", "test-run", 100)
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	// 2 aggregates + 1 probe (notes skipped) + 1 aggregate = 4 points.
	if n != 4 {
		t.Errorf("expected 4 points, got %d", n)
	}

	// Verify line protocol content.
	allLines := strings.Join(receivedLines, "\n")
	if !strings.Contains(allLines, "content_score") {
		t.Error("missing content_score measurement")
	}
	if !strings.Contains(allLines, "probe_score") {
		t.Error("missing probe_score measurement")
	}
	if !strings.Contains(allLines, "has_kernel=true") {
		t.Error("missing has_kernel=true for kernel label")
	}
	if !strings.Contains(allLines, "iteration=200i") {
		t.Error("missing iteration=200i")
	}
}

func TestIngestCapabilityScores(t *testing.T) {
	var receivedLines []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		receivedLines = append(receivedLines, strings.Split(string(body), "\n")...)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	dir := t.TempDir()
	path := filepath.Join(dir, "capability.jsonl")

	entries := []capabilityScoreEntry{
		{
			Label:    "deepseek@400",
			Accuracy: 82.6,
			Correct:  19,
			Total:    23,
			ByCategory: map[string]capabilityCatEntry{
				"math":  {Correct: 7, Total: 8},
				"logic": {Correct: 4, Total: 5},
				"empty": {Correct: 0, Total: 0}, // Should be skipped.
			},
		},
	}

	f, _ := os.Create(path)
	for _, e := range entries {
		data, _ := json.Marshal(e)
		f.Write(data)
		f.WriteString("\n")
	}
	f.Close()

	influx := &InfluxClient{url: ts.URL, db: "test", token: "test"}
	n, err := ingestCapabilityScores(influx, path, "deepseek-r1-7b", "test-run", 100)
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	// 1 overall + 2 categories (empty skipped) = 3.
	if n != 3 {
		t.Errorf("expected 3 points, got %d", n)
	}

	allLines := strings.Join(receivedLines, "\n")
	if !strings.Contains(allLines, "category=overall") {
		t.Error("missing overall category")
	}
	if !strings.Contains(allLines, "category=math") {
		t.Error("missing math category")
	}
	if !strings.Contains(allLines, "iteration=400i") {
		t.Error("missing iteration=400i")
	}
}

func TestIngestTrainingCurve(t *testing.T) {
	var receivedLines []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		receivedLines = append(receivedLines, strings.Split(string(body), "\n")...)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	dir := t.TempDir()
	path := filepath.Join(dir, "training.log")

	logContent := `Loading model from mlx-community/gemma-3-1b-it-qat-4bit
Starting training...
Iter 10: Train loss 2.534, Learning Rate 1.000e-05, It/sec 3.21, Tokens/sec 1205.4
Iter 20: Train loss 1.891, Learning Rate 1.000e-05, It/sec 3.18, Tokens/sec 1198.2
Iter 25: Val loss 1.756
Iter 30: Train loss 1.654, Learning Rate 1.000e-05, It/sec 3.22, Tokens/sec 1210.0
Some random log line that should be ignored
Iter 50: Val loss 1.523
`
	os.WriteFile(path, []byte(logContent), 0644)

	influx := &InfluxClient{url: ts.URL, db: "test", token: "test"}
	n, err := ingestTrainingCurve(influx, path, "gemma3-1b", "test-run", 100)
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	// 3 train + 2 val = 5.
	if n != 5 {
		t.Errorf("expected 5 points, got %d", n)
	}

	allLines := strings.Join(receivedLines, "\n")
	if !strings.Contains(allLines, "loss_type=val") {
		t.Error("missing val loss")
	}
	if !strings.Contains(allLines, "loss_type=train") {
		t.Error("missing train loss")
	}
	if !strings.Contains(allLines, "tokens_per_sec=") {
		t.Error("missing tokens_per_sec field")
	}
}

func TestToFloat64(t *testing.T) {
	tests := []struct {
		input interface{}
		want  float64
		ok    bool
	}{
		{7.5, 7.5, true},
		{42, 42.0, true},
		{"not a number", 0, false},
		{nil, 0, false},
	}

	for _, tt := range tests {
		got, ok := toFloat64(tt.input)
		if ok != tt.ok {
			t.Errorf("toFloat64(%v) ok=%v, want %v", tt.input, ok, tt.ok)
		}
		if ok && got != tt.want {
			t.Errorf("toFloat64(%v) = %f, want %f", tt.input, got, tt.want)
		}
	}
}
