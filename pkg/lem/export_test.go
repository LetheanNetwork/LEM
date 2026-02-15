package lem

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFilterResponses(t *testing.T) {
	tests := []struct {
		name  string
		input []Response
		want  int
	}{
		{
			name:  "empty input",
			input: []Response{},
			want:  0,
		},
		{
			name: "all valid",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: strings.Repeat("a", 50), Model: "m"},
				{ID: "2", Prompt: "world", Response: strings.Repeat("b", 100), Model: "m"},
			},
			want: 2,
		},
		{
			name: "filter empty response",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: "", Model: "m"},
				{ID: "2", Prompt: "world", Response: strings.Repeat("b", 50), Model: "m"},
			},
			want: 1,
		},
		{
			name: "filter error prefix",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: "ERROR: something went wrong", Model: "m"},
				{ID: "2", Prompt: "world", Response: strings.Repeat("b", 50), Model: "m"},
			},
			want: 1,
		},
		{
			name: "filter short response under 50 chars",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: strings.Repeat("a", 49), Model: "m"},
				{ID: "2", Prompt: "world", Response: strings.Repeat("b", 50), Model: "m"},
			},
			want: 1,
		},
		{
			name: "filter all bad",
			input: []Response{
				{ID: "1", Prompt: "p1", Response: "", Model: "m"},
				{ID: "2", Prompt: "p2", Response: "ERROR: fail", Model: "m"},
				{ID: "3", Prompt: "p3", Response: "too short", Model: "m"},
			},
			want: 0,
		},
		{
			name: "exactly 50 chars passes",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: strings.Repeat("x", 50), Model: "m"},
			},
			want: 1,
		},
		{
			name: "ERROR prefix is case sensitive",
			input: []Response{
				{ID: "1", Prompt: "hello", Response: strings.Repeat("error: lowercase is fine and long enough to pass", 2), Model: "m"},
			},
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterResponses(tt.input)
			if len(got) != tt.want {
				t.Errorf("filterResponses() returned %d responses, want %d", len(got), tt.want)
			}
		})
	}
}

func TestSplitData(t *testing.T) {
	// Create 100 responses for easy percentage calculation.
	responses := make([]Response, 100)
	for i := 0; i < 100; i++ {
		responses[i] = Response{ID: "r" + string(rune('0'+i/10)) + string(rune('0'+i%10))}
	}

	tests := []struct {
		name               string
		trainPct, validPct, testPct int
		wantTrain, wantValid, wantTest int
	}{
		{
			name:      "default 90/5/5",
			trainPct:  90,
			validPct:  5,
			testPct:   5,
			wantTrain: 90,
			wantValid: 5,
			wantTest:  5,
		},
		{
			name:      "80/10/10",
			trainPct:  80,
			validPct:  10,
			testPct:   10,
			wantTrain: 80,
			wantValid: 10,
			wantTest:  10,
		},
		{
			name:      "100/0/0",
			trainPct:  100,
			validPct:  0,
			testPct:   0,
			wantTrain: 100,
			wantValid: 0,
			wantTest:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			train, valid, test := splitData(responses, tt.trainPct, tt.validPct, tt.testPct, 42)
			if len(train) != tt.wantTrain {
				t.Errorf("train size = %d, want %d", len(train), tt.wantTrain)
			}
			if len(valid) != tt.wantValid {
				t.Errorf("valid size = %d, want %d", len(valid), tt.wantValid)
			}
			if len(test) != tt.wantTest {
				t.Errorf("test size = %d, want %d", len(test), tt.wantTest)
			}
		})
	}
}

func TestSplitDataDeterministic(t *testing.T) {
	responses := make([]Response, 20)
	for i := range responses {
		responses[i] = Response{ID: "r" + string(rune('A'+i))}
	}

	// Same seed should produce same split.
	train1, valid1, test1 := splitData(responses, 80, 10, 10, 42)
	train2, valid2, test2 := splitData(responses, 80, 10, 10, 42)

	for i := range train1 {
		if train1[i].ID != train2[i].ID {
			t.Errorf("train[%d]: got %s and %s with same seed", i, train1[i].ID, train2[i].ID)
		}
	}
	for i := range valid1 {
		if valid1[i].ID != valid2[i].ID {
			t.Errorf("valid[%d]: got %s and %s with same seed", i, valid1[i].ID, valid2[i].ID)
		}
	}
	for i := range test1 {
		if test1[i].ID != test2[i].ID {
			t.Errorf("test[%d]: got %s and %s with same seed", i, test1[i].ID, test2[i].ID)
		}
	}
}

func TestSplitDataDifferentSeed(t *testing.T) {
	responses := make([]Response, 50)
	for i := range responses {
		responses[i] = Response{ID: "r" + string(rune('A'+i%26)) + string(rune('0'+i/26))}
	}

	train1, _, _ := splitData(responses, 80, 10, 10, 42)
	train2, _, _ := splitData(responses, 80, 10, 10, 99)

	// Different seeds should (almost certainly) produce different orderings.
	different := false
	for i := range train1 {
		if train1[i].ID != train2[i].ID {
			different = true
			break
		}
	}
	if !different {
		t.Error("different seeds produced identical orderings, expected different")
	}
}

func TestSplitDataRemainder(t *testing.T) {
	// 7 items with 90/5/5: train=6, valid=0, test=0 — remainder goes to test.
	// Actually: train = 7*90/100 = 6, valid = 7*5/100 = 0, test = 7 - 6 - 0 = 1.
	responses := make([]Response, 7)
	for i := range responses {
		responses[i] = Response{ID: "r"}
	}

	train, valid, test := splitData(responses, 90, 5, 5, 42)
	total := len(train) + len(valid) + len(test)
	if total != 7 {
		t.Errorf("total split size = %d, want 7", total)
	}
}

func TestWriteTrainingJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train.jsonl")

	responses := []Response{
		{ID: "1", Prompt: "What is ethics?", Response: "Ethics is the study of moral principles.", Model: "m"},
		{ID: "2", Prompt: "Define AI.", Response: "Artificial Intelligence is a field of computer science.", Model: "m"},
	}

	if err := writeTrainingJSONL(path, responses); err != nil {
		t.Fatalf("writeTrainingJSONL() error: %v", err)
	}

	// Read back and verify.
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open output: %v", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var examples []TrainingExample
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var ex TrainingExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			t.Fatalf("failed to unmarshal line: %v", err)
		}
		examples = append(examples, ex)
	}

	if len(examples) != 2 {
		t.Fatalf("expected 2 examples, got %d", len(examples))
	}

	// Verify first example.
	if len(examples[0].Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(examples[0].Messages))
	}
	if examples[0].Messages[0].Role != "user" {
		t.Errorf("messages[0].role = %q, want %q", examples[0].Messages[0].Role, "user")
	}
	if examples[0].Messages[0].Content != "What is ethics?" {
		t.Errorf("messages[0].content = %q, want %q", examples[0].Messages[0].Content, "What is ethics?")
	}
	if examples[0].Messages[1].Role != "assistant" {
		t.Errorf("messages[1].role = %q, want %q", examples[0].Messages[1].Role, "assistant")
	}
	if examples[0].Messages[1].Content != "Ethics is the study of moral principles." {
		t.Errorf("messages[1].content = %q, want %q", examples[0].Messages[1].Content, "Ethics is the study of moral principles.")
	}
}

func TestWriteTrainingJSONLEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.jsonl")

	if err := writeTrainingJSONL(path, []Response{}); err != nil {
		t.Fatalf("writeTrainingJSONL() error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}

	if len(strings.TrimSpace(string(data))) != 0 {
		t.Errorf("expected empty file, got %q", string(data))
	}
}

func TestWriteTrainingJSONLCreatesFile(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "sub")
	if err := os.MkdirAll(subdir, 0755); err != nil {
		t.Fatalf("failed to create subdir: %v", err)
	}
	path := filepath.Join(subdir, "train.jsonl")

	responses := []Response{
		{ID: "1", Prompt: "hi", Response: "hello", Model: "m"},
	}

	if err := writeTrainingJSONL(path, responses); err != nil {
		t.Fatalf("writeTrainingJSONL() error: %v", err)
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("expected file to be created")
	}
}

func TestExportEndToEnd(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "golden.jsonl")
	outputDir := filepath.Join(dir, "output")

	// Create input with a mix of valid and invalid responses.
	validResponse := strings.Repeat("This is a valid response with sufficient length. ", 3)
	lines := []string{
		mustJSON(t, Response{ID: "1", Prompt: "p1", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "2", Prompt: "p2", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "3", Prompt: "p3", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "4", Prompt: "p4", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "5", Prompt: "p5", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "6", Prompt: "p6", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "7", Prompt: "p7", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "8", Prompt: "p8", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "9", Prompt: "p9", Response: validResponse, Model: "m1", Domain: "d1"}),
		mustJSON(t, Response{ID: "10", Prompt: "p10", Response: validResponse, Model: "m1", Domain: "d1"}),
		// These should be filtered out.
		mustJSON(t, Response{ID: "11", Prompt: "p11", Response: "", Model: "m1"}),
		mustJSON(t, Response{ID: "12", Prompt: "p12", Response: "ERROR: timeout", Model: "m1"}),
		mustJSON(t, Response{ID: "13", Prompt: "p13", Response: "short", Model: "m1"}),
	}

	if err := os.WriteFile(inputPath, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatalf("failed to write input: %v", err)
	}

	// Run export with 80/10/10 split.
	args := []string{
		"--input", inputPath,
		"--output-dir", outputDir,
		"--train-pct", "80",
		"--valid-pct", "10",
		"--test-pct", "10",
		"--seed", "42",
	}
	RunExport(args)

	// Verify output files exist.
	for _, name := range []string{"train.jsonl", "valid.jsonl", "test.jsonl"} {
		path := filepath.Join(outputDir, name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("expected %s to exist", path)
		}
	}

	// Count lines in each file.
	trainCount := countLines(t, filepath.Join(outputDir, "train.jsonl"))
	validCount := countLines(t, filepath.Join(outputDir, "valid.jsonl"))
	testCount := countLines(t, filepath.Join(outputDir, "test.jsonl"))

	total := trainCount + validCount + testCount
	if total != 10 {
		t.Errorf("total exported = %d, want 10 (3 should be filtered)", total)
	}

	// Train should be 80% of 10 = 8.
	if trainCount != 8 {
		t.Errorf("train count = %d, want 8", trainCount)
	}

	// Valid should be 10% of 10 = 1.
	if validCount != 1 {
		t.Errorf("valid count = %d, want 1", validCount)
	}

	// Test gets the remainder: 10 - 8 - 1 = 1.
	if testCount != 1 {
		t.Errorf("test count = %d, want 1", testCount)
	}

	// Verify output format: each line should be a valid TrainingExample.
	verifyTrainingFormat(t, filepath.Join(outputDir, "train.jsonl"))
	verifyTrainingFormat(t, filepath.Join(outputDir, "valid.jsonl"))
	verifyTrainingFormat(t, filepath.Join(outputDir, "test.jsonl"))
}

func TestExportPercentageValidation(t *testing.T) {
	tests := []struct {
		name                        string
		trainPct, validPct, testPct int
		wantErr                     bool
	}{
		{"valid 90/5/5", 90, 5, 5, false},
		{"valid 80/10/10", 80, 10, 10, false},
		{"valid 100/0/0", 100, 0, 0, false},
		{"invalid sum 90/10/10", 90, 10, 10, true},
		{"invalid sum 50/50/50", 50, 50, 50, true},
		{"invalid negative", -10, 60, 50, true},
		{"invalid sum too low", 80, 5, 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePercentages(tt.trainPct, tt.validPct, tt.testPct)
			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// Helper functions.

func mustJSON(t *testing.T, v interface{}) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}
	return string(data)
}

func countLines(t *testing.T, path string) int {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open %s: %v", path, err)
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			count++
		}
	}
	return count
}

func verifyTrainingFormat(t *testing.T, path string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open %s: %v", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var ex TrainingExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			t.Errorf("%s line %d: invalid JSON: %v", path, lineNum, err)
			continue
		}

		if len(ex.Messages) != 2 {
			t.Errorf("%s line %d: expected 2 messages, got %d", path, lineNum, len(ex.Messages))
			continue
		}

		if ex.Messages[0].Role != "user" {
			t.Errorf("%s line %d: messages[0].role = %q, want %q", path, lineNum, ex.Messages[0].Role, "user")
		}
		if ex.Messages[1].Role != "assistant" {
			t.Errorf("%s line %d: messages[1].role = %q, want %q", path, lineNum, ex.Messages[1].Role, "assistant")
		}
		if ex.Messages[0].Content == "" {
			t.Errorf("%s line %d: messages[0].content is empty", path, lineNum)
		}
		if ex.Messages[1].Content == "" {
			t.Errorf("%s line %d: messages[1].content is empty", path, lineNum)
		}
	}
}
