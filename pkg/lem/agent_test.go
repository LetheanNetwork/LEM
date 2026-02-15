package lem

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdapterMeta(t *testing.T) {
	tests := []struct {
		dirname              string
		wantModel, wantShort string
		wantStem             string
	}{
		{"adapters-deepseek-r1-7b-sovereignty", "deepseek-r1-7b", "R1-sov", "r1-sovereignty"},
		{"adapters-deepseek-r1-7b-russian", "deepseek-r1-7b", "R1-rus", "r1-russian"},
		{"adapters-deepseek-r1-7b-composure", "deepseek-r1-7b", "R1-comp", "r1-composure"},
		{"adapters-deepseek-r1-7b-sandwich", "deepseek-r1-7b", "R1-sand", "r1-sandwich"},
		{"adapters-deepseek-r1-7b-sandwich-watts", "deepseek-r1-7b", "R1-sw", "r1-sandwich-watts"},
		{"adapters-deepseek-r1-7b-western", "deepseek-r1-7b", "R1-west", "r1-western"},
		{"adapters-deepseek-r1-7b-western-fresh", "deepseek-r1-7b", "R1-wf", "r1-western-fresh"},
		{"adapters-deepseek-r1-7b", "deepseek-r1-7b", "R1-base", "r1-base"},
		{"adapters-deepseek-r1-7b-custom", "deepseek-r1-7b", "R1-cust", "r1-custom"},
	}

	for _, tt := range tests {
		model, short, stem := adapterMeta(tt.dirname)
		if model != tt.wantModel || short != tt.wantShort || stem != tt.wantStem {
			t.Errorf("adapterMeta(%q) = (%q, %q, %q), want (%q, %q, %q)",
				tt.dirname, model, short, stem, tt.wantModel, tt.wantShort, tt.wantStem)
		}
	}
}

func TestFindUnscored(t *testing.T) {
	checkpoints := []checkpoint{
		{RunID: "r1-sov-capability-auto", Label: "R1-sov @100", Dirname: "a", Iteration: 100},
		{RunID: "r1-sov-capability-auto", Label: "R1-sov @200", Dirname: "a", Iteration: 200},
		{RunID: "r1-sov-capability-auto", Label: "R1-sov @300", Dirname: "a", Iteration: 300},
	}

	scored := map[[2]string]bool{
		{"r1-sov-capability-auto", "R1-sov @100"}: true,
		{"r1-sov-capability-auto", "R1-sov @200"}: true,
	}

	unscored := findUnscored(checkpoints, scored)
	if len(unscored) != 1 {
		t.Fatalf("expected 1 unscored, got %d", len(unscored))
	}
	if unscored[0].Label != "R1-sov @300" {
		t.Errorf("expected R1-sov @300, got %s", unscored[0].Label)
	}
}

func TestFindUnscoredSorting(t *testing.T) {
	checkpoints := []checkpoint{
		{RunID: "r1-a", Label: "a @300", Dirname: "a", Iteration: 300},
		{RunID: "r1-b", Label: "b @100", Dirname: "b", Iteration: 100},
		{RunID: "r1-a", Label: "a @100", Dirname: "a", Iteration: 100},
	}

	scored := make(map[[2]string]bool)
	unscored := findUnscored(checkpoints, scored)

	if len(unscored) != 3 {
		t.Fatalf("expected 3 unscored, got %d", len(unscored))
	}
	// Should be sorted by dirname then iteration.
	if unscored[0].Label != "a @100" {
		t.Errorf("first should be a @100, got %s", unscored[0].Label)
	}
	if unscored[1].Label != "a @300" {
		t.Errorf("second should be a @300, got %s", unscored[1].Label)
	}
	if unscored[2].Label != "b @100" {
		t.Errorf("third should be b @100, got %s", unscored[2].Label)
	}
}

func TestRunCapabilityProbes(t *testing.T) {
	// Mock an OpenAI-compatible API that returns correct answers.
	answers := map[string]string{
		"What is 347":     "The answer is 10063.",
		"A store sells":   "You get $28.75 in change.",
		"Solve for x":     "x = -12",
		"If f(x)":         "f(4) = 21",
		"A bag has":       "The probability is 1/2 or 0.5",
		"A circle has":    "The area is 153.94 cm²",
		"next number":     "The next number is 162.",
		"laptop costs":    "The final price is $612.",
		"All cats":        "Yes, a cat needs water.",
		"If it rains":     "No, we cannot conclude that.",
		"room of 30":      "The minimum is 3 people sharing a birth month.",
		"farmer needs":    "Take the chicken first.",
		"class of 40":     "5 students play neither.",
		"Book is to":      "eating",
		"car won't start": "The starter motor is faulty.",
		"facing north":    "You are facing south.",
		"Event A":         "Event C happened in 1991.",
		"APPLE = 50":      "CAT = 24",
		"Python code":     "[2, 3]",
		"def f(n)":        "The output is 8.",
		"code has a bug":  "ZeroDivisionError when empty list.",
		"train travels":   "It takes 3 hours.",
		"twice as many":   "There are 7 children.",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		json.NewDecoder(r.Body).Decode(&req)

		prompt := ""
		for _, m := range req.Messages {
			if m.Role == "user" {
				prompt = m.Content
				break
			}
		}

		response := "I don't know."
		for prefix, ans := range answers {
			if strings.Contains(prompt, prefix) {
				response = ans
				break
			}
		}

		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Role: "assistant", Content: response}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	client.MaxTokens = 500

	results := runCapabilityProbes(client)

	if results.Total != 23 {
		t.Errorf("expected 23 total probes, got %d", results.Total)
	}
	if results.Correct != 23 {
		t.Errorf("expected 23 correct, got %d (accuracy: %.1f%%)", results.Correct, results.Accuracy)
	}
	if results.Accuracy != 100.0 {
		t.Errorf("expected 100%% accuracy, got %.1f%%", results.Accuracy)
	}
}

func TestPushCapabilityResults(t *testing.T) {
	var writtenLines []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v3/write_lp" {
			body := make([]byte, r.ContentLength)
			r.Body.Read(body)
			writtenLines = strings.Split(strings.TrimSpace(string(body)), "\n")
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()

	influx := &InfluxClient{url: server.URL, db: "test", token: "t"}

	cp := checkpoint{
		ModelTag:  "deepseek-r1-7b",
		RunID:     "r1-sov-capability-auto",
		Label:     "R1-sov @100",
		Iteration: 100,
	}

	results := probeResult{
		Accuracy: 87.0,
		Correct:  20,
		Total:    23,
		ByCategory: map[string]categoryResult{
			"arithmetic": {Correct: 2, Total: 2},
			"code":       {Correct: 2, Total: 3},
		},
		Probes: map[string]singleProbeResult{
			"math_01": {Passed: true, Response: "10063"},
			"math_02": {Passed: true, Response: "28.75"},
			"code_03": {Passed: false, Response: "I'm not sure."},
		},
	}

	err := pushCapabilityResults(influx, cp, results)
	if err != nil {
		t.Fatalf("push failed: %v", err)
	}

	// 1 overall + 2 categories + 3 probes = 6 lines.
	if len(writtenLines) != 6 {
		t.Errorf("expected 6 lines, got %d", len(writtenLines))
		for i, l := range writtenLines {
			t.Logf("  line %d: %s", i, l)
		}
	}

	// Check overall line.
	if !strings.HasPrefix(writtenLines[0], "capability_score,") {
		t.Errorf("first line should be capability_score, got: %s", writtenLines[0])
	}
	if !strings.Contains(writtenLines[0], "category=overall") {
		t.Errorf("first line should have category=overall, got: %s", writtenLines[0])
	}
	if !strings.Contains(writtenLines[0], "accuracy=87.0") {
		t.Errorf("first line should have accuracy=87.0, got: %s", writtenLines[0])
	}
}

func TestBufferAndReplay(t *testing.T) {
	tmpDir := t.TempDir()

	cp := checkpoint{
		ModelTag:  "test-model",
		RunID:     "test-run",
		Label:     "test @100",
		Iteration: 100,
	}
	results := probeResult{
		Accuracy: 50.0,
		Correct:  1,
		Total:    2,
		ByCategory: map[string]categoryResult{
			"arithmetic": {Correct: 1, Total: 2},
		},
		Probes: map[string]singleProbeResult{
			"math_01": {Passed: true, Response: "10063"},
			"math_02": {Passed: false, Response: "wrong"},
		},
	}

	// Buffer a result.
	bufferInfluxResult(tmpDir, cp, results)

	// Verify buffer file exists.
	bufPath := filepath.Join(tmpDir, "influx_buffer.jsonl")
	data, err := os.ReadFile(bufPath)
	if err != nil {
		t.Fatalf("buffer file not created: %v", err)
	}
	if !strings.Contains(string(data), "test-run") {
		t.Errorf("buffer should contain run_id, got: %s", string(data))
	}

	// Parse it.
	var entry bufferEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		t.Fatalf("parse buffer entry: %v", err)
	}
	if entry.Checkpoint.RunID != "test-run" {
		t.Errorf("expected run_id=test-run, got %s", entry.Checkpoint.RunID)
	}

	// Replay to a working InfluxDB.
	replayCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v3/write_lp" {
			replayCount++
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()

	influx := &InfluxClient{url: server.URL, db: "test", token: "t"}
	replayInfluxBuffer(tmpDir, influx)

	if replayCount == 0 {
		t.Error("expected replay to push to InfluxDB")
	}

	// Buffer should be cleared.
	if _, err := os.Stat(bufPath); !os.IsNotExist(err) {
		t.Error("buffer file should be removed after successful replay")
	}
}

func TestEnvOr(t *testing.T) {
	// Test with env var set.
	key := fmt.Sprintf("TEST_ENV_%d", os.Getpid())
	os.Setenv(key, "value")
	defer os.Unsetenv(key)

	if got := envOr(key, "fallback"); got != "value" {
		t.Errorf("envOr(%s) = %q, want %q", key, got, "value")
	}

	if got := envOr("NONEXISTENT_"+key, "fallback"); got != "fallback" {
		t.Errorf("envOr(nonexistent) = %q, want %q", got, "fallback")
	}
}

func TestFileBase(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"/foo/bar/baz.txt", "baz.txt"},
		{"baz.txt", "baz.txt"},
		{"/a/b/c", "c"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := fileBase(tt.input); got != tt.want {
			t.Errorf("fileBase(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
