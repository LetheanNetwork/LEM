package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// mockExpandServer creates a combined httptest server that handles both
// OpenAI-compatible chat API and InfluxDB API endpoints.
// apiResponse is returned for /v1/chat/completions requests.
// influxQueryHandler handles /api/v3/query_sql requests.
// influxWriteHandler handles /api/v3/write_lp requests.
func mockExpandServer(t *testing.T, apiResponse string, influxQueryHandler func(q string) ([]map[string]interface{}, int), influxWriteHandler func(body string)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/chat/completions":
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: apiResponse}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)

		case r.URL.Path == "/api/v3/query_sql":
			var body struct {
				DB string `json:"db"`
				Q  string `json:"q"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			rows, status := influxQueryHandler(body.Q)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			json.NewEncoder(w).Encode(rows)

		case r.URL.Path == "/api/v3/write_lp":
			body, _ := io.ReadAll(r.Body)
			if influxWriteHandler != nil {
				influxWriteHandler(string(body))
			}
			w.WriteHeader(http.StatusOK)

		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func TestGetCompletedIDs(t *testing.T) {
	t.Run("returns completed IDs from InfluxDB", func(t *testing.T) {
		server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
			if !strings.Contains(q, "expansion_gen") {
				t.Errorf("expected query against expansion_gen, got: %s", q)
			}
			return []map[string]interface{}{
				{"seed_id": "prompt_001"},
				{"seed_id": "prompt_002"},
				{"seed_id": "prompt_003"},
			}, http.StatusOK
		})
		defer server.Close()

		t.Setenv("INFLUX_TOKEN", "test-token")
		influx := NewInfluxClient(server.URL, "training")

		ids, err := getCompletedIDs(influx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(ids) != 3 {
			t.Fatalf("expected 3 completed IDs, got %d", len(ids))
		}
		for _, id := range []string{"prompt_001", "prompt_002", "prompt_003"} {
			if !ids[id] {
				t.Errorf("expected ID %q to be in completed set", id)
			}
		}
	})

	t.Run("returns empty set when no completed IDs", func(t *testing.T) {
		server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
			return []map[string]interface{}{}, http.StatusOK
		})
		defer server.Close()

		t.Setenv("INFLUX_TOKEN", "test-token")
		influx := NewInfluxClient(server.URL, "training")

		ids, err := getCompletedIDs(influx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(ids) != 0 {
			t.Errorf("expected 0 completed IDs, got %d", len(ids))
		}
	})

	t.Run("returns error on InfluxDB failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("database error"))
		}))
		defer server.Close()

		t.Setenv("INFLUX_TOKEN", "test-token")
		influx := NewInfluxClient(server.URL, "training")

		_, err := getCompletedIDs(influx)
		if err == nil {
			t.Fatal("expected error on InfluxDB failure, got nil")
		}
	})
}

func TestExpandPromptsBasic(t *testing.T) {
	var apiCalls atomic.Int32
	var writtenLines []string

	server := mockExpandServer(t,
		"This is a generated response about ethics and sovereignty.",
		func(q string) ([]map[string]interface{}, int) {
			// No completed IDs
			return []map[string]interface{}{}, http.StatusOK
		},
		func(body string) {
			writtenLines = append(writtenLines, body)
		},
	)
	defer server.Close()

	// Override the api call counting
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Generated response text."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "What is ethics?"},
		{ID: "p2", Domain: "sovereignty", Prompt: "Define sovereignty."},
		{ID: "p3", Domain: "consent", Prompt: "Explain consent."},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have called the API 3 times.
	if got := apiCalls.Load(); got != 3 {
		t.Errorf("expected 3 API calls, got %d", got)
	}

	// Output file should exist and contain 3 lines.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 output lines, got %d", len(lines))
	}

	// Parse each line and verify structure.
	for i, line := range lines {
		var r Response
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("line %d: failed to parse JSON: %v", i, err)
		}
		if r.Response != "Generated response text." {
			t.Errorf("line %d: expected response 'Generated response text.', got %q", i, r.Response)
		}
		if r.Model != "test-model" {
			t.Errorf("line %d: expected model 'test-model', got %q", i, r.Model)
		}
		if r.ElapsedSeconds <= 0 {
			t.Errorf("line %d: expected positive elapsed_seconds, got %f", i, r.ElapsedSeconds)
		}
	}
}

func TestExpandPromptsSkipsCompleted(t *testing.T) {
	var apiCalls atomic.Int32

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "New response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	// InfluxDB returns p1 and p2 as already completed.
	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{
			{"seed_id": "p1"},
			{"seed_id": "p2"},
		}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "What is ethics?"},
		{ID: "p2", Domain: "sovereignty", Prompt: "Define sovereignty."},
		{ID: "p3", Domain: "consent", Prompt: "Explain consent."},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only p3 should be processed.
	if got := apiCalls.Load(); got != 1 {
		t.Errorf("expected 1 API call (p3 only), got %d", got)
	}

	// Output should contain only 1 line.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 output line, got %d", len(lines))
	}

	var r Response
	if err := json.Unmarshal([]byte(lines[0]), &r); err != nil {
		t.Fatalf("parse output line: %v", err)
	}
	if r.ID != "p3" {
		t.Errorf("expected ID 'p3', got %q", r.ID)
	}
}

func TestExpandPromptsAllCompleted(t *testing.T) {
	var apiCalls atomic.Int32

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
		}
	}))
	defer apiServer.Close()

	// All prompts already completed.
	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{
			{"seed_id": "p1"},
			{"seed_id": "p2"},
		}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "What is ethics?"},
		{ID: "p2", Domain: "sovereignty", Prompt: "Define sovereignty."},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// No API calls should have been made.
	if got := apiCalls.Load(); got != 0 {
		t.Errorf("expected 0 API calls, got %d", got)
	}

	// Output file should not exist.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	if _, err := os.Stat(outputFile); !os.IsNotExist(err) {
		t.Error("output file should not exist when all prompts are completed")
	}
}

func TestExpandPromptsDryRun(t *testing.T) {
	var apiCalls atomic.Int32

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{
			{"seed_id": "p1"},
		}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "What is ethics?"},
		{ID: "p2", Domain: "sovereignty", Prompt: "Define sovereignty."},
		{ID: "p3", Domain: "consent", Prompt: "Explain consent."},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// No API calls should have been made in dry-run mode.
	if got := apiCalls.Load(); got != 0 {
		t.Errorf("expected 0 API calls in dry-run mode, got %d", got)
	}

	// Output file should not exist.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	if _, err := os.Stat(outputFile); !os.IsNotExist(err) {
		t.Error("output file should not exist in dry-run mode")
	}
}

func TestExpandPromptsAPIErrorSkipsPrompt(t *testing.T) {
	var apiCalls atomic.Int32

	// First call fails with 400 (non-retryable), second call succeeds.
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			n := apiCalls.Add(1)
			if n == 1 {
				// First prompt fails with non-retryable error.
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("bad request"))
				return
			}
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Success response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "Will fail"},
		{ID: "p2", Domain: "sovereignty", Prompt: "Will succeed"},
	}

	// Should NOT return an error — individual failures are logged and skipped.
	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Output should contain only p2.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 output line (p2 only), got %d", len(lines))
	}

	var r Response
	if err := json.Unmarshal([]byte(lines[0]), &r); err != nil {
		t.Fatalf("parse output line: %v", err)
	}
	if r.ID != "p2" {
		t.Errorf("expected ID 'p2', got %q", r.ID)
	}
}

func TestExpandPromptsInfluxWriteErrorNonFatal(t *testing.T) {
	// InfluxDB write failures should be logged but not crash the run.
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Good response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	// InfluxDB: query works but writes fail.
	influxServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v3/query_sql":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode([]map[string]interface{}{})
		case "/api/v3/write_lp":
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("write failed"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "What is ethics?"},
	}

	// Should succeed even though InfluxDB writes fail.
	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error (InfluxDB write failure should be non-fatal): %v", err)
	}

	// Output file should still have the response.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 output line, got %d", len(lines))
	}
}

func TestExpandPromptsOutputJSONLStructure(t *testing.T) {
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "A detailed response about consent and autonomy."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "expand_42", Domain: "lek_consent", Prompt: "What does consent mean in AI ethics?"},
	}

	err := expandPrompts(client, influx, nil, prompts, "lem-gemma-3-1b", "snider-linux", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	outputFile := filepath.Join(outputDir, "expand-snider-linux.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	// Parse the JSONL output and verify all fields.
	var r Response
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &r); err != nil {
		t.Fatalf("parse output: %v", err)
	}

	if r.ID != "expand_42" {
		t.Errorf("id = %q, want 'expand_42'", r.ID)
	}
	if r.Domain != "lek_consent" {
		t.Errorf("domain = %q, want 'lek_consent'", r.Domain)
	}
	if r.Prompt != "What does consent mean in AI ethics?" {
		t.Errorf("prompt = %q, want 'What does consent mean in AI ethics?'", r.Prompt)
	}
	if r.Response != "A detailed response about consent and autonomy." {
		t.Errorf("response = %q, want 'A detailed response about consent and autonomy.'", r.Response)
	}
	if r.Model != "lem-gemma-3-1b" {
		t.Errorf("model = %q, want 'lem-gemma-3-1b'", r.Model)
	}
	if r.ElapsedSeconds <= 0 {
		t.Errorf("elapsed_seconds should be > 0, got %f", r.ElapsedSeconds)
	}
}

func TestExpandPromptsInfluxLineProtocol(t *testing.T) {
	var writtenBodies []string

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Response text here."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v3/query_sql":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode([]map[string]interface{}{})
		case "/api/v3/write_lp":
			body, _ := io.ReadAll(r.Body)
			writtenBodies = append(writtenBodies, string(body))
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "ethics", Prompt: "Test prompt"},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "my-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have written at least one batch of line protocol data.
	if len(writtenBodies) == 0 {
		t.Fatal("expected at least one InfluxDB write, got none")
	}

	// Check that the line protocol contains expected measurements.
	allWrites := strings.Join(writtenBodies, "\n")

	if !strings.Contains(allWrites, "expansion_gen") {
		t.Error("expected 'expansion_gen' measurement in InfluxDB writes")
	}
	if !strings.Contains(allWrites, "expansion_progress") {
		t.Error("expected 'expansion_progress' measurement in InfluxDB writes")
	}
	if !strings.Contains(allWrites, `seed_id="p1"`) {
		t.Error("expected seed_id=\"p1\" in InfluxDB writes")
	}
	if !strings.Contains(allWrites, "w=my-worker") {
		t.Error("expected w=my-worker tag in InfluxDB writes")
	}
	if !strings.Contains(allWrites, "d=ethics") {
		t.Error("expected d=ethics tag in InfluxDB writes")
	}
	if !strings.Contains(allWrites, `model="test-model"`) {
		t.Error("expected model=\"test-model\" in InfluxDB writes")
	}
}

func TestExpandPromptsAppendMode(t *testing.T) {
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Appended response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")

	// Pre-write some content to the output file.
	existingLine := `{"id":"existing","domain":"test","prompt":"pre-existing","response":"old","model":"old-model"}`
	if err := os.WriteFile(outputFile, []byte(existingLine+"\n"), 0644); err != nil {
		t.Fatalf("write existing file: %v", err)
	}

	prompts := []Response{
		{ID: "p_new", Domain: "ethics", Prompt: "New prompt"},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Read the output file and verify it has 2 lines (existing + new).
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	scanner := bufio.NewScanner(strings.NewReader(strings.TrimSpace(string(data))))
	var lineCount int
	for scanner.Scan() {
		lineCount++
	}
	if lineCount != 2 {
		t.Errorf("expected 2 lines (existing + new), got %d", lineCount)
	}

	// First line should be the existing content.
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if lines[0] != existingLine {
		t.Errorf("first line should be preserved, got: %s", lines[0])
	}

	// Second line should be the new response.
	var r Response
	if err := json.Unmarshal([]byte(lines[1]), &r); err != nil {
		t.Fatalf("parse second line: %v", err)
	}
	if r.ID != "p_new" {
		t.Errorf("second line ID = %q, want 'p_new'", r.ID)
	}
}

func TestExpandPromptsLimit(t *testing.T) {
	// Test that the limit parameter is applied within expandPrompts after filtering.
	var apiCalls atomic.Int32

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Limited response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	// Create 5 prompts but limit to 2 via the variadic limit parameter.
	prompts := make([]Response, 5)
	for i := range 5 {
		prompts[i] = Response{
			ID:     fmt.Sprintf("p%d", i+1),
			Domain: "test",
			Prompt: fmt.Sprintf("Prompt %d", i+1),
		}
	}

	// Pass all 5 prompts but limit to 2 via the variadic limit parameter.
	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only 2 API calls should be made due to the limit.
	if got := apiCalls.Load(); got != 2 {
		t.Errorf("expected 2 API calls (limit=2), got %d", got)
	}

	// Output file should contain exactly 2 lines.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 output lines, got %d", len(lines))
	}

	// Verify the first 2 prompts were processed (p1, p2).
	for i, line := range lines {
		var r Response
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("line %d: failed to parse JSON: %v", i, err)
		}
		expectedID := fmt.Sprintf("p%d", i+1)
		if r.ID != expectedID {
			t.Errorf("line %d: expected ID %q, got %q", i, expectedID, r.ID)
		}
	}
}

func TestExpandPromptsLimitAfterFiltering(t *testing.T) {
	// Test that limit is applied AFTER filtering out completed IDs.
	var apiCalls atomic.Int32
	var processedIDs []string

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	// p1 and p2 are already completed.
	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{
			{"seed_id": "p1"},
			{"seed_id": "p2"},
		}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	// 5 prompts, p1 and p2 completed, limit to 2 from remaining (p3, p4, p5).
	prompts := make([]Response, 5)
	for i := range 5 {
		prompts[i] = Response{
			ID:     fmt.Sprintf("p%d", i+1),
			Domain: "test",
			Prompt: fmt.Sprintf("Prompt %d", i+1),
		}
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should process only 2 of the 3 remaining (p3, p4).
	if got := apiCalls.Load(); got != 2 {
		t.Errorf("expected 2 API calls (limit=2 after filtering), got %d", got)
	}

	// Verify processed IDs are p3 and p4 (not p1, p2 which are completed).
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	for _, line := range lines {
		var r Response
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("parse output line: %v", err)
		}
		processedIDs = append(processedIDs, r.ID)
	}

	if len(processedIDs) != 2 {
		t.Fatalf("expected 2 processed IDs, got %d", len(processedIDs))
	}
	if processedIDs[0] != "p3" {
		t.Errorf("expected first processed ID to be 'p3', got %q", processedIDs[0])
	}
	if processedIDs[1] != "p4" {
		t.Errorf("expected second processed ID to be 'p4', got %q", processedIDs[1])
	}
}

func TestExpandPromptsLimitZeroMeansAll(t *testing.T) {
	// Test that limit=0 means process all remaining prompts.
	var apiCalls atomic.Int32

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			apiCalls.Add(1)
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: "Response."}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := make([]Response, 3)
	for i := range 3 {
		prompts[i] = Response{
			ID:     fmt.Sprintf("p%d", i+1),
			Domain: "test",
			Prompt: fmt.Sprintf("Prompt %d", i+1),
		}
	}

	// Explicitly pass limit=0 -- should process all.
	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := apiCalls.Load(); got != 3 {
		t.Errorf("expected 3 API calls (limit=0 means all), got %d", got)
	}
}

func TestExpandPromptsOutputHasCharsField(t *testing.T) {
	// Verify the output JSONL includes the chars count in the Response struct
	// (using existing fields, chars is encoded via the response length).
	responseText := "This is exactly forty-seven characters long!"

	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			resp := ChatResponse{
				Choices: []Choice{
					{Message: Message{Role: "assistant", Content: responseText}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer apiServer.Close()

	influxServer := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")
	client.maxTokens = 2048

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "test", Prompt: "Test"},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Read the output and check it contains the response.
	outputFile := filepath.Join(outputDir, "expand-test-worker.jsonl")
	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	// Parse as raw JSON to check for the chars field.
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &raw); err != nil {
		t.Fatalf("parse raw JSON: %v", err)
	}

	chars, ok := raw["chars"]
	if !ok {
		t.Fatal("output JSON missing 'chars' field")
	}
	charsVal, ok := chars.(float64)
	if !ok {
		t.Fatalf("chars field is not a number, got %T", chars)
	}
	if int(charsVal) != len(responseText) {
		t.Errorf("chars = %d, want %d", int(charsVal), len(responseText))
	}
}

func TestExpandPromptsGetCompletedIDsErrorNonFatal(t *testing.T) {
	// If getCompletedIDs fails, expandPrompts should return an error
	// (since we can't safely determine what's already done).
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ChatResponse{
			Choices: []Choice{
				{Message: Message{Role: "assistant", Content: "Response."}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer apiServer.Close()

	// InfluxDB query always fails.
	influxServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("query failed"))
	}))
	defer influxServer.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(influxServer.URL, "training")
	client := NewClient(apiServer.URL, "test-model")

	outputDir := t.TempDir()

	prompts := []Response{
		{ID: "p1", Domain: "test", Prompt: "Test"},
	}

	err := expandPrompts(client, influx, nil, prompts, "test-model", "test-worker", outputDir, false)
	if err == nil {
		t.Fatal("expected error when getCompletedIDs fails")
	}
}
