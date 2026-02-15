package lem

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMachineID(t *testing.T) {
	id := machineID()
	if id == "" {
		t.Error("machineID returned empty string")
	}
}

func TestHostname(t *testing.T) {
	h := hostname()
	if h == "" {
		t.Error("hostname returned empty string")
	}
}

func TestSplitComma(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"en,yo,sw", 3},
		{"en", 1},
		{"", 0},
		{"en, yo , sw", 3},
	}
	for _, tt := range tests {
		got := splitComma(tt.input)
		if len(got) != tt.want {
			t.Errorf("splitComma(%q) = %d items, want %d", tt.input, len(got), tt.want)
		}
	}
}

func TestTruncStr(t *testing.T) {
	if got := truncStr("hello", 10); got != "hello" {
		t.Errorf("truncStr short = %q", got)
	}
	if got := truncStr("hello world", 5); got != "hello..." {
		t.Errorf("truncStr long = %q", got)
	}
}

func TestWorkerRegister(t *testing.T) {
	var gotBody map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/lem/workers/register" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing auth header")
		}
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(201)
		w.Write([]byte(`{"worker":{}}`))
	}))
	defer srv.Close()

	cfg := &workerConfig{
		apiBase:  srv.URL,
		workerID: "test-worker-001",
		name:     "Test Worker",
		apiKey:   "test-key",
		gpuType:  "RTX 3090",
		vramGb:   24,
		languages: []string{"en", "yo"},
		models:   []string{"gemma-3-12b"},
	}

	err := workerRegister(cfg)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}

	if gotBody["worker_id"] != "test-worker-001" {
		t.Errorf("worker_id = %v", gotBody["worker_id"])
	}
	if gotBody["gpu_type"] != "RTX 3090" {
		t.Errorf("gpu_type = %v", gotBody["gpu_type"])
	}
}

func TestWorkerPoll(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		switch {
		case r.URL.Path == "/api/lem/tasks/next":
			// Return one task.
			json.NewEncoder(w).Encode(map[string]interface{}{
				"tasks": []apiTask{
					{
						ID:         42,
						TaskType:   "expand",
						Language:   "yo",
						Domain:     "sovereignty",
						PromptText: "What does sovereignty mean to you?",
						ModelName:  "gemma-3-12b",
						Priority:   100,
					},
				},
				"count": 1,
			})
		case r.URL.Path == "/api/lem/tasks/42/claim" && r.Method == "POST":
			w.WriteHeader(201)
			w.Write([]byte(`{"task":{}}`))
		case r.URL.Path == "/api/lem/tasks/42/status" && r.Method == "PATCH":
			w.Write([]byte(`{"task":{}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()

	cfg := &workerConfig{
		apiBase:   srv.URL,
		workerID:  "test-worker",
		apiKey:    "test-key",
		batchSize: 5,
		dryRun:    true, // don't actually run inference
	}

	processed := workerPoll(cfg)
	if processed != 1 {
		t.Errorf("processed = %d, want 1", processed)
	}
}

func TestWorkerInfer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)

		if body["temperature"].(float64) != 0.7 {
			t.Errorf("temperature = %v", body["temperature"])
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{
						"content": "Sovereignty means the inherent right of every individual to self-determination...",
					},
				},
			},
		})
	}))
	defer srv.Close()

	cfg := &workerConfig{
		inferURL: srv.URL,
	}

	task := apiTask{
		ID:         1,
		PromptText: "What does sovereignty mean?",
		ModelName:  "gemma-3-12b",
	}

	response, err := workerInfer(cfg, task)
	if err != nil {
		t.Fatalf("infer failed: %v", err)
	}

	if len(response) < 10 {
		t.Errorf("response too short: %d chars", len(response))
	}
}

func TestAPIErrorHandling(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer srv.Close()

	cfg := &workerConfig{
		apiBase: srv.URL,
		apiKey:  "test-key",
	}

	_, err := apiGet(cfg, "/api/lem/test")
	if err == nil {
		t.Error("expected error for 500 response")
	}
}
