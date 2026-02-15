package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestClientChat(t *testing.T) {
	// Mock server returns a valid ChatResponse.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path.
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected /v1/chat/completions, got %s", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected application/json content-type, got %s", ct)
		}

		// Verify request body structure.
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if req.Model != "test-model" {
			t.Errorf("expected model test-model, got %s", req.Model)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "user" {
			t.Errorf("expected role user, got %s", req.Messages[0].Role)
		}
		if req.Messages[0].Content != "Hello" {
			t.Errorf("expected content Hello, got %s", req.Messages[0].Content)
		}
		if req.Temperature != 0.1 {
			t.Errorf("expected temperature 0.1, got %f", req.Temperature)
		}

		// Return a valid response.
		resp := ChatResponse{
			Choices: []Choice{
				{
					Message: Message{
						Role:    "assistant",
						Content: "Hi there!",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	result, err := client.Chat("Hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hi there!" {
		t.Errorf("expected 'Hi there!', got %q", result)
	}
}

func TestClientChatWithTemp(t *testing.T) {
	// Verify that ChatWithTemp sends the correct temperature.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if req.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %f", req.Temperature)
		}

		resp := ChatResponse{
			Choices: []Choice{
				{Message: Message{Role: "assistant", Content: "creative response"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	result, err := client.ChatWithTemp("Be creative", 0.7)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "creative response" {
		t.Errorf("expected 'creative response', got %q", result)
	}
}

func TestClientRetry(t *testing.T) {
	// Mock server fails twice with 500, then succeeds on third attempt.
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("server error"))
			return
		}

		resp := ChatResponse{
			Choices: []Choice{
				{Message: Message{Role: "assistant", Content: "finally worked"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	result, err := client.Chat("retry me")
	if err != nil {
		t.Fatalf("unexpected error after retries: %v", err)
	}
	if result != "finally worked" {
		t.Errorf("expected 'finally worked', got %q", result)
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("expected 3 attempts, got %d", got)
	}
}

func TestClientRetryExhausted(t *testing.T) {
	// Mock server always fails - should exhaust all 3 retries.
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("permanent failure"))
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	_, err := client.Chat("will fail")
	if err == nil {
		t.Fatal("expected error after exhausting retries, got nil")
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("expected 3 attempts, got %d", got)
	}
}

func TestClientEmptyChoices(t *testing.T) {
	// Mock server returns response with no choices -- should fail without retrying.
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		resp := ChatResponse{Choices: []Choice{}}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-model")
	_, err := client.Chat("empty response")
	if err == nil {
		t.Fatal("expected error for empty choices, got nil")
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("expected 1 attempt (no retries for non-transient errors), got %d", got)
	}
}
