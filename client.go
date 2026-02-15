package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Message is a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request body for /v1/chat/completions.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// Choice is a single completion choice.
type Choice struct {
	Message Message `json:"message"`
}

// ChatResponse is the response from /v1/chat/completions.
type ChatResponse struct {
	Choices []Choice `json:"choices"`
}

// retryableError marks errors that should be retried.
type retryableError struct {
	err error
}

func (e *retryableError) Error() string { return e.err.Error() }
func (e *retryableError) Unwrap() error { return e.err }

// Client talks to an OpenAI-compatible API.
type Client struct {
	baseURL    string
	model      string
	maxTokens  int
	httpClient *http.Client
}

// NewClient creates a Client for the given base URL and model.
func NewClient(baseURL, model string) *Client {
	return &Client{
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

// Chat sends a prompt and returns the assistant's reply.
// Uses the default temperature of 0.1.
func (c *Client) Chat(prompt string) (string, error) {
	return c.ChatWithTemp(prompt, 0.1)
}

// ChatWithTemp sends a prompt with a specific temperature and returns
// the assistant's reply. Retries up to 3 times with exponential backoff
// on transient failures (HTTP 5xx or network errors).
func (c *Client) ChatWithTemp(prompt string, temp float64) (string, error) {
	req := ChatRequest{
		Model: c.model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		Temperature: temp,
		MaxTokens:   c.maxTokens,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	maxAttempts := 3
	var lastErr error

	for attempt := range maxAttempts {
		if attempt > 0 {
			// Exponential backoff: 100ms, 200ms
			backoff := time.Duration(100<<uint(attempt-1)) * time.Millisecond
			time.Sleep(backoff)
		}

		result, err := c.doRequest(body)
		if err == nil {
			return result, nil
		}
		lastErr = err

		// Only retry on transient (retryable) errors.
		var re *retryableError
		if !errors.As(err, &re) {
			return "", err
		}
	}

	return "", fmt.Errorf("exhausted %d retries: %w", maxAttempts, lastErr)
}

// doRequest sends a single HTTP request and parses the response.
func (c *Client) doRequest(body []byte) (string, error) {
	url := c.baseURL + "/v1/chat/completions"

	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", &retryableError{fmt.Errorf("http request: %w", err)}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", &retryableError{fmt.Errorf("read response: %w", err)}
	}

	if resp.StatusCode >= 500 {
		return "", &retryableError{fmt.Errorf("server error %d: %s", resp.StatusCode, string(respBody))}
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}
