package lem

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// workerConfig holds the worker's runtime configuration.
type workerConfig struct {
	apiBase      string
	workerID     string
	name         string
	apiKey       string
	gpuType      string
	vramGb       int
	languages    []string
	models       []string
	inferURL     string
	taskType     string
	batchSize    int
	pollInterval time.Duration
	oneShot      bool
	dryRun       bool
}

// apiTask represents a task from the LEM API.
type apiTask struct {
	ID         int    `json:"id"`
	TaskType   string `json:"task_type"`
	Status     string `json:"status"`
	Language   string `json:"language"`
	Domain     string `json:"domain"`
	ModelName  string `json:"model_name"`
	PromptID   string `json:"prompt_id"`
	PromptText string `json:"prompt_text"`
	Config     *struct {
		Temperature float64 `json:"temperature,omitempty"`
		MaxTokens   int     `json:"max_tokens,omitempty"`
	} `json:"config"`
	Priority int `json:"priority"`
}

// RunWorker is the CLI entry point for `lem worker`.
func RunWorker(args []string) {
	fs := flag.NewFlagSet("worker", flag.ExitOnError)

	cfg := workerConfig{}
	var langs, mods string

	fs.StringVar(&cfg.apiBase, "api", envOr("LEM_API", "https://infer.lthn.ai"), "LEM API base URL")
	fs.StringVar(&cfg.workerID, "id", envOr("LEM_WORKER_ID", machineID()), "Worker ID (machine UUID)")
	fs.StringVar(&cfg.name, "name", envOr("LEM_WORKER_NAME", hostname()), "Worker display name")
	fs.StringVar(&cfg.apiKey, "key", envOr("LEM_API_KEY", ""), "API key (or use LEM_API_KEY env)")
	fs.StringVar(&cfg.gpuType, "gpu", envOr("LEM_GPU", ""), "GPU type (e.g. 'RTX 3090')")
	fs.IntVar(&cfg.vramGb, "vram", intEnvOr("LEM_VRAM_GB", 0), "GPU VRAM in GB")
	fs.StringVar(&langs, "languages", envOr("LEM_LANGUAGES", ""), "Comma-separated language codes (e.g. 'en,yo,sw')")
	fs.StringVar(&mods, "models", envOr("LEM_MODELS", ""), "Comma-separated supported model names")
	fs.StringVar(&cfg.inferURL, "infer", envOr("LEM_INFER_URL", "http://localhost:8090"), "Local inference endpoint")
	fs.StringVar(&cfg.taskType, "type", "", "Filter by task type (expand, score, translate, seed)")
	fs.IntVar(&cfg.batchSize, "batch", 5, "Number of tasks to fetch per poll")
	dur := fs.Duration("poll", 30*time.Second, "Poll interval")
	fs.BoolVar(&cfg.oneShot, "one-shot", false, "Process one batch and exit")
	fs.BoolVar(&cfg.dryRun, "dry-run", false, "Fetch tasks but don't run inference")

	fs.Parse(args)
	cfg.pollInterval = *dur

	if langs != "" {
		cfg.languages = splitComma(langs)
	}
	if mods != "" {
		cfg.models = splitComma(mods)
	}

	if cfg.apiKey == "" {
		cfg.apiKey = readKeyFile()
	}

	if cfg.apiKey == "" {
		log.Fatal("No API key. Set LEM_API_KEY or run: lem worker-auth")
	}

	log.Printf("LEM Worker starting")
	log.Printf("  ID:       %s", cfg.workerID)
	log.Printf("  Name:     %s", cfg.name)
	log.Printf("  API:      %s", cfg.apiBase)
	log.Printf("  Infer:    %s", cfg.inferURL)
	log.Printf("  GPU:      %s (%d GB)", cfg.gpuType, cfg.vramGb)
	log.Printf("  Langs:    %v", cfg.languages)
	log.Printf("  Models:   %v", cfg.models)
	log.Printf("  Batch:    %d", cfg.batchSize)
	log.Printf("  Dry-run:  %v", cfg.dryRun)

	// Register with the API.
	if err := workerRegister(&cfg); err != nil {
		log.Fatalf("Registration failed: %v", err)
	}
	log.Println("Registered with LEM API")

	// Main loop.
	for {
		processed := workerPoll(&cfg)

		if cfg.oneShot {
			log.Printf("One-shot mode: processed %d tasks, exiting", processed)
			return
		}

		if processed == 0 {
			log.Printf("No tasks available, sleeping %v", cfg.pollInterval)
			time.Sleep(cfg.pollInterval)
		}

		// Heartbeat.
		workerHeartbeat(&cfg)
	}
}

func workerRegister(cfg *workerConfig) error {
	body := map[string]interface{}{
		"worker_id": cfg.workerID,
		"name":      cfg.name,
		"version":   "0.1.0",
		"os":        runtime.GOOS,
		"arch":      runtime.GOARCH,
	}
	if cfg.gpuType != "" {
		body["gpu_type"] = cfg.gpuType
	}
	if cfg.vramGb > 0 {
		body["vram_gb"] = cfg.vramGb
	}
	if len(cfg.languages) > 0 {
		body["languages"] = cfg.languages
	}
	if len(cfg.models) > 0 {
		body["supported_models"] = cfg.models
	}

	_, err := apiPost(cfg, "/api/lem/workers/register", body)
	return err
}

func workerHeartbeat(cfg *workerConfig) {
	body := map[string]interface{}{
		"worker_id": cfg.workerID,
	}
	apiPost(cfg, "/api/lem/workers/heartbeat", body)
}

func workerPoll(cfg *workerConfig) int {
	// Fetch available tasks.
	url := fmt.Sprintf("/api/lem/tasks/next?worker_id=%s&limit=%d", cfg.workerID, cfg.batchSize)
	if cfg.taskType != "" {
		url += "&type=" + cfg.taskType
	}

	resp, err := apiGet(cfg, url)
	if err != nil {
		log.Printf("Error fetching tasks: %v", err)
		return 0
	}

	var result struct {
		Tasks []apiTask `json:"tasks"`
		Count int       `json:"count"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		log.Printf("Error parsing tasks: %v", err)
		return 0
	}

	if result.Count == 0 {
		return 0
	}

	log.Printf("Got %d tasks", result.Count)
	processed := 0

	for _, task := range result.Tasks {
		if err := workerProcessTask(cfg, task); err != nil {
			log.Printf("Task %d failed: %v", task.ID, err)
			// Release the claim so someone else can try.
			apiDelete(cfg, fmt.Sprintf("/api/lem/tasks/%d/claim", task.ID), map[string]interface{}{
				"worker_id": cfg.workerID,
			})
			continue
		}
		processed++
	}

	return processed
}

func workerProcessTask(cfg *workerConfig, task apiTask) error {
	log.Printf("Processing task %d: %s [%s/%s] %d chars prompt",
		task.ID, task.TaskType, task.Language, task.Domain, len(task.PromptText))

	// Claim the task.
	_, err := apiPost(cfg, fmt.Sprintf("/api/lem/tasks/%d/claim", task.ID), map[string]interface{}{
		"worker_id": cfg.workerID,
	})
	if err != nil {
		return fmt.Errorf("claim: %w", err)
	}

	// Update to in_progress.
	apiPatch(cfg, fmt.Sprintf("/api/lem/tasks/%d/status", task.ID), map[string]interface{}{
		"worker_id": cfg.workerID,
		"status":    "in_progress",
	})

	if cfg.dryRun {
		log.Printf("  [DRY-RUN] Would generate response for: %.80s...", task.PromptText)
		return nil
	}

	// Run inference via local API.
	start := time.Now()
	response, err := workerInfer(cfg, task)
	genTime := time.Since(start)

	if err != nil {
		// Report failure, release task.
		apiPatch(cfg, fmt.Sprintf("/api/lem/tasks/%d/status", task.ID), map[string]interface{}{
			"worker_id": cfg.workerID,
			"status":    "abandoned",
		})
		return fmt.Errorf("inference: %w", err)
	}

	// Submit result.
	modelUsed := task.ModelName
	if modelUsed == "" {
		modelUsed = "default"
	}

	_, err = apiPost(cfg, fmt.Sprintf("/api/lem/tasks/%d/result", task.ID), map[string]interface{}{
		"worker_id":     cfg.workerID,
		"response_text": response,
		"model_used":    modelUsed,
		"gen_time_ms":   int(genTime.Milliseconds()),
	})
	if err != nil {
		return fmt.Errorf("submit result: %w", err)
	}

	log.Printf("  Completed: %d chars in %v", len(response), genTime.Round(time.Millisecond))
	return nil
}

func workerInfer(cfg *workerConfig, task apiTask) (string, error) {
	// Build the chat request for the local inference endpoint.
	messages := []map[string]string{
		{"role": "user", "content": task.PromptText},
	}

	temp := 0.7
	maxTokens := 2048
	if task.Config != nil {
		if task.Config.Temperature > 0 {
			temp = task.Config.Temperature
		}
		if task.Config.MaxTokens > 0 {
			maxTokens = task.Config.MaxTokens
		}
	}

	reqBody := map[string]interface{}{
		"model":       task.ModelName,
		"messages":    messages,
		"temperature": temp,
		"max_tokens":  maxTokens,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", cfg.inferURL+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("inference request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("inference HTTP %d: %s", resp.StatusCode, truncStr(string(body), 200))
	}

	// Parse OpenAI-compatible response.
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	content := chatResp.Choices[0].Message.Content
	if len(content) < 10 {
		return "", fmt.Errorf("response too short: %d chars", len(content))
	}

	return content, nil
}

// HTTP helpers for the LEM API.

func apiGet(cfg *workerConfig, path string) ([]byte, error) {
	req, err := http.NewRequest("GET", cfg.apiBase+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.apiKey)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncStr(string(body), 200))
	}

	return body, nil
}

func apiPost(cfg *workerConfig, path string, data map[string]interface{}) ([]byte, error) {
	return apiRequest(cfg, "POST", path, data)
}

func apiPatch(cfg *workerConfig, path string, data map[string]interface{}) ([]byte, error) {
	return apiRequest(cfg, "PATCH", path, data)
}

func apiDelete(cfg *workerConfig, path string, data map[string]interface{}) ([]byte, error) {
	return apiRequest(cfg, "DELETE", path, data)
}

func apiRequest(cfg *workerConfig, method, path string, data map[string]interface{}) ([]byte, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(method, cfg.apiBase+path, bytes.NewReader(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncStr(string(body), 200))
	}

	return body, nil
}

// Utility functions.

func machineID() string {
	// Try /etc/machine-id (Linux), then hostname fallback.
	if data, err := os.ReadFile("/etc/machine-id"); err == nil {
		id := string(bytes.TrimSpace(data))
		if len(id) > 0 {
			return id
		}
	}
	h, _ := os.Hostname()
	return h
}

func hostname() string {
	h, _ := os.Hostname()
	return h
}

func readKeyFile() string {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, ".config", "lem", "api_key")
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(data))
}

func splitComma(s string) []string {
	var result []string
	for _, part := range bytes.Split([]byte(s), []byte(",")) {
		trimmed := bytes.TrimSpace(part)
		if len(trimmed) > 0 {
			result = append(result, string(trimmed))
		}
	}
	return result
}

func truncStr(s string, n int) string {
	maxLen := n
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
