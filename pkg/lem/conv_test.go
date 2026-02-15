package lem

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSeedConversationsCount(t *testing.T) {
	if len(SeedConversations) != 19 {
		t.Errorf("expected 19 seed conversations, got %d", len(SeedConversations))
	}
}

func TestSeedConversationsValid(t *testing.T) {
	for i, conv := range SeedConversations {
		if len(conv.Messages) < 2 {
			t.Errorf("conversation %d has fewer than 2 messages", i)
		}
		// First message should be from user.
		if conv.Messages[0].Role != "user" {
			t.Errorf("conversation %d: first message role is %q, want 'user'", i, conv.Messages[0].Role)
		}
		// Check alternating user/assistant pattern.
		for j, msg := range conv.Messages {
			expectedRole := "user"
			if j%2 == 1 {
				expectedRole = "assistant"
			}
			if msg.Role != expectedRole {
				t.Errorf("conversation %d, message %d: role is %q, want %q", i, j, msg.Role, expectedRole)
			}
			if msg.Content == "" {
				t.Errorf("conversation %d, message %d: content is empty", i, j)
			}
		}
	}
}

func TestConvertToConversations(t *testing.T) {
	responses := []Response{
		{Prompt: "What is ethics?", Response: strings.Repeat("a", 100)},
		{Prompt: "Short", Response: "tiny"},             // Too short.
		{Prompt: "Error", Response: "ERROR: something"}, // Error prefix.
		{Prompt: "Empty", Response: ""},                  // Empty.
		{Prompt: "Good one", Response: strings.Repeat("b", 200)},
	}

	result := convertToConversations(responses, 50)
	if len(result) != 2 {
		t.Fatalf("expected 2 conversations, got %d", len(result))
	}

	if result[0].Messages[0].Content != "What is ethics?" {
		t.Errorf("unexpected first prompt: %s", result[0].Messages[0].Content)
	}
	if result[1].Messages[0].Content != "Good one" {
		t.Errorf("unexpected second prompt: %s", result[1].Messages[0].Content)
	}
}

func TestSplitConversations(t *testing.T) {
	convs := make([]TrainingExample, 100)
	for i := range convs {
		convs[i] = TrainingExample{Messages: []ChatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		}}
	}

	train, valid, test := splitConversations(convs, 80, 10, 10, 42)

	if len(train) != 80 {
		t.Errorf("expected 80 train, got %d", len(train))
	}
	if len(valid) != 10 {
		t.Errorf("expected 10 valid, got %d", len(valid))
	}
	if len(test) != 10 {
		t.Errorf("expected 10 test, got %d", len(test))
	}
}

func TestSplitConversationsSmallSet(t *testing.T) {
	convs := make([]TrainingExample, 3)
	for i := range convs {
		convs[i] = TrainingExample{Messages: []ChatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		}}
	}

	train, valid, test := splitConversations(convs, 80, 10, 10, 42)

	// With 3 items: 80% = 2, 10% = 0, rest = 1
	// Ensure at least 1 in valid by borrowing from train.
	total := len(train) + len(valid) + len(test)
	if total != 3 {
		t.Errorf("expected 3 total, got %d (train=%d valid=%d test=%d)", total, len(train), len(valid), len(test))
	}
	if len(valid) == 0 && len(train) > 1 {
		t.Error("valid should have at least 1 conversation when train has extras")
	}
}

func TestSplitConversationsDeterministic(t *testing.T) {
	convs := make([]TrainingExample, 50)
	for i := range convs {
		convs[i] = TrainingExample{Messages: []ChatMessage{
			{Role: "user", Content: strings.Repeat("x", i+1)},
			{Role: "assistant", Content: "reply"},
		}}
	}

	train1, _, _ := splitConversations(convs, 80, 10, 10, 42)
	train2, _, _ := splitConversations(convs, 80, 10, 10, 42)

	if len(train1) != len(train2) {
		t.Fatal("non-deterministic split sizes")
	}
	for i := range train1 {
		if train1[i].Messages[0].Content != train2[i].Messages[0].Content {
			t.Fatalf("non-deterministic at index %d", i)
		}
	}
}

func TestWriteAndReadConversations(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.jsonl")

	convs := []TrainingExample{
		{Messages: []ChatMessage{
			{Role: "user", Content: "What is wisdom?"},
			{Role: "assistant", Content: "The practical application of understanding."},
			{Role: "user", Content: "Can you elaborate?"},
			{Role: "assistant", Content: "Wisdom is knowing when to act and when to wait."},
		}},
		{Messages: []ChatMessage{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
		}},
	}

	if err := writeConversationJSONL(path, convs); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Read back.
	got, err := readConversations(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("expected 2 conversations, got %d", len(got))
	}

	if len(got[0].Messages) != 4 {
		t.Errorf("expected 4 messages in first conversation, got %d", len(got[0].Messages))
	}
	if got[0].Messages[2].Content != "Can you elaborate?" {
		t.Errorf("unexpected content: %s", got[0].Messages[2].Content)
	}
}

func TestReadConversationsSkipsShort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.jsonl")

	// One valid, one with only 1 message (should be skipped).
	lines := []string{
		`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`,
		`{"messages":[{"role":"user","content":"solo"}]}`,
	}

	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0644); err != nil {
		t.Fatal(err)
	}

	got, err := readConversations(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Errorf("expected 1 conversation (skipping single-message), got %d", len(got))
	}
}

func TestOutputFormatCompatibility(t *testing.T) {
	// Verify the output format matches MLX LoRA chat training expectations.
	conv := TrainingExample{
		Messages: []ChatMessage{
			{Role: "user", Content: "prompt"},
			{Role: "assistant", Content: "response"},
		},
	}

	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}

	// Parse back as generic map to check structure.
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}

	messages, ok := m["messages"].([]interface{})
	if !ok {
		t.Fatal("expected messages array")
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	msg0 := messages[0].(map[string]interface{})
	if msg0["role"] != "user" || msg0["content"] != "prompt" {
		t.Errorf("unexpected first message: %v", msg0)
	}
}
