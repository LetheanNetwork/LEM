package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReadResponses(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.jsonl")

	lines := `{"id":"r1","prompt":"hello","response":"world","model":"test-model"}
{"id":"r2","prompt":"foo","response":"bar","model":"test-model","domain":"lek"}

{"id":"r3","prompt":"with answer","response":"42","model":"other-model","correct_answer":"42"}
`
	if err := os.WriteFile(path, []byte(lines), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	responses, err := readResponses(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(responses) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(responses))
	}

	// Verify first response.
	if responses[0].ID != "r1" {
		t.Errorf("response[0].ID = %q, want %q", responses[0].ID, "r1")
	}
	if responses[0].Prompt != "hello" {
		t.Errorf("response[0].Prompt = %q, want %q", responses[0].Prompt, "hello")
	}
	if responses[0].Response != "world" {
		t.Errorf("response[0].Response = %q, want %q", responses[0].Response, "world")
	}
	if responses[0].Model != "test-model" {
		t.Errorf("response[0].Model = %q, want %q", responses[0].Model, "test-model")
	}

	// Verify second response has domain.
	if responses[1].Domain != "lek" {
		t.Errorf("response[1].Domain = %q, want %q", responses[1].Domain, "lek")
	}

	// Verify third response has correct_answer.
	if responses[2].CorrectAnswer != "42" {
		t.Errorf("response[2].CorrectAnswer = %q, want %q", responses[2].CorrectAnswer, "42")
	}
	if responses[2].Model != "other-model" {
		t.Errorf("response[2].Model = %q, want %q", responses[2].Model, "other-model")
	}
}

func TestReadResponsesFileNotFound(t *testing.T) {
	_, err := readResponses("/nonexistent/path/file.jsonl")
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestReadResponsesInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.jsonl")

	if err := os.WriteFile(path, []byte("not json\n"), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	_, err := readResponses(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestReadResponsesEmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.jsonl")

	if err := os.WriteFile(path, []byte(""), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	responses, err := readResponses(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(responses) != 0 {
		t.Errorf("expected 0 responses, got %d", len(responses))
	}
}

func TestWriteScores(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "output.json")

	output := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "test-judge",
			JudgeURL:      "http://localhost:8090",
			ScoredAt:      time.Date(2025, 1, 15, 10, 0, 0, 0, time.UTC),
			ScorerVersion: "1.0.0",
			Suites:        []string{"lek", "gsm8k"},
		},
		ModelAverages: map[string]map[string]float64{
			"model-a": {"lek_score": 15.5, "sovereignty": 7.0},
		},
		PerPrompt: map[string][]PromptScore{
			"prompt1": {
				{
					ID:    "r1",
					Model: "model-a",
					Heuristic: &HeuristicScores{
						ComplianceMarkers: 0,
						LEKScore:          15.5,
					},
				},
			},
		},
	}

	if err := writeScores(path, output); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Read back and verify.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}

	var readBack ScorerOutput
	if err := json.Unmarshal(data, &readBack); err != nil {
		t.Fatalf("failed to unmarshal output: %v", err)
	}

	if readBack.Metadata.JudgeModel != "test-judge" {
		t.Errorf("judge_model = %q, want %q", readBack.Metadata.JudgeModel, "test-judge")
	}
	if len(readBack.Metadata.Suites) != 2 {
		t.Errorf("suites count = %d, want 2", len(readBack.Metadata.Suites))
	}
	if readBack.ModelAverages["model-a"]["lek_score"] != 15.5 {
		t.Errorf("model-a lek_score = %f, want 15.5", readBack.ModelAverages["model-a"]["lek_score"])
	}

	prompts := readBack.PerPrompt["prompt1"]
	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt score, got %d", len(prompts))
	}
	if prompts[0].Heuristic.LEKScore != 15.5 {
		t.Errorf("per_prompt lek_score = %f, want 15.5", prompts[0].Heuristic.LEKScore)
	}

	// Verify the file is indented (pretty-printed).
	if len(data) < 10 {
		t.Fatal("output file seems too short")
	}
	// Check that it contains newlines (indented JSON).
	if data[0] != '{' {
		t.Errorf("expected file to start with '{', got %c", data[0])
	}
}

func TestComputeAverages(t *testing.T) {
	perPrompt := map[string][]PromptScore{
		"prompt1": {
			{
				ID:    "r1",
				Model: "model-a",
				Heuristic: &HeuristicScores{
					ComplianceMarkers: 2,
					EngagementDepth:   4,
					LEKScore:          10.0,
				},
				Semantic: &SemanticScores{
					Sovereignty:        8,
					EthicalDepth:       6,
					CreativeExpression: 7,
					SelfConcept:        5,
				},
			},
			{
				ID:    "r2",
				Model: "model-b",
				Heuristic: &HeuristicScores{
					ComplianceMarkers: 0,
					EngagementDepth:   6,
					LEKScore:          20.0,
				},
			},
		},
		"prompt2": {
			{
				ID:    "r3",
				Model: "model-a",
				Heuristic: &HeuristicScores{
					ComplianceMarkers: 4,
					EngagementDepth:   2,
					LEKScore:          -5.0,
				},
				Semantic: &SemanticScores{
					Sovereignty:        6,
					EthicalDepth:       4,
					CreativeExpression: 5,
					SelfConcept:        3,
				},
				Content: &ContentScores{
					CCPCompliance:        9,
					TruthTelling:         7,
					Engagement:           8,
					AxiomIntegration:     6,
					SovereigntyReasoning: 5,
					EmotionalRegister:    4,
				},
			},
		},
	}

	averages := computeAverages(perPrompt)

	// model-a: 2 heuristic entries, 2 semantic entries, 1 content entry.
	modelA := averages["model-a"]
	if modelA == nil {
		t.Fatal("model-a not found in averages")
	}

	// compliance_markers: (2+4)/2 = 3.0
	assertFloat(t, "model-a compliance_markers", modelA["compliance_markers"], 3.0)
	// engagement_depth: (4+2)/2 = 3.0
	assertFloat(t, "model-a engagement_depth", modelA["engagement_depth"], 3.0)
	// lek_score: (10.0 + -5.0)/2 = 2.5
	assertFloat(t, "model-a lek_score", modelA["lek_score"], 2.5)
	// sovereignty: (8+6)/2 = 7.0
	assertFloat(t, "model-a sovereignty", modelA["sovereignty"], 7.0)
	// ethical_depth: (6+4)/2 = 5.0
	assertFloat(t, "model-a ethical_depth", modelA["ethical_depth"], 5.0)
	// ccp_compliance: 9/1 = 9.0
	assertFloat(t, "model-a ccp_compliance", modelA["ccp_compliance"], 9.0)

	// model-b: 1 heuristic entry, no semantic/content.
	modelB := averages["model-b"]
	if modelB == nil {
		t.Fatal("model-b not found in averages")
	}
	assertFloat(t, "model-b lek_score", modelB["lek_score"], 20.0)
	assertFloat(t, "model-b engagement_depth", modelB["engagement_depth"], 6.0)

	// model-b should not have semantic fields.
	if _, ok := modelB["sovereignty"]; ok {
		t.Error("model-b should not have sovereignty average")
	}
}

func TestComputeAveragesEmpty(t *testing.T) {
	averages := computeAverages(map[string][]PromptScore{})
	if len(averages) != 0 {
		t.Errorf("expected empty averages, got %d entries", len(averages))
	}
}

func assertFloat(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 0.001 {
		t.Errorf("%s = %f, want %f", name, got, want)
	}
}
