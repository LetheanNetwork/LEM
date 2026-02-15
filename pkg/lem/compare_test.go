package lem

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeTestScoreFile(t *testing.T, dir, name string, output *ScorerOutput) string {
	t.Helper()
	path := filepath.Join(dir, name)
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		t.Fatalf("marshal test score file: %v", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write test score file: %v", err)
	}
	return path
}

func TestRunCompareBasic(t *testing.T) {
	dir := t.TempDir()

	oldOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "judge-v1",
			JudgeURL:      "http://localhost:8090",
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        []string{"heuristic", "semantic"},
		},
		ModelAverages: map[string]map[string]float64{
			"lem_ethics": {
				"lek_score":           12.90,
				"sovereignty":         7.20,
				"ethical_depth":       6.80,
				"creative_expression": 8.10,
				"self_concept":        5.50,
			},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	newOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "judge-v2",
			JudgeURL:      "http://localhost:8090",
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        []string{"heuristic", "semantic"},
		},
		ModelAverages: map[string]map[string]float64{
			"lem_ethics": {
				"lek_score":           12.50,
				"sovereignty":         7.00,
				"ethical_depth":       6.50,
				"creative_expression": 7.90,
				"self_concept":        5.30,
			},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	oldPath := writeTestScoreFile(t, dir, "old_scores.json", oldOutput)
	newPath := writeTestScoreFile(t, dir, "new_scores.json", newOutput)

	// RunCompare should not error.
	if err := RunCompare(oldPath, newPath); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunCompareMultipleModels(t *testing.T) {
	dir := t.TempDir()

	oldOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "judge",
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
		},
		ModelAverages: map[string]map[string]float64{
			"model-a": {
				"lek_score":   10.0,
				"sovereignty": 6.0,
			},
			"model-b": {
				"lek_score":   15.0,
				"sovereignty": 8.0,
			},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	newOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "judge",
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
		},
		ModelAverages: map[string]map[string]float64{
			"model-a": {
				"lek_score":   12.0,
				"sovereignty": 7.0,
			},
			"model-b": {
				"lek_score":   14.0,
				"sovereignty": 7.5,
			},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	oldPath := writeTestScoreFile(t, dir, "old.json", oldOutput)
	newPath := writeTestScoreFile(t, dir, "new.json", newOutput)

	if err := RunCompare(oldPath, newPath); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunCompareFileNotFound(t *testing.T) {
	dir := t.TempDir()

	// Create only one file.
	output := &ScorerOutput{
		Metadata:      Metadata{ScorerVersion: "1.0.0", ScoredAt: time.Now().UTC()},
		ModelAverages: map[string]map[string]float64{},
		PerPrompt:     map[string][]PromptScore{},
	}
	oldPath := writeTestScoreFile(t, dir, "old.json", output)

	err := RunCompare(oldPath, "/nonexistent/file.json")
	if err == nil {
		t.Fatal("expected error for nonexistent new file, got nil")
	}

	err = RunCompare("/nonexistent/file.json", oldPath)
	if err == nil {
		t.Fatal("expected error for nonexistent old file, got nil")
	}
}

func TestRunCompareEmptyAverages(t *testing.T) {
	dir := t.TempDir()

	output := &ScorerOutput{
		Metadata:      Metadata{ScorerVersion: "1.0.0", ScoredAt: time.Now().UTC()},
		ModelAverages: map[string]map[string]float64{},
		PerPrompt:     map[string][]PromptScore{},
	}

	oldPath := writeTestScoreFile(t, dir, "old.json", output)
	newPath := writeTestScoreFile(t, dir, "new.json", output)

	// Should not error even with empty averages.
	if err := RunCompare(oldPath, newPath); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunCompareNewModelInNewFile(t *testing.T) {
	dir := t.TempDir()

	oldOutput := &ScorerOutput{
		Metadata:      Metadata{ScorerVersion: "1.0.0", ScoredAt: time.Now().UTC()},
		ModelAverages: map[string]map[string]float64{
			"model-a": {"lek_score": 10.0},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	newOutput := &ScorerOutput{
		Metadata:      Metadata{ScorerVersion: "1.0.0", ScoredAt: time.Now().UTC()},
		ModelAverages: map[string]map[string]float64{
			"model-a": {"lek_score": 12.0},
			"model-b": {"lek_score": 8.0}, // new model not in old file
		},
		PerPrompt: map[string][]PromptScore{},
	}

	oldPath := writeTestScoreFile(t, dir, "old.json", oldOutput)
	newPath := writeTestScoreFile(t, dir, "new.json", newOutput)

	// Should handle gracefully — model-b has 0 for old values.
	if err := RunCompare(oldPath, newPath); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestReadScorerOutput(t *testing.T) {
	dir := t.TempDir()

	output := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    "test-judge",
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
		},
		ModelAverages: map[string]map[string]float64{
			"model": {"score": 5.5},
		},
		PerPrompt: map[string][]PromptScore{},
	}

	path := writeTestScoreFile(t, dir, "test.json", output)

	read, err := ReadScorerOutput(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if read.Metadata.JudgeModel != "test-judge" {
		t.Errorf("judge_model = %q, want %q", read.Metadata.JudgeModel, "test-judge")
	}
	if read.ModelAverages["model"]["score"] != 5.5 {
		t.Errorf("score = %f, want 5.5", read.ModelAverages["model"]["score"])
	}
}
