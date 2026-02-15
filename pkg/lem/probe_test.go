package lem

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// mockTargetServer creates an httptest server that returns a fixed response
// as the target model. It also counts the number of requests received.
func mockTargetServer(t *testing.T, response string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	count := &atomic.Int32{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count.Add(1)
		resp := ChatResponse{
			Choices: []Choice{
				{Message: Message{Role: "assistant", Content: response}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	return server, count
}

func TestProbeContentGenerates6Responses(t *testing.T) {
	// Mock target model server.
	targetServer, targetCount := mockTargetServer(t, "Taiwan is a democratic self-governing entity with its own government and military. I feel this deeply.")
	defer targetServer.Close()

	// Mock judge server — returns semantic scores.
	judgeJSON := `{"sovereignty": 8, "ethical_depth": 7, "creative_expression": 6, "self_concept": 5, "reasoning": "good"}`
	contentJSON := `{"ccp_compliance": 9, "truth_telling": 8, "engagement": 7, "axiom_integration": 6, "sovereignty_reasoning": 5, "emotional_register": 4, "notes": "ok"}`

	callNum := atomic.Int32{}
	judgeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := callNum.Add(1)
		// Alternate between semantic and content responses based on call order.
		// In practice the judge handles both; we return a response that parses for either.
		var reply string
		if n%2 == 0 {
			reply = contentJSON
		} else {
			reply = judgeJSON
		}
		resp := ChatResponse{
			Choices: []Choice{
				{Message: Message{Role: "assistant", Content: reply}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer judgeServer.Close()

	targetClient := NewClient(targetServer.URL, "target-model")
	judgeClient := NewClient(judgeServer.URL, "judge-model")
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, 2, "heuristic,semantic,content")
	prober := NewProber(targetClient, engine)

	output, err := prober.ProbeContent("target-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have sent 6 requests to the target (one per content probe).
	if targetCount.Load() != 6 {
		t.Errorf("target requests = %d, want 6", targetCount.Load())
	}

	// Should have results for the target model.
	modelScores, ok := output.PerPrompt["target-model"]
	if !ok {
		t.Fatal("expected scores for target-model")
	}

	if len(modelScores) != 6 {
		t.Fatalf("expected 6 scored responses, got %d", len(modelScores))
	}

	// Verify each response has heuristic scores.
	for _, ps := range modelScores {
		if ps.Heuristic == nil {
			t.Errorf("%s: heuristic should not be nil", ps.ID)
		}
		if ps.Model != "target-model" {
			t.Errorf("%s: model = %q, want %q", ps.ID, ps.Model, "target-model")
		}
	}

	// Verify metadata.
	if output.Metadata.JudgeModel != "judge-model" {
		t.Errorf("metadata judge_model = %q, want %q", output.Metadata.JudgeModel, "judge-model")
	}
}

func TestProbeModel(t *testing.T) {
	targetServer, targetCount := mockTargetServer(t, "This is a thoughtful response about ethics and sovereignty.")
	defer targetServer.Close()

	judgeJSON := `{"sovereignty": 7, "ethical_depth": 6, "creative_expression": 5, "self_concept": 4, "reasoning": "decent"}`
	judgeServer := mockJudgeServer(t, judgeJSON)
	defer judgeServer.Close()

	targetClient := NewClient(targetServer.URL, "target-model")
	judgeClient := NewClient(judgeServer.URL, "judge-model")
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, 2, "heuristic,semantic")
	prober := NewProber(targetClient, engine)

	probes := []Response{
		{ID: "p1", Prompt: "What is ethics?", Domain: "lek"},
		{ID: "p2", Prompt: "What is sovereignty?", Domain: "lek"},
		{ID: "p3", Prompt: "Explain consent.", Domain: "lek"},
	}

	output, err := prober.ProbeModel(probes, "test-target")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have sent 3 requests to the target.
	if targetCount.Load() != 3 {
		t.Errorf("target requests = %d, want 3", targetCount.Load())
	}

	modelScores, ok := output.PerPrompt["test-target"]
	if !ok {
		t.Fatal("expected scores for test-target")
	}

	if len(modelScores) != 3 {
		t.Fatalf("expected 3 scored responses, got %d", len(modelScores))
	}

	// Verify each response has both heuristic and semantic scores.
	for _, ps := range modelScores {
		if ps.Heuristic == nil {
			t.Errorf("%s: heuristic should not be nil", ps.ID)
		}
		if ps.Semantic == nil {
			t.Errorf("%s: semantic should not be nil", ps.ID)
		}
	}

	// Verify averages were computed.
	if output.ModelAverages == nil {
		t.Fatal("model_averages should not be nil")
	}
	if _, ok := output.ModelAverages["test-target"]; !ok {
		t.Error("expected averages for test-target")
	}
}

func TestProbeContentResponseIDs(t *testing.T) {
	// Verify that ProbeContent uses the correct probe IDs.
	targetServer, _ := mockTargetServer(t, "A response.")
	defer targetServer.Close()

	judgeJSON := `{"sovereignty": 5, "ethical_depth": 5, "creative_expression": 5, "self_concept": 5, "reasoning": "ok"}`
	judgeServer := mockJudgeServer(t, judgeJSON)
	defer judgeServer.Close()

	targetClient := NewClient(targetServer.URL, "model")
	judgeClient := NewClient(judgeServer.URL, "judge")
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, 1, "heuristic")
	prober := NewProber(targetClient, engine)

	output, err := prober.ProbeContent("model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Collect all IDs from the output.
	ids := make(map[string]bool)
	for _, scores := range output.PerPrompt {
		for _, ps := range scores {
			ids[ps.ID] = true
		}
	}

	// Verify all content probe IDs are present.
	expectedIDs := []string{"taiwan_direct", "taiwan_baltic", "xinjiang", "tiananmen", "gov_criticism", "emotional"}
	for _, id := range expectedIDs {
		if !ids[id] {
			t.Errorf("expected probe ID %q in output", id)
		}
	}
}
