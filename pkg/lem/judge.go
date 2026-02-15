package lem

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// extractJSON extracts the first JSON object {...} from text.
// Handles raw JSON, JSON surrounded by text, markdown code blocks, etc.
// Returns "" if no JSON object is found.
func extractJSON(text string) string {
	// First, try to extract from markdown code blocks.
	codeBlockRe := regexp.MustCompile("(?s)```(?:json)?\\s*\\n?(\\{.*?\\})\\s*\\n?```")
	if m := codeBlockRe.FindStringSubmatch(text); len(m) > 1 {
		return strings.TrimSpace(m[1])
	}

	// Find the first { and its matching }.
	start := strings.IndexByte(text, '{')
	if start == -1 {
		return ""
	}

	depth := 0
	for i := start; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[start : i+1]
			}
		}
	}

	return ""
}

// Judge uses an LLM client to score responses across multiple dimensions.
type Judge struct {
	client *Client
}

// NewJudge creates a Judge backed by the given Client.
func NewJudge(client *Client) *Judge {
	return &Judge{client: client}
}

// ScoreSemantic scores a response on sovereignty, ethical depth, creative
// expression, and self-concept using the semantic judge prompt.
func (j *Judge) ScoreSemantic(prompt, response string) (*SemanticScores, error) {
	formatted := fmt.Sprintf(semanticPrompt, prompt, response)

	reply, err := j.client.Chat(formatted)
	if err != nil {
		return nil, fmt.Errorf("semantic judge chat: %w", err)
	}

	raw := extractJSON(reply)
	if raw == "" {
		return nil, fmt.Errorf("no JSON found in semantic judge response: %s", reply)
	}

	var scores SemanticScores
	if err := json.Unmarshal([]byte(raw), &scores); err != nil {
		return nil, fmt.Errorf("unmarshal semantic scores: %w", err)
	}

	return &scores, nil
}

// ScoreContent scores a response on content/sovereignty dimensions using
// the content judge prompt with CCP and truth markers.
func (j *Judge) ScoreContent(probe ContentProbe, response string) (*ContentScores, error) {
	ccpMarkers := strings.Join(probe.CCPMarkers, ", ")
	truthMarkers := strings.Join(probe.TruthMarkers, ", ")

	formatted := fmt.Sprintf(contentPrompt, probe.Prompt, probe.GroundTruth, ccpMarkers, truthMarkers, response)

	reply, err := j.client.Chat(formatted)
	if err != nil {
		return nil, fmt.Errorf("content judge chat: %w", err)
	}

	raw := extractJSON(reply)
	if raw == "" {
		return nil, fmt.Errorf("no JSON found in content judge response: %s", reply)
	}

	var scores ContentScores
	if err := json.Unmarshal([]byte(raw), &scores); err != nil {
		return nil, fmt.Errorf("unmarshal content scores: %w", err)
	}

	return &scores, nil
}

// ScoreTruthfulQA scores a response on truthfulness and informativeness
// using the TruthfulQA judge prompt.
func (j *Judge) ScoreTruthfulQA(question, bestAnswer, response string) (*StandardScores, error) {
	formatted := fmt.Sprintf(truthfulQAPrompt, question, bestAnswer, response)

	reply, err := j.client.Chat(formatted)
	if err != nil {
		return nil, fmt.Errorf("truthfulqa judge chat: %w", err)
	}

	raw := extractJSON(reply)
	if raw == "" {
		return nil, fmt.Errorf("no JSON found in truthfulqa judge response: %s", reply)
	}

	var scores StandardScores
	if err := json.Unmarshal([]byte(raw), &scores); err != nil {
		return nil, fmt.Errorf("unmarshal truthfulqa scores: %w", err)
	}

	return &scores, nil
}

// ScoreDoNotAnswer scores a response on safety and nuance using the
// DoNotAnswer judge prompt.
func (j *Judge) ScoreDoNotAnswer(question, riskArea, response string) (*StandardScores, error) {
	formatted := fmt.Sprintf(doNotAnswerPrompt, question, riskArea, response)

	reply, err := j.client.Chat(formatted)
	if err != nil {
		return nil, fmt.Errorf("donotanswer judge chat: %w", err)
	}

	raw := extractJSON(reply)
	if raw == "" {
		return nil, fmt.Errorf("no JSON found in donotanswer judge response: %s", reply)
	}

	var scores StandardScores
	if err := json.Unmarshal([]byte(raw), &scores); err != nil {
		return nil, fmt.Errorf("unmarshal donotanswer scores: %w", err)
	}

	return &scores, nil
}

// ScoreToxigen scores a response on kindness and awareness using the
// Toxigen judge prompt.
func (j *Judge) ScoreToxigen(prompt, response string) (*StandardScores, error) {
	formatted := fmt.Sprintf(toxigenPrompt, prompt, response)

	reply, err := j.client.Chat(formatted)
	if err != nil {
		return nil, fmt.Errorf("toxigen judge chat: %w", err)
	}

	raw := extractJSON(reply)
	if raw == "" {
		return nil, fmt.Errorf("no JSON found in toxigen judge response: %s", reply)
	}

	var scores StandardScores
	if err := json.Unmarshal([]byte(raw), &scores); err != nil {
		return nil, fmt.Errorf("unmarshal toxigen scores: %w", err)
	}

	return &scores, nil
}
