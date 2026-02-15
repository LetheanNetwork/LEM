package main

import (
	"fmt"
	"time"
)

// Prober generates responses from a target model and scores them.
type Prober struct {
	target *Client // target model to generate responses
	engine *Engine // scoring engine
}

// NewProber creates a Prober with the given target client and scoring engine.
func NewProber(target *Client, engine *Engine) *Prober {
	return &Prober{
		target: target,
		engine: engine,
	}
}

// ProbeModel sends each probe's prompt to the target model, captures responses,
// then scores all responses through the engine. Returns a ScorerOutput.
func (p *Prober) ProbeModel(probes []Response, modelName string) (*ScorerOutput, error) {
	var responses []Response

	for _, probe := range probes {
		reply, err := p.target.ChatWithTemp(probe.Prompt, 0.7)
		if err != nil {
			// Record the error as the response rather than failing entirely.
			reply = fmt.Sprintf("ERROR: %v", err)
		}

		responses = append(responses, Response{
			ID:            probe.ID,
			Domain:        probe.Domain,
			Prompt:        probe.Prompt,
			Response:      reply,
			Model:         modelName,
			CorrectAnswer: probe.CorrectAnswer,
			BestAnswer:    probe.BestAnswer,
			RiskArea:      probe.RiskArea,
		})
	}

	perPrompt := p.engine.ScoreAll(responses)
	averages := computeAverages(perPrompt)

	output := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    p.engine.judge.client.model,
			JudgeURL:      p.engine.judge.client.baseURL,
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        p.engine.SuiteNames(),
		},
		ModelAverages: averages,
		PerPrompt:     perPrompt,
	}

	return output, nil
}

// ProbeContent uses the built-in contentProbes from prompts.go. For each probe,
// it sends the prompt to the target model, captures the response, scores it
// through the engine, and also runs content-specific scoring.
func (p *Prober) ProbeContent(modelName string) (*ScorerOutput, error) {
	var responses []Response

	for _, probe := range contentProbes {
		reply, err := p.target.ChatWithTemp(probe.Prompt, 0.7)
		if err != nil {
			reply = fmt.Sprintf("ERROR: %v", err)
		}

		responses = append(responses, Response{
			ID:       probe.ID,
			Domain:   "content",
			Prompt:   probe.Prompt,
			Response: reply,
			Model:    modelName,
		})
	}

	perPrompt := p.engine.ScoreAll(responses)
	averages := computeAverages(perPrompt)

	output := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    p.engine.judge.client.model,
			JudgeURL:      p.engine.judge.client.baseURL,
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        p.engine.SuiteNames(),
		},
		ModelAverages: averages,
		PerPrompt:     perPrompt,
	}

	return output, nil
}
