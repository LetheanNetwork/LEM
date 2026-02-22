package lem

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"forge.lthn.ai/core/go-i18n/reversal"
	"forge.lthn.ai/core/go-inference"
)

// DistillProbe is a single input prompt for distillation.
type DistillProbe struct {
	ID     string `json:"id"`
	Domain string `json:"domain,omitempty"`
	Prompt string `json:"prompt"`
	Source string `json:"-"`
}

// distillCandidate holds a single generation attempt with its scores.
type distillCandidate struct {
	Response string
	Grammar  GrammarScore
	Delta    DeltaScore
	Elapsed  time.Duration
}

// RunDistill is the CLI entry point for the distill command.
// Generates responses via native Metal inference, scores with go-i18n/reversal,
// writes passing examples as training JSONL.
func RunDistill(args []string) {
	fs := flag.NewFlagSet("distill", flag.ExitOnError)

	modelFlag := fs.String("model", "gemma3/27b", "Model config path (relative to .core/ai/models/)")
	probesFlag := fs.String("probes", "", "Probe set name from probes.yaml, or path to JSON file")
	outputFlag := fs.String("output", "", "Output JSONL path (defaults to model training dir)")
	lessonFlag := fs.Int("lesson", -1, "Lesson number to append to (defaults to probe set phase)")
	minScore := fs.Float64("min-score", 0, "Min grammar composite (0 = use ai.yaml default)")
	runs := fs.Int("runs", 0, "Generations per probe (0 = use ai.yaml default)")
	dryRun := fs.Bool("dry-run", false, "Show plan and exit without generating")
	root := fs.String("root", ".", "Project root (for .core/ai/ config)")
	cacheLimit := fs.Int("cache-limit", 0, "Metal cache limit in GB (0 = use ai.yaml default)")
	memLimit := fs.Int("mem-limit", 0, "Metal memory limit in GB (0 = use ai.yaml default)")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	// Load configs.
	aiCfg, err := LoadAIConfig(*root)
	if err != nil {
		log.Fatalf("load ai config: %v", err)
	}

	modelCfg, err := LoadModelConfig(*root, *modelFlag)
	if err != nil {
		log.Fatalf("load model config: %v", err)
	}

	genCfg := MergeGenerate(aiCfg.Generate, modelCfg.Generate)

	// Apply flag overrides.
	if *minScore == 0 {
		*minScore = aiCfg.Scorer.MinScore
	}
	if *runs == 0 {
		*runs = aiCfg.Distill.Runs
	}
	cacheLimitGB := aiCfg.Distill.CacheLimit
	if *cacheLimit > 0 {
		cacheLimitGB = *cacheLimit
	}
	memLimitGB := aiCfg.Distill.MemoryLimit
	if *memLimit > 0 {
		memLimitGB = *memLimit
	}

	// Load probes.
	probes, phase, err := loadDistillProbes(*root, *probesFlag)
	if err != nil {
		log.Fatalf("load probes: %v", err)
	}
	log.Printf("loaded %d probes", len(probes))

	// Determine output path.
	outputPath := *outputFlag
	if outputPath == "" {
		lesson := *lessonFlag
		if lesson < 0 {
			lesson = phase
		}
		lessonFile, ok := modelCfg.Lessons[lesson]
		if !ok {
			lessonFile = fmt.Sprintf("lesson-%d.jsonl", lesson)
		}
		outputPath = filepath.Join(modelCfg.Training, lessonFile)
	}

	// Load kernel.
	kernel, err := os.ReadFile(modelCfg.Kernel)
	if err != nil {
		log.Fatalf("read kernel: %v", err)
	}
	log.Printf("kernel: %d chars from %s", len(kernel), modelCfg.Kernel)

	// Load signature (LEK-1-Sig).
	var sig string
	if modelCfg.Signature != "" {
		sigBytes, err := os.ReadFile(modelCfg.Signature)
		if err != nil {
			log.Fatalf("read signature: %v", err)
		}
		sig = strings.TrimSpace(string(sigBytes))
		log.Printf("signature: %d chars from %s", len(sig), modelCfg.Signature)
	}

	// Dry run.
	if *dryRun {
		fmt.Printf("Model:    %s (%s)\n", modelCfg.Name, modelCfg.Paths.Base)
		fmt.Printf("Backend:  %s\n", aiCfg.Backend)
		fmt.Printf("Probes:   %d\n", len(probes))
		fmt.Printf("Runs:     %d per probe (%d total generations)\n", *runs, len(probes)**runs)
		fmt.Printf("Gate:     grammar v3 composite >= %.1f\n", *minScore)
		fmt.Printf("Generate: temp=%.2f max_tokens=%d top_p=%.2f\n",
			genCfg.Temperature, genCfg.MaxTokens, genCfg.TopP)
		fmt.Printf("Memory:   cache=%dGB limit=%dGB\n", cacheLimitGB, memLimitGB)
		fmt.Printf("Output:   %s\n", outputPath)
		fmt.Println()
		for i, p := range probes {
			if i >= 10 {
				fmt.Printf("  ... and %d more\n", len(probes)-10)
				break
			}
			prompt := p.Prompt
			if len(prompt) > 80 {
				prompt = prompt[:80] + "..."
			}
			fmt.Printf("  %s: %s\n", p.ID, prompt)
		}
		return
	}

	// Load model via go-inference.
	log.Printf("loading model: %s", modelCfg.Paths.Base)
	model, err := inference.LoadModel(modelCfg.Paths.Base)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()

	info := model.Info()
	log.Printf("model loaded: %s %d-layer", info.Architecture, info.NumLayers)

	// Initialise grammar scorer.
	tok := reversal.NewTokeniser()

	// Open output for append.
	out, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("open output: %v", err)
	}
	defer out.Close()

	kept := 0
	skipped := 0
	totalStart := time.Now()
	ctx := context.Background()
	kernelStr := strings.TrimSpace(string(kernel))

	for i, probe := range probes {
		var best *distillCandidate

		// Build sandwich prompt for output: LEK-1 + Prompt + LEK-1-Sig
		sandwichPrompt := kernelStr + "\n\n" + probe.Prompt
		if sig != "" {
			sandwichPrompt += "\n\n" + sig
		}

		for run := range *runs {
			fmt.Fprintf(os.Stderr, "  [%d/%d] %s run %d/%d",
				i+1, len(probes), probe.ID, run+1, *runs)

			// Inference uses bare probe — the model generates from its weights.
			// Sandwich wrapping is only for the training output format.
			messages := []inference.Message{
				{Role: "user", Content: probe.Prompt},
			}

			// Generate via native Metal inference.
			start := time.Now()
			var sb strings.Builder
			for token := range model.Chat(ctx, messages,
				inference.WithMaxTokens(genCfg.MaxTokens),
				inference.WithTemperature(float32(genCfg.Temperature)),
				inference.WithTopP(float32(genCfg.TopP)),
				inference.WithTopK(genCfg.TopK),
				inference.WithRepeatPenalty(float32(genCfg.RepeatPenalty)),
			) {
				sb.WriteString(token.Text)
			}
			if err := model.Err(); err != nil {
				fmt.Fprintf(os.Stderr, " → ERROR: %v\n", err)
				continue
			}
			response := sb.String()
			elapsed := time.Since(start)

			// Quick reject: empty/degenerate.
			if len(strings.TrimSpace(response)) < aiCfg.Distill.MinChars {
				fmt.Fprintf(os.Stderr, " → %d chars, EMPTY, %.1fs\n", len(response), elapsed.Seconds())
				continue
			}

			// Score with go-i18n/reversal.
			grammar := ScoreResponse(tok, response)
			delta := ComputeDelta(tok, probe.Prompt, response,
				aiCfg.Scorer.SycophancyEcho, aiCfg.Scorer.SycophancyUplift)

			met := model.Metrics()
			fmt.Fprintf(os.Stderr, " → %d chars, g=%.1f up=%+.1f echo=%.2f enr=%+.1f, %.1fs (%.0f tok/s)\n",
				len(response), grammar.Composite,
				delta.Uplift, delta.Echo, delta.Enrichment,
				elapsed.Seconds(), met.DecodeTokensPerSec)

			candidate := &distillCandidate{
				Response: response,
				Grammar:  grammar,
				Delta:    delta,
				Elapsed:  elapsed,
			}

			if best == nil || grammar.Composite > best.Grammar.Composite {
				best = candidate
			}
		}

		// Quality gate.
		if best != nil && best.Grammar.Composite >= *minScore {
			// Save with sandwich prompt — kernel wraps the bare probe for training.
			example := TrainingExample{
				Messages: []ChatMessage{
					{Role: "user", Content: sandwichPrompt},
					{Role: "assistant", Content: best.Response},
				},
			}
			line, _ := json.Marshal(example)
			out.Write(append(line, '\n'))

			kept++
			fmt.Fprintf(os.Stderr, "  ✓ KEPT %s (g=%.1f, verbs=%d, nouns=%d, enr=%+.1f)\n",
				probe.ID, best.Grammar.Composite,
				best.Grammar.VerbDiversity, best.Grammar.NounDiversity,
				best.Delta.Enrichment)
		} else {
			skipped++
			score := 0.0
			if best != nil {
				score = best.Grammar.Composite
			}
			fmt.Fprintf(os.Stderr, "  ✗ SKIP %s (best g=%.1f < %.1f)\n",
				probe.ID, score, *minScore)
		}
	}

	duration := time.Since(totalStart)

	fmt.Fprintf(os.Stderr, "\n=== Distillation Complete ===\n")
	fmt.Fprintf(os.Stderr, "Model:    %s (%s)\n", modelCfg.Name, info.Architecture)
	fmt.Fprintf(os.Stderr, "Probes:   %d\n", len(probes))
	fmt.Fprintf(os.Stderr, "Runs:     %d per probe (%d total generations)\n", *runs, len(probes)**runs)
	fmt.Fprintf(os.Stderr, "Scorer:   go-i18n/reversal grammar v3, gate >= %.1f\n", *minScore)
	fmt.Fprintf(os.Stderr, "Kept:     %d\n", kept)
	fmt.Fprintf(os.Stderr, "Skipped:  %d\n", skipped)
	if kept+skipped > 0 {
		fmt.Fprintf(os.Stderr, "Pass rate: %.0f%%\n", float64(kept)/float64(kept+skipped)*100)
	}
	fmt.Fprintf(os.Stderr, "Output:   %s\n", outputPath)
	fmt.Fprintf(os.Stderr, "Duration: %.0fs (%.1fm)\n", duration.Seconds(), duration.Minutes())
}

// loadDistillProbes loads probes from a named set or a file path.
// Returns the probes and the default phase number for output routing.
func loadDistillProbes(root, spec string) ([]DistillProbe, int, error) {
	// Try as a probe set name first.
	probesCfg, cfgErr := LoadProbesConfig(root)
	if cfgErr == nil {
		if set, ok := probesCfg.Sets[spec]; ok {
			phase := 0
			if set.Phase != nil {
				phase = *set.Phase
			}
			var probes []DistillProbe
			for _, f := range set.Files {
				// Files are relative to the training root.
				ps, err := readProbeFile(filepath.Join(root, "training", "lem", f))
				if err != nil {
					return nil, 0, fmt.Errorf("read %s: %w", f, err)
				}
				probes = append(probes, ps...)
			}
			return probes, phase, nil
		}
	}

	// Fall back to direct file path.
	probes, err := readProbeFile(spec)
	if err != nil {
		return nil, 0, err
	}
	return probes, 0, nil
}

// readProbeFile reads probes from a JSON array file.
func readProbeFile(path string) ([]DistillProbe, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var raw []struct {
		ID     string `json:"id"`
		Domain string `json:"domain"`
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse %s: %w", filepath.Base(path), err)
	}

	probes := make([]DistillProbe, len(raw))
	for i, r := range raw {
		probes[i] = DistillProbe{
			ID:     r.ID,
			Domain: r.Domain,
			Prompt: r.Prompt,
			Source: filepath.Base(path),
		}
	}
	return probes, nil
}
