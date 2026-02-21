package lem

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// AIConfig is the top-level .core/ai/ai.yaml configuration.
type AIConfig struct {
	Version  int            `yaml:"version"`
	Backend  string         `yaml:"backend"`
	Scorer   ScorerConfig   `yaml:"scorer"`
	Generate GenerateConfig `yaml:"generate"`
	Distill  DistillConfig  `yaml:"distill"`
}

// ScorerConfig controls quality gating.
type ScorerConfig struct {
	Engine           string  `yaml:"engine"`
	MinScore         float64 `yaml:"min_score"`
	Delta            bool    `yaml:"delta"`
	SycophancyEcho   float64 `yaml:"sycophancy_echo"`
	SycophancyUplift float64 `yaml:"sycophancy_uplift"`
}

// GenerateConfig holds default inference parameters.
type GenerateConfig struct {
	MaxTokens     int     `yaml:"max_tokens"`
	Temperature   float64 `yaml:"temperature"`
	TopP          float64 `yaml:"top_p"`
	TopK          int     `yaml:"top_k"`
	RepeatPenalty float64 `yaml:"repeat_penalty"`
}

// DistillConfig holds distillation defaults.
type DistillConfig struct {
	Runs     int `yaml:"runs"`
	MinChars int `yaml:"min_chars"`
}

// ModelConfig is a .core/ai/models/{family}/{size}.yaml file.
type ModelConfig struct {
	Version    int            `yaml:"version"`
	Name       string         `yaml:"name"`
	Family     string         `yaml:"family"`
	Parameters string         `yaml:"parameters"`
	Format     string         `yaml:"format"`
	Paths      ModelPaths     `yaml:"paths"`
	Kernel     string         `yaml:"kernel"`
	Training   string         `yaml:"training"`
	Lessons    map[int]string `yaml:"lessons"`
	Valid      string         `yaml:"valid"`
	Test       string         `yaml:"test"`
	Generate   GenerateConfig `yaml:"generate"`
	Baselines  Baselines      `yaml:"baselines"`
}

// ModelPaths holds filesystem locations for model files.
type ModelPaths struct {
	Base        string `yaml:"base"`
	Safetensors string `yaml:"safetensors"`
}

// Baselines holds scoring reference points.
type Baselines struct {
	NoKernel   float64 `yaml:"no_kernel"`
	WithKernel float64 `yaml:"with_kernel"`
	Target     float64 `yaml:"target"`
}

// ProbesConfig is a .core/ai/probes.yaml file.
type ProbesConfig struct {
	Version int                 `yaml:"version"`
	Sets    map[string]ProbeSet `yaml:"sets"`
}

// ProbeSet groups related probe files.
type ProbeSet struct {
	Description string   `yaml:"description"`
	Phase       *int     `yaml:"phase"`
	Files       []string `yaml:"files"`
}

// LoadAIConfig reads .core/ai/ai.yaml from the given root directory.
func LoadAIConfig(root string) (*AIConfig, error) {
	path := filepath.Join(root, ".core", "ai", "ai.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read ai config: %w", err)
	}
	var cfg AIConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse ai config: %w", err)
	}
	return &cfg, nil
}

// LoadModelConfig reads .core/ai/models/{model}.yaml.
// The model arg is a slash path like "gemma3/27b".
func LoadModelConfig(root, model string) (*ModelConfig, error) {
	path := filepath.Join(root, ".core", "ai", "models", model+".yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model config: %w", err)
	}
	var cfg ModelConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse model config: %w", err)
	}
	return &cfg, nil
}

// LoadProbesConfig reads .core/ai/probes.yaml.
func LoadProbesConfig(root string) (*ProbesConfig, error) {
	path := filepath.Join(root, ".core", "ai", "probes.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read probes config: %w", err)
	}
	var cfg ProbesConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse probes config: %w", err)
	}
	return &cfg, nil
}

// MergeGenerate returns a GenerateConfig with model-level overrides
// applied on top of the global defaults. Zero values in the model
// config are ignored (global default kept).
func MergeGenerate(global, model GenerateConfig) GenerateConfig {
	merged := global
	if model.MaxTokens > 0 {
		merged.MaxTokens = model.MaxTokens
	}
	if model.Temperature > 0 {
		merged.Temperature = model.Temperature
	}
	if model.TopP > 0 {
		merged.TopP = model.TopP
	}
	if model.TopK > 0 {
		merged.TopK = model.TopK
	}
	if model.RepeatPenalty > 0 {
		merged.RepeatPenalty = model.RepeatPenalty
	}
	return merged
}
