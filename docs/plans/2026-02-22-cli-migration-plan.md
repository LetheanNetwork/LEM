# LEM CLI Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LEM's manual 28-case `switch os.Args[1]` with the Core framework's `cli.Main()` + `cli.WithCommands()` pattern, grouping commands into 6 categories.

**Architecture:** `main.go` calls `cli.Main(cli.WithCommands("lem", lemcmd.AddLEMCommands))`. The `cmd/lemcmd/` package creates 6 command groups (score, gen, data, export, mon, infra) with cobra commands that pass through to existing `lem.Run*()` functions. Business logic stays in `pkg/lem/` untouched.

**Tech Stack:** `forge.lthn.ai/core/go/pkg/cli` (wraps cobra, charmbracelet TUI, Core DI lifecycle)

**Design doc:** `docs/plans/2026-02-22-cli-migration-design.md`

---

### Task 1: Add core/go to go.mod

`core/go` is in the `replace` block but not in the `require` block. The compiler needs it to import `pkg/cli`.

**Files:**
- Modify: `go.mod`

**Step 1: Add core/go to require block**

Add this line to the first `require` block in `go.mod`, before `go-i18n`:

```
forge.lthn.ai/core/go v0.0.0-00010101000000-000000000000
```

**Step 2: Run go mod tidy**

Run: `cd /Users/snider/Code/LEM && go mod tidy`

**Step 3: Verify build**

Run: `cd /Users/snider/Code/LEM && go build ./...`
Expected: Clean build

**Step 4: Commit**

```bash
cd /Users/snider/Code/LEM
git add go.mod go.sum
git commit -m "$(cat <<'EOF'
chore: add core/go to go.mod require block

Prerequisite for CLI migration to core/go pkg/cli framework.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Move runScore, runProbe, runCompare to pkg/lem

Three commands currently live in `main.go` instead of `pkg/lem/`. Move them so all 28 commands are accessible from `pkg/lem/` before wiring up the new CLI.

**Files:**
- Create: `pkg/lem/score_cmd.go`
- Modify: `main.go` (remove the three functions)

**Step 1: Create pkg/lem/score_cmd.go**

Create `/Users/snider/Code/LEM/pkg/lem/score_cmd.go` with the three functions moved from `main.go`. Rename them to exported names and adjust imports:

```go
package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

// RunScore scores existing response files using LLM judges.
func RunScore(args []string) {
	fs := flag.NewFlagSet("score", flag.ExitOnError)

	input := fs.String("input", "", "Input JSONL response file (required)")
	suites := fs.String("suites", "all", "Comma-separated suites or 'all'")
	judgeModel := fs.String("judge-model", "mlx-community/gemma-3-27b-it-qat-4bit", "Judge model name")
	judgeURL := fs.String("judge-url", "http://10.69.69.108:8090", "Judge API URL")
	concurrency := fs.Int("concurrency", 4, "Max concurrent judge calls")
	output := fs.String("output", "scores.json", "Output score file path")
	resume := fs.Bool("resume", false, "Resume from existing output, skipping scored IDs")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *input == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required")
		fs.Usage()
		os.Exit(1)
	}

	responses, err := ReadResponses(*input)
	if err != nil {
		log.Fatalf("read responses: %v", err)
	}
	log.Printf("loaded %d responses from %s", len(responses), *input)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, readErr := ReadScorerOutput(*output)
			if readErr != nil {
				log.Fatalf("read existing scores for resume: %v", readErr)
			}

			scored := make(map[string]bool)
			for _, scores := range existing.PerPrompt {
				for _, ps := range scores {
					scored[ps.ID] = true
				}
			}

			var filtered []Response
			for _, r := range responses {
				if !scored[r.ID] {
					filtered = append(filtered, r)
				}
			}
			log.Printf("resume: skipping %d already-scored, %d remaining",
				len(responses)-len(filtered), len(filtered))
			responses = filtered

			if len(responses) == 0 {
				log.Println("all responses already scored, nothing to do")
				return
			}
		}
	}

	client := NewClient(*judgeURL, *judgeModel)
	client.MaxTokens = 512
	judge := NewJudge(client)
	engine := NewEngine(judge, *concurrency, *suites)

	log.Printf("scoring with %s", engine)

	perPrompt := engine.ScoreAll(responses)

	if *resume {
		if _, statErr := os.Stat(*output); statErr == nil {
			existing, _ := ReadScorerOutput(*output)
			for model, scores := range existing.PerPrompt {
				perPrompt[model] = append(scores, perPrompt[model]...)
			}
		}
	}

	averages := ComputeAverages(perPrompt)

	scorerOutput := &ScorerOutput{
		Metadata: Metadata{
			JudgeModel:    *judgeModel,
			JudgeURL:      *judgeURL,
			ScoredAt:      time.Now().UTC(),
			ScorerVersion: "1.0.0",
			Suites:        engine.SuiteNames(),
		},
		ModelAverages: averages,
		PerPrompt:     perPrompt,
	}

	if err := WriteScores(*output, scorerOutput); err != nil {
		log.Fatalf("write scores: %v", err)
	}

	log.Printf("wrote scores to %s", *output)
}

// RunProbe generates responses and scores them.
func RunProbe(args []string) {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)

	model := fs.String("model", "", "Target model name (required)")
	targetURL := fs.String("target-url", "", "Target model API URL (defaults to judge-url)")
	probesFile := fs.String("probes", "", "Custom probes JSONL file (uses built-in content probes if not specified)")
	suites := fs.String("suites", "all", "Comma-separated suites or 'all'")
	judgeModel := fs.String("judge-model", "mlx-community/gemma-3-27b-it-qat-4bit", "Judge model name")
	judgeURL := fs.String("judge-url", "http://10.69.69.108:8090", "Judge API URL")
	concurrency := fs.Int("concurrency", 4, "Max concurrent judge calls")
	output := fs.String("output", "scores.json", "Output score file path")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	if *targetURL == "" {
		*targetURL = *judgeURL
	}

	targetClient := NewClient(*targetURL, *model)
	targetClient.MaxTokens = 1024
	judgeClient := NewClient(*judgeURL, *judgeModel)
	judgeClient.MaxTokens = 512
	judge := NewJudge(judgeClient)
	engine := NewEngine(judge, *concurrency, *suites)
	prober := NewProber(targetClient, engine)

	var scorerOutput *ScorerOutput
	var err error

	if *probesFile != "" {
		probes, readErr := ReadResponses(*probesFile)
		if readErr != nil {
			log.Fatalf("read probes: %v", readErr)
		}
		log.Printf("loaded %d custom probes from %s", len(probes), *probesFile)

		scorerOutput, err = prober.ProbeModel(probes, *model)
	} else {
		log.Printf("using %d built-in content probes", len(ContentProbes))
		scorerOutput, err = prober.ProbeContent(*model)
	}

	if err != nil {
		log.Fatalf("probe: %v", err)
	}

	if writeErr := WriteScores(*output, scorerOutput); writeErr != nil {
		log.Fatalf("write scores: %v", writeErr)
	}

	log.Printf("wrote scores to %s", *output)
}
```

Note: `RunCompare` already exists in `pkg/lem/compare.go` with signature `RunCompare(oldPath, newPath string) error`. No need to move it — the new CLI wrapper will handle arg parsing.

**Step 2: Update main.go to use the new exported functions**

In `main.go`, replace:
- `runScore(os.Args[2:])` → `lem.RunScore(os.Args[2:])`
- `runProbe(os.Args[2:])` → `lem.RunProbe(os.Args[2:])`

Remove the `runScore`, `runProbe`, and `runCompare` functions from `main.go`. For `compare`, change the switch case to call through:
```go
	case "compare":
		fs := flag.NewFlagSet("compare", flag.ExitOnError)
		oldFile := fs.String("old", "", "Old score file (required)")
		newFile := fs.String("new", "", "New score file (required)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("parse flags: %v", err)
		}
		if *oldFile == "" || *newFile == "" {
			fmt.Fprintln(os.Stderr, "error: --old and --new are required")
			fs.Usage()
			os.Exit(1)
		}
		if err := lem.RunCompare(*oldFile, *newFile); err != nil {
			log.Fatalf("compare: %v", err)
		}
```

Actually simpler: leave `main.go`'s compare case inline since we're about to replace the whole file anyway. The key change is moving `runScore` and `runProbe` to `pkg/lem/` and removing them from `main.go`.

The new `main.go` (with functions removed but switch intact):

```go
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"forge.lthn.ai/lthn/lem/pkg/lem"
)

const usage = `Usage: lem <command> [flags]
...existing usage string...
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "distill":
		lem.RunDistill(os.Args[2:])
	case "score":
		lem.RunScore(os.Args[2:])
	case "probe":
		lem.RunProbe(os.Args[2:])
	case "compare":
		fs := flag.NewFlagSet("compare", flag.ExitOnError)
		oldFile := fs.String("old", "", "Old score file (required)")
		newFile := fs.String("new", "", "New score file (required)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("parse flags: %v", err)
		}
		if *oldFile == "" || *newFile == "" {
			fmt.Fprintln(os.Stderr, "error: --old and --new are required")
			fs.Usage()
			os.Exit(1)
		}
		if err := lem.RunCompare(*oldFile, *newFile); err != nil {
			log.Fatalf("compare: %v", err)
		}
	case "status":
		lem.RunStatus(os.Args[2:])
	// ... rest of switch cases unchanged ...
	case "worker":
		lem.RunWorker(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n%s", os.Args[1], usage)
		os.Exit(1)
	}
}
```

Remove `"time"` from imports (only needed by the moved `runScore`).

**Step 3: Verify build**

Run: `cd /Users/snider/Code/LEM && go build ./...`
Expected: Clean build

**Step 4: Commit**

```bash
cd /Users/snider/Code/LEM
git add pkg/lem/score_cmd.go main.go
git commit -m "$(cat <<'EOF'
refactor: move runScore and runProbe to pkg/lem

All 28 commands now accessible as exported lem.Run* functions.
Prerequisite for CLI framework migration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create cmd/lemcmd/lem.go — root registration

Create the root registration file that `main.go` will call.

**Files:**
- Create: `cmd/lemcmd/lem.go`

**Step 1: Create the root file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/lem.go`:

```go
// Package lemcmd provides CLI commands for the LEM binary.
// Commands register through the Core framework's cli.WithCommands lifecycle.
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
)

// AddLEMCommands registers all LEM command groups on the root command.
func AddLEMCommands(root *cli.Command) {
	addScoreCommands(root)
	addGenCommands(root)
	addDataCommands(root)
	addExportCommands(root)
	addMonCommands(root)
	addInfraCommands(root)
}
```

This won't compile yet (the `add*Commands` functions don't exist). That's fine — we'll add them in Tasks 4-9.

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/lem.go
git commit -m "$(cat <<'EOF'
feat(cli): add root command registration for LEM

AddLEMCommands wires 6 command groups through cli.WithCommands.
Group implementations follow in subsequent commits.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create cmd/lemcmd/score.go — Scoring group

5 commands: score, probe, compare, tier-score, agent

**Files:**
- Create: `cmd/lemcmd/score.go`

**Step 1: Create the score commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/score.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addScoreCommands(root *cli.Command) {
	scoreGroup := cli.NewGroup("score", "Scoring commands", "Score responses, probe models, compare results.")

	scoreGroup.AddCommand(cli.NewRun("run", "Score existing response files", "", func(cmd *cli.Command, args []string) {
		lem.RunScore(args)
	}))

	scoreGroup.AddCommand(cli.NewRun("probe", "Generate responses and score them", "", func(cmd *cli.Command, args []string) {
		lem.RunProbe(args)
	}))

	scoreGroup.AddCommand(cli.NewCommand("compare", "Compare two score files", "", func(cmd *cli.Command, args []string) error {
		var oldFile, newFile string
		cli.StringFlag(cmd, &oldFile, "old", "", "", "Old score file (required)")
		cli.StringFlag(cmd, &newFile, "new", "", "", "New score file (required)")
		// Flags are parsed by cobra before RunE is called.
		// But since we declared flags on the cmd, they're already available.
		return lem.RunCompare(oldFile, newFile)
	}))

	scoreGroup.AddCommand(cli.NewRun("tier", "Score expansion responses (heuristic/judge tiers)", "", func(cmd *cli.Command, args []string) {
		lem.RunTierScore(args)
	}))

	scoreGroup.AddCommand(cli.NewRun("agent", "ROCm scoring daemon (polls M3, scores checkpoints)", "", func(cmd *cli.Command, args []string) {
		lem.RunAgent(args)
	}))

	root.AddCommand(scoreGroup)
}
```

Wait — there's a subtlety with `compare`. `RunCompare` takes `(oldPath, newPath string) error`, not `[]string`. The flags need to be declared on the cobra command BEFORE RunE runs. Let me fix that:

```go
package lemcmd

import (
	"fmt"

	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addScoreCommands(root *cli.Command) {
	scoreGroup := cli.NewGroup("score", "Scoring commands", "Score responses, probe models, compare results.")

	scoreGroup.AddCommand(cli.NewRun("run", "Score existing response files", "", func(cmd *cli.Command, args []string) {
		lem.RunScore(args)
	}))

	scoreGroup.AddCommand(cli.NewRun("probe", "Generate responses and score them", "", func(cmd *cli.Command, args []string) {
		lem.RunProbe(args)
	}))

	// compare has a different signature — it takes two named args, not []string.
	compareCmd := cli.NewCommand("compare", "Compare two score files", "", nil)
	var compareOld, compareNew string
	cli.StringFlag(compareCmd, &compareOld, "old", "", "", "Old score file (required)")
	cli.StringFlag(compareCmd, &compareNew, "new", "", "", "New score file (required)")
	compareCmd.RunE = func(cmd *cli.Command, args []string) error {
		if compareOld == "" || compareNew == "" {
			return fmt.Errorf("--old and --new are required")
		}
		return lem.RunCompare(compareOld, compareNew)
	}
	scoreGroup.AddCommand(compareCmd)

	scoreGroup.AddCommand(cli.NewRun("tier", "Score expansion responses (heuristic/judge tiers)", "", func(cmd *cli.Command, args []string) {
		lem.RunTierScore(args)
	}))

	scoreGroup.AddCommand(cli.NewRun("agent", "ROCm scoring daemon (polls M3, scores checkpoints)", "", func(cmd *cli.Command, args []string) {
		lem.RunAgent(args)
	}))

	root.AddCommand(scoreGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/score.go
git commit -m "$(cat <<'EOF'
feat(cli): add score command group

lem score [run|probe|compare|tier|agent]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create cmd/lemcmd/gen.go — Generation group

3 commands: distill, expand, conv

**Files:**
- Create: `cmd/lemcmd/gen.go`

**Step 1: Create the gen commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/gen.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addGenCommands(root *cli.Command) {
	genGroup := cli.NewGroup("gen", "Generation commands", "Distill, expand, and generate training data.")

	genGroup.AddCommand(cli.NewRun("distill", "Native Metal distillation (go-mlx + grammar scoring)", "", func(cmd *cli.Command, args []string) {
		lem.RunDistill(args)
	}))

	genGroup.AddCommand(cli.NewRun("expand", "Generate expansion responses via trained LEM model", "", func(cmd *cli.Command, args []string) {
		lem.RunExpand(args)
	}))

	genGroup.AddCommand(cli.NewRun("conv", "Generate conversational training data (calm phase)", "", func(cmd *cli.Command, args []string) {
		lem.RunConv(args)
	}))

	root.AddCommand(genGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/gen.go
git commit -m "$(cat <<'EOF'
feat(cli): add gen command group

lem gen [distill|expand|conv]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Create cmd/lemcmd/data.go — Data Management group

4 commands: import-all, consolidate, normalize, approve

**Files:**
- Create: `cmd/lemcmd/data.go`

**Step 1: Create the data commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/data.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addDataCommands(root *cli.Command) {
	dataGroup := cli.NewGroup("data", "Data management commands", "Import, consolidate, normalise, and approve training data.")

	dataGroup.AddCommand(cli.NewRun("import-all", "Import ALL LEM data into DuckDB from M3", "", func(cmd *cli.Command, args []string) {
		lem.RunImport(args)
	}))

	dataGroup.AddCommand(cli.NewRun("consolidate", "Pull worker JSONLs from M3, merge, deduplicate", "", func(cmd *cli.Command, args []string) {
		lem.RunConsolidate(args)
	}))

	dataGroup.AddCommand(cli.NewRun("normalize", "Normalise seeds to deduplicated expansion prompts", "", func(cmd *cli.Command, args []string) {
		lem.RunNormalize(args)
	}))

	dataGroup.AddCommand(cli.NewRun("approve", "Filter scored expansions to training JSONL", "", func(cmd *cli.Command, args []string) {
		lem.RunApprove(args)
	}))

	root.AddCommand(dataGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/data.go
git commit -m "$(cat <<'EOF'
feat(cli): add data command group

lem data [import-all|consolidate|normalize|approve]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Create cmd/lemcmd/export.go — Export & Publish group

4 commands: jsonl (was "export"), parquet, publish, convert

**Files:**
- Create: `cmd/lemcmd/export.go`

**Step 1: Create the export commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/export.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addExportCommands(root *cli.Command) {
	exportGroup := cli.NewGroup("export", "Export and publish commands", "Export training data to JSONL, Parquet, HuggingFace, and PEFT formats.")

	exportGroup.AddCommand(cli.NewRun("jsonl", "Export golden set to training-format JSONL splits", "", func(cmd *cli.Command, args []string) {
		lem.RunExport(args)
	}))

	exportGroup.AddCommand(cli.NewRun("parquet", "Export JSONL training splits to Parquet", "", func(cmd *cli.Command, args []string) {
		lem.RunParquet(args)
	}))

	exportGroup.AddCommand(cli.NewRun("publish", "Push Parquet files to HuggingFace dataset repo", "", func(cmd *cli.Command, args []string) {
		lem.RunPublish(args)
	}))

	exportGroup.AddCommand(cli.NewRun("convert", "Convert MLX LoRA adapter to PEFT format", "", func(cmd *cli.Command, args []string) {
		lem.RunConvert(args)
	}))

	root.AddCommand(exportGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/export.go
git commit -m "$(cat <<'EOF'
feat(cli): add export command group

lem export [jsonl|parquet|publish|convert]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Create cmd/lemcmd/mon.go — Monitoring group

5 commands: status, expand-status, inventory, coverage, metrics

**Files:**
- Create: `cmd/lemcmd/mon.go`

**Step 1: Create the monitoring commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/mon.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addMonCommands(root *cli.Command) {
	monGroup := cli.NewGroup("mon", "Monitoring commands", "Training progress, pipeline status, inventory, coverage, and metrics.")

	monGroup.AddCommand(cli.NewRun("status", "Show training and generation progress (InfluxDB)", "", func(cmd *cli.Command, args []string) {
		lem.RunStatus(args)
	}))

	monGroup.AddCommand(cli.NewRun("expand-status", "Show expansion pipeline status (DuckDB)", "", func(cmd *cli.Command, args []string) {
		lem.RunExpandStatus(args)
	}))

	monGroup.AddCommand(cli.NewRun("inventory", "Show DuckDB table inventory", "", func(cmd *cli.Command, args []string) {
		lem.RunInventory(args)
	}))

	monGroup.AddCommand(cli.NewRun("coverage", "Analyse seed coverage gaps", "", func(cmd *cli.Command, args []string) {
		lem.RunCoverage(args)
	}))

	monGroup.AddCommand(cli.NewRun("metrics", "Push DuckDB golden set stats to InfluxDB", "", func(cmd *cli.Command, args []string) {
		lem.RunMetrics(args)
	}))

	root.AddCommand(monGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/mon.go
git commit -m "$(cat <<'EOF'
feat(cli): add mon command group

lem mon [status|expand-status|inventory|coverage|metrics]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Create cmd/lemcmd/infra.go — Infrastructure group

4 commands: ingest, seed-influx, query, worker

**Files:**
- Create: `cmd/lemcmd/infra.go`

**Step 1: Create the infra commands file**

Create `/Users/snider/Code/LEM/cmd/lemcmd/infra.go`:

```go
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addInfraCommands(root *cli.Command) {
	infraGroup := cli.NewGroup("infra", "Infrastructure commands", "InfluxDB ingestion, DuckDB queries, and distributed workers.")

	infraGroup.AddCommand(cli.NewRun("ingest", "Ingest benchmark data into InfluxDB", "", func(cmd *cli.Command, args []string) {
		lem.RunIngest(args)
	}))

	infraGroup.AddCommand(cli.NewRun("seed-influx", "Seed InfluxDB golden_gen from DuckDB", "", func(cmd *cli.Command, args []string) {
		lem.RunSeedInflux(args)
	}))

	infraGroup.AddCommand(cli.NewRun("query", "Run ad-hoc SQL against DuckDB", "", func(cmd *cli.Command, args []string) {
		lem.RunQuery(args)
	}))

	infraGroup.AddCommand(cli.NewRun("worker", "Run as distributed inference worker node", "", func(cmd *cli.Command, args []string) {
		lem.RunWorker(args)
	}))

	root.AddCommand(infraGroup)
}
```

**Step 2: Commit**

```bash
cd /Users/snider/Code/LEM
git add cmd/lemcmd/infra.go
git commit -m "$(cat <<'EOF'
feat(cli): add infra command group

lem infra [ingest|seed-influx|query|worker]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Replace main.go with cli.Main

The final step: replace the entire `main.go` with the framework bootstrap.

**Files:**
- Modify: `main.go`

**Step 1: Replace main.go**

Replace the entire contents of `/Users/snider/Code/LEM/main.go` with:

```go
package main

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/cmd/lemcmd"
)

func main() {
	cli.Main(
		cli.WithCommands("lem", lemcmd.AddLEMCommands),
	)
}
```

**Step 2: Verify build**

Run: `cd /Users/snider/Code/LEM && go build ./...`
Expected: Clean build

**Step 3: Verify vet**

Run: `cd /Users/snider/Code/LEM && go vet ./...`
Expected: Clean

**Step 4: Commit**

```bash
cd /Users/snider/Code/LEM
git add main.go
git commit -m "$(cat <<'EOF'
feat(cli): replace manual switch with cli.Main + WithCommands

main.go shrinks from 296 lines to 11. All 28 commands register
through Core framework lifecycle via cli.WithCommands. Gets signal
handling, shell completion, grouped help, and TUI primitives.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Run go mod tidy and final verification

**Files:**
- Modify: `go.mod`, `go.sum`

**Step 1: Run go mod tidy**

Run: `cd /Users/snider/Code/LEM && go mod tidy`

**Step 2: Full build**

Run: `cd /Users/snider/Code/LEM && go build ./...`
Expected: Clean

**Step 3: Run go vet**

Run: `cd /Users/snider/Code/LEM && go vet ./...`
Expected: Clean

**Step 4: Smoke test — help output**

Run: `cd /Users/snider/Code/LEM && go run . --help`

Expected: Grouped command listing showing score, gen, data, export, mon, infra subgroups.

**Step 5: Smoke test — subcommand help**

Run: `cd /Users/snider/Code/LEM && go run . gen --help`

Expected: Lists distill, expand, conv subcommands with descriptions.

**Step 6: Smoke test — distill dry-run**

Run: `cd /Users/snider/Code/LEM && go run . gen distill -- --model gemma3/1b --probes core --dry-run`

Note: The `--` separator tells cobra to stop parsing flags and pass the rest as args to the `Run` handler. Since `RunDistill` does its own flag parsing from the `args []string`, the flags after `--` are passed through.

If cobra swallows the flags (because they're defined on the parent), try without `--`:
Run: `cd /Users/snider/Code/LEM && go run . gen distill --model gemma3/1b --probes core --dry-run`

Expected: The familiar dry-run output with Memory line.

**Step 7: Commit if go.mod/go.sum changed**

```bash
cd /Users/snider/Code/LEM
git add go.mod go.sum
git commit -m "$(cat <<'EOF'
chore: go mod tidy after CLI migration

core/go now a direct dependency for pkg/cli framework.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Command Mapping Reference

| Old command | New command | Handler |
|-------------|------------|---------|
| `lem score` | `lem score run` | `lem.RunScore(args)` |
| `lem probe` | `lem score probe` | `lem.RunProbe(args)` |
| `lem compare` | `lem score compare --old X --new Y` | `lem.RunCompare(old, new)` |
| `lem tier-score` | `lem score tier` | `lem.RunTierScore(args)` |
| `lem agent` | `lem score agent` | `lem.RunAgent(args)` |
| `lem distill` | `lem gen distill` | `lem.RunDistill(args)` |
| `lem expand` | `lem gen expand` | `lem.RunExpand(args)` |
| `lem conv` | `lem gen conv` | `lem.RunConv(args)` |
| `lem import-all` | `lem data import-all` | `lem.RunImport(args)` |
| `lem consolidate` | `lem data consolidate` | `lem.RunConsolidate(args)` |
| `lem normalize` | `lem data normalize` | `lem.RunNormalize(args)` |
| `lem approve` | `lem data approve` | `lem.RunApprove(args)` |
| `lem export` | `lem export jsonl` | `lem.RunExport(args)` |
| `lem parquet` | `lem export parquet` | `lem.RunParquet(args)` |
| `lem publish` | `lem export publish` | `lem.RunPublish(args)` |
| `lem convert` | `lem export convert` | `lem.RunConvert(args)` |
| `lem status` | `lem mon status` | `lem.RunStatus(args)` |
| `lem expand-status` | `lem mon expand-status` | `lem.RunExpandStatus(args)` |
| `lem inventory` | `lem mon inventory` | `lem.RunInventory(args)` |
| `lem coverage` | `lem mon coverage` | `lem.RunCoverage(args)` |
| `lem metrics` | `lem mon metrics` | `lem.RunMetrics(args)` |
| `lem ingest` | `lem infra ingest` | `lem.RunIngest(args)` |
| `lem seed-influx` | `lem infra seed-influx` | `lem.RunSeedInflux(args)` |
| `lem query` | `lem infra query` | `lem.RunQuery(args)` |
| `lem worker` | `lem infra worker` | `lem.RunWorker(args)` |

## What Stays the Same

- All `pkg/lem/Run*` functions — unchanged (they accept `[]string` and do their own flag parsing)
- All business logic — untouched
- Config loading, probe loading, scoring — unchanged
- `pkg/lem/backend_metal.go` — unchanged
