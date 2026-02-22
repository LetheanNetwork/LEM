# LEM CLI Migration Design

Date: 2026-02-22
Status: Approved

## Problem

LEM's `main.go` is a 296-line manual `switch os.Args[1]` with `flag.FlagSet` per command. No signal handling, no shell completion, no grouped help, no framework lifecycle. The Core Go Framework provides `pkg/cli` — a full CLI SDK wrapping cobra, charmbracelet TUI, and the DI lifecycle. Every other domain repo in the fleet uses it.

## Solution

Replace `main.go` with `cli.Main()` + `cli.WithCommands()`. Commands register through the Core framework lifecycle. LEM gets signal handling, structured logging, shell completion, grouped help, TUI primitives (Spinner, ProgressBar, Viewport), and workspace support for free.

### Single import rule

LEM imports `forge.lthn.ai/core/go/pkg/cli` and **nothing else** for CLI concerns. No cobra, no lipgloss, no bubbletea. `pkg/cli` wraps everything.

### New main.go (~10 lines)

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

### Command Groups (6 groups, 28 commands)

```
lem score [score|probe|compare|tier-score|agent]         — Scoring
lem gen [distill|expand|conv]                            — Generation
lem data [import-all|consolidate|normalize|approve]       — Data Management
lem export [jsonl|parquet|publish|convert]                — Export & Publish
lem mon [status|expand-status|inventory|coverage|metrics] — Monitoring
lem infra [ingest|seed-influx|query|worker]              — Infrastructure
```

### File Layout

```
cmd/lemcmd/
├── lem.go          # AddLEMCommands — creates groups, registers all
├── score.go        # score, probe, compare, tier-score, agent
├── gen.go          # distill, expand, conv
├── data.go         # import-all, consolidate, normalize, approve
├── export.go       # export (renamed jsonl), parquet, publish, convert
├── mon.go          # status, expand-status, inventory, coverage, metrics
└── infra.go        # ingest, seed-influx, query, worker
```

### Registration Pattern

Following the fleet pattern (go-ml, go-devops, cli/):

```go
// cmd/lemcmd/lem.go
package lemcmd

import "forge.lthn.ai/core/go/pkg/cli"

func AddLEMCommands(root *cli.Command) {
    addScoreCommands(root)
    addGenCommands(root)
    addDataCommands(root)
    addExportCommands(root)
    addMonCommands(root)
    addInfraCommands(root)
}
```

Each group file:

```go
// cmd/lemcmd/gen.go
package lemcmd

import "forge.lthn.ai/core/go/pkg/cli"

func addGenCommands(root *cli.Command) {
    genCmd := cli.NewGroup("gen", "Generation commands", "")

    distillCmd := cli.NewCommand("distill", "Native Metal distillation", "", runDistill)
    // flags via cli.StringFlag, cli.IntFlag, etc.

    genCmd.AddCommand(distillCmd)
    root.AddCommand(genCmd)
}
```

### Phase 1: Pass-through to existing RunFoo functions

Each `RunE` handler builds an `[]string` args slice from cobra flags and calls the existing `lem.RunFoo(args)` function. No business logic changes. This keeps the migration purely structural.

### Phase 2 (future): Native cobra flags

Migrate individual commands to use cobra flags directly instead of rebuilding `[]string`. This is optional and can be done command-by-command over time.

### What changes

- `main.go` shrinks from 296 lines to ~10 lines
- `runScore()` and `runProbe()` (currently in main.go) move to `cmd/lemcmd/score.go`
- `core/go` added as a full dependency (DI, lifecycle, signals, logging, workspace)
- Each command gets proper `--help`, shell completion, grouped help output

### What stays the same

- All `pkg/lem/Run*` functions — unchanged
- All business logic in `pkg/lem/` — untouched
- Config loading, probe loading, scoring — unchanged

### Dependencies

- `forge.lthn.ai/core/go` (already in replace block, needs adding to require)
- Transitively pulls in cobra, charmbracelet — but LEM never imports them directly
