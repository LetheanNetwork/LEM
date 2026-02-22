package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addMonCommands(root *cli.Command) {
	monGroup := cli.NewGroup("mon", "Monitoring commands", "Training progress, pipeline status, inventory, coverage, and metrics.")

	monGroup.AddCommand(passthrough("status", "Show training and generation progress (InfluxDB)", lem.RunStatus))
	monGroup.AddCommand(passthrough("expand-status", "Show expansion pipeline status (DuckDB)", lem.RunExpandStatus))
	monGroup.AddCommand(passthrough("inventory", "Show DuckDB table inventory", lem.RunInventory))
	monGroup.AddCommand(passthrough("coverage", "Analyse seed coverage gaps", lem.RunCoverage))
	monGroup.AddCommand(passthrough("metrics", "Push DuckDB golden set stats to InfluxDB", lem.RunMetrics))

	root.AddCommand(monGroup)
}
