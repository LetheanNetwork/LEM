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
