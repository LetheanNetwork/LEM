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
