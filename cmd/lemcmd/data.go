package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addDataCommands(root *cli.Command) {
	dataGroup := cli.NewGroup("data", "Data management commands", "Import, consolidate, normalise, and approve training data.")

	dataGroup.AddCommand(passthrough("import-all", "Import ALL LEM data into DuckDB from M3", lem.RunImport))
	dataGroup.AddCommand(passthrough("consolidate", "Pull worker JSONLs from M3, merge, deduplicate", lem.RunConsolidate))
	dataGroup.AddCommand(passthrough("normalize", "Normalise seeds to deduplicated expansion prompts", lem.RunNormalize))
	dataGroup.AddCommand(passthrough("approve", "Filter scored expansions to training JSONL", lem.RunApprove))

	root.AddCommand(dataGroup)
}
