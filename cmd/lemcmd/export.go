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
