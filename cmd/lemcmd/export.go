package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addExportCommands(root *cli.Command) {
	exportGroup := cli.NewGroup("export", "Export and publish commands", "Export training data to JSONL, Parquet, HuggingFace, and PEFT formats.")

	exportGroup.AddCommand(passthrough("jsonl", "Export golden set to training-format JSONL splits", lem.RunExport))
	exportGroup.AddCommand(passthrough("parquet", "Export JSONL training splits to Parquet", lem.RunParquet))
	exportGroup.AddCommand(passthrough("publish", "Push Parquet files to HuggingFace dataset repo", lem.RunPublish))
	exportGroup.AddCommand(passthrough("convert", "Convert MLX LoRA adapter to PEFT format", lem.RunConvert))

	root.AddCommand(exportGroup)
}
