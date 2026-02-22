package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addGenCommands(root *cli.Command) {
	genGroup := cli.NewGroup("gen", "Generation commands", "Distill, expand, and generate training data.")

	genGroup.AddCommand(passthrough("distill", "Native Metal distillation (go-mlx + grammar scoring)", lem.RunDistill))
	genGroup.AddCommand(passthrough("expand", "Generate expansion responses via trained LEM model", lem.RunExpand))
	genGroup.AddCommand(passthrough("conv", "Generate conversational training data (calm phase)", lem.RunConv))

	root.AddCommand(genGroup)
}
