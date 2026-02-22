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
