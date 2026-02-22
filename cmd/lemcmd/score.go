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
