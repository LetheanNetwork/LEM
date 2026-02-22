package lemcmd

import (
	"fmt"

	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addScoreCommands(root *cli.Command) {
	scoreGroup := cli.NewGroup("score", "Scoring commands", "Score responses, probe models, compare results.")

	scoreGroup.AddCommand(passthrough("run", "Score existing response files", lem.RunScore))
	scoreGroup.AddCommand(passthrough("probe", "Generate responses and score them", lem.RunProbe))

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

	scoreGroup.AddCommand(passthrough("tier", "Score expansion responses (heuristic/judge tiers)", lem.RunTierScore))
	scoreGroup.AddCommand(passthrough("agent", "ROCm scoring daemon (polls M3, scores checkpoints)", lem.RunAgent))

	root.AddCommand(scoreGroup)
}
