// Package lemcmd provides CLI commands for the LEM binary.
// Commands register through the Core framework's cli.WithCommands lifecycle.
package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
)

// AddLEMCommands registers all LEM command groups on the root command.
func AddLEMCommands(root *cli.Command) {
	addScoreCommands(root)
	addGenCommands(root)
	addDataCommands(root)
	addExportCommands(root)
	addMonCommands(root)
	addInfraCommands(root)
}

// passthrough creates a command that passes all args (including flags) to fn.
// Used for commands that do their own flag parsing with flag.FlagSet.
func passthrough(use, short string, fn func([]string)) *cli.Command {
	cmd := cli.NewRun(use, short, "", func(_ *cli.Command, args []string) {
		fn(args)
	})
	cmd.DisableFlagParsing = true
	return cmd
}
