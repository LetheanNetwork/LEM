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
