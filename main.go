package main

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/cmd/lemcmd"
)

func main() {
	cli.Main(
		cli.WithCommands("lem", lemcmd.AddLEMCommands),
	)
}
