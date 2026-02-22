package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addInfraCommands(root *cli.Command) {
	infraGroup := cli.NewGroup("infra", "Infrastructure commands", "InfluxDB ingestion, DuckDB queries, and distributed workers.")

	infraGroup.AddCommand(cli.NewRun("ingest", "Ingest benchmark data into InfluxDB", "", func(cmd *cli.Command, args []string) {
		lem.RunIngest(args)
	}))

	infraGroup.AddCommand(cli.NewRun("seed-influx", "Seed InfluxDB golden_gen from DuckDB", "", func(cmd *cli.Command, args []string) {
		lem.RunSeedInflux(args)
	}))

	infraGroup.AddCommand(cli.NewRun("query", "Run ad-hoc SQL against DuckDB", "", func(cmd *cli.Command, args []string) {
		lem.RunQuery(args)
	}))

	infraGroup.AddCommand(cli.NewRun("worker", "Run as distributed inference worker node", "", func(cmd *cli.Command, args []string) {
		lem.RunWorker(args)
	}))

	root.AddCommand(infraGroup)
}
