package lemcmd

import (
	"forge.lthn.ai/core/go/pkg/cli"
	"forge.lthn.ai/lthn/lem/pkg/lem"
)

func addInfraCommands(root *cli.Command) {
	infraGroup := cli.NewGroup("infra", "Infrastructure commands", "InfluxDB ingestion, DuckDB queries, and distributed workers.")

	infraGroup.AddCommand(passthrough("ingest", "Ingest benchmark data into InfluxDB", lem.RunIngest))
	infraGroup.AddCommand(passthrough("seed-influx", "Seed InfluxDB golden_gen from DuckDB", lem.RunSeedInflux))
	infraGroup.AddCommand(passthrough("query", "Run ad-hoc SQL against DuckDB", lem.RunQuery))
	infraGroup.AddCommand(passthrough("worker", "Run as distributed inference worker node", lem.RunWorker))

	root.AddCommand(infraGroup)
}
