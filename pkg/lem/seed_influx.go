package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// RunSeedInflux is the CLI entry point for the seed-influx command.
// Seeds InfluxDB golden_gen measurement from DuckDB golden_set data.
// One-time migration tool for bootstrapping InfluxDB from existing data.
func RunSeedInflux(args []string) {
	fs := flag.NewFlagSet("seed-influx", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	influxURL := fs.String("influx", "", "InfluxDB URL")
	influxDB := fs.String("influx-db", "", "InfluxDB database name")
	force := fs.Bool("force", false, "Re-seed even if InfluxDB already has data")
	batchSize := fs.Int("batch-size", 500, "Lines per InfluxDB write batch")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}
	if *dbPath == "" {
		fmt.Fprintln(os.Stderr, "error: --db or LEM_DB required")
		os.Exit(1)
	}

	db, err := OpenDB(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	var total int
	if err := db.conn.QueryRow("SELECT count(*) FROM golden_set").Scan(&total); err != nil {
		log.Fatalf("No golden_set table. Run ingest first.")
	}

	influx := NewInfluxClient(*influxURL, *influxDB)

	// Check existing count in InfluxDB.
	existing := 0
	rows, err := influx.QuerySQL("SELECT count(DISTINCT i) AS n FROM gold_gen")
	if err == nil && len(rows) > 0 {
		if n, ok := rows[0]["n"].(float64); ok {
			existing = int(n)
		}
	}

	fmt.Printf("DuckDB has %d records, InfluxDB golden_gen has %d\n", total, existing)

	if existing >= total && !*force {
		fmt.Println("InfluxDB already has all records. Use --force to re-seed.")
		return
	}

	// Read all rows.
	dbRows, err := db.conn.Query(`
		SELECT idx, seed_id, domain, voice, gen_time, char_count
		FROM golden_set ORDER BY idx
	`)
	if err != nil {
		log.Fatalf("query golden_set: %v", err)
	}
	defer dbRows.Close()

	var lines []string
	written := 0

	for dbRows.Next() {
		var idx, charCount int
		var seedID, domain, voice string
		var genTime float64

		if err := dbRows.Scan(&idx, &seedID, &domain, &voice, &genTime, &charCount); err != nil {
			log.Fatalf("scan: %v", err)
		}

		sid := strings.ReplaceAll(seedID, `"`, `\"`)
		lp := fmt.Sprintf(`gold_gen,i=%d,w=migration,d=%s,v=%s seed_id="%s",gen_time=%.1f,chars=%di`,
			idx, escapeLp(domain), escapeLp(voice), sid, genTime, charCount)
		lines = append(lines, lp)

		if len(lines) >= *batchSize {
			if err := influx.WriteLp(lines); err != nil {
				log.Fatalf("write batch at %d: %v", written, err)
			}
			written += len(lines)
			lines = lines[:0]

			if written%2000 == 0 {
				fmt.Printf("  Seeded %d/%d records\n", written, total)
			}
		}
	}

	if len(lines) > 0 {
		if err := influx.WriteLp(lines); err != nil {
			log.Fatalf("flush: %v", err)
		}
		written += len(lines)
	}

	fmt.Printf("Seeded %d golden_gen records into InfluxDB\n", written)
}
