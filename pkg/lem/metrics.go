package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

const targetTotal = 15000

// RunMetrics is the CLI entry point for the metrics command.
// Reads golden set stats from DuckDB and pushes them to InfluxDB as
// golden_set_stats, golden_set_domain, and golden_set_voice measurements.
func RunMetrics(args []string) {
	fs := flag.NewFlagSet("metrics", flag.ExitOnError)

	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	influxURL := fs.String("influx", "", "InfluxDB URL")
	influxDB := fs.String("influx-db", "", "InfluxDB database name")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *dbPath == "" {
		*dbPath = os.Getenv("LEM_DB")
	}
	if *dbPath == "" {
		fmt.Fprintln(os.Stderr, "error: --db or LEM_DB required (path to DuckDB file)")
		os.Exit(1)
	}

	db, err := OpenDB(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	// Query overall stats.
	var total, domains, voices int
	var avgGenTime, avgChars float64

	err = db.conn.QueryRow(`
		SELECT count(*), count(DISTINCT domain), count(DISTINCT voice),
		       coalesce(avg(gen_time), 0), coalesce(avg(char_count), 0)
		FROM golden_set
	`).Scan(&total, &domains, &voices, &avgGenTime, &avgChars)
	if err != nil {
		log.Fatalf("query golden_set stats: %v", err)
	}

	if total == 0 {
		fmt.Println("No golden set data in DuckDB.")
		return
	}

	nowNs := time.Now().UTC().UnixNano()
	pct := float64(total) / float64(targetTotal) * 100.0

	var lines []string

	// Overall stats measurement.
	lines = append(lines, fmt.Sprintf(
		"golden_set_stats total_examples=%di,domains=%di,voices=%di,avg_gen_time=%.2f,avg_response_chars=%.0f,completion_pct=%.1f %d",
		total, domains, voices, avgGenTime, avgChars, pct, nowNs,
	))

	// Per-domain stats.
	domainRows, err := db.conn.Query(`
		SELECT domain, count(*) AS n, avg(gen_time) AS avg_t
		FROM golden_set GROUP BY domain
	`)
	if err != nil {
		log.Fatalf("query domains: %v", err)
	}
	domainCount := 0
	for domainRows.Next() {
		var domain string
		var n int
		var avgT float64
		if err := domainRows.Scan(&domain, &n, &avgT); err != nil {
			log.Fatalf("scan domain row: %v", err)
		}
		lines = append(lines, fmt.Sprintf(
			"golden_set_domain,domain=%s count=%di,avg_gen_time=%.2f %d",
			escapeLp(domain), n, avgT, nowNs,
		))
		domainCount++
	}
	domainRows.Close()

	// Per-voice stats.
	voiceRows, err := db.conn.Query(`
		SELECT voice, count(*) AS n, avg(char_count) AS avg_c, avg(gen_time) AS avg_t
		FROM golden_set GROUP BY voice
	`)
	if err != nil {
		log.Fatalf("query voices: %v", err)
	}
	voiceCount := 0
	for voiceRows.Next() {
		var voice string
		var n int
		var avgC, avgT float64
		if err := voiceRows.Scan(&voice, &n, &avgC, &avgT); err != nil {
			log.Fatalf("scan voice row: %v", err)
		}
		lines = append(lines, fmt.Sprintf(
			"golden_set_voice,voice=%s count=%di,avg_chars=%.0f,avg_gen_time=%.2f %d",
			escapeLp(voice), n, avgC, avgT, nowNs,
		))
		voiceCount++
	}
	voiceRows.Close()

	// Write to InfluxDB.
	influx := NewInfluxClient(*influxURL, *influxDB)
	if err := influx.WriteLp(lines); err != nil {
		log.Fatalf("write metrics: %v", err)
	}

	fmt.Printf("Wrote metrics to InfluxDB: %d examples, %d domains, %d voices (%d points)\n",
		total, domainCount, voiceCount, len(lines))
}
