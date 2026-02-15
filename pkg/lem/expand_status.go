package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
)

// RunExpandStatus is the CLI entry point for the expand-status command.
// Shows the expansion pipeline progress from DuckDB.
func RunExpandStatus(args []string) {
	fs := flag.NewFlagSet("expand-status", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")

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

	fmt.Println("LEM Expansion Pipeline Status")
	fmt.Println("==================================================")

	// Expansion prompts.
	var epTotal, epPending int
	err = db.conn.QueryRow("SELECT count(*) FROM expansion_prompts").Scan(&epTotal)
	if err != nil {
		fmt.Println("  Expansion prompts:  not created (run: lem normalize)")
		db.Close()
		return
	}
	db.conn.QueryRow("SELECT count(*) FROM expansion_prompts WHERE status = 'pending'").Scan(&epPending)
	fmt.Printf("  Expansion prompts:  %d total, %d pending\n", epTotal, epPending)

	// Generated responses.
	var generated int
	err = db.conn.QueryRow("SELECT count(*) FROM expansion_raw").Scan(&generated)
	if err != nil {
		fmt.Println("  Generated:          0 (run: lem expand)")
	} else {
		rows, _ := db.conn.Query("SELECT model, count(*) FROM expansion_raw GROUP BY model")
		if rows != nil {
			var parts []string
			for rows.Next() {
				var model string
				var n int
				rows.Scan(&model, &n)
				parts = append(parts, fmt.Sprintf("%s: %d", model, n))
			}
			rows.Close()
			if len(parts) > 0 {
				fmt.Printf("  Generated:          %d (%s)\n", generated, joinStrings(parts, ", "))
			} else {
				fmt.Printf("  Generated:          %d\n", generated)
			}
		}
	}

	// Scored.
	var scored, hPassed, jScored, jPassed int
	err = db.conn.QueryRow("SELECT count(*) FROM expansion_scores").Scan(&scored)
	if err != nil {
		fmt.Println("  Scored:             0 (run: lem score --tier 1)")
	} else {
		db.conn.QueryRow("SELECT count(*) FROM expansion_scores WHERE heuristic_pass = true").Scan(&hPassed)
		fmt.Printf("  Heuristic scored:   %d (%d passed)\n", scored, hPassed)

		db.conn.QueryRow("SELECT count(*) FROM expansion_scores WHERE judge_average IS NOT NULL").Scan(&jScored)
		db.conn.QueryRow("SELECT count(*) FROM expansion_scores WHERE judge_pass = true").Scan(&jPassed)
		if jScored > 0 {
			fmt.Printf("  Judge scored:       %d (%d passed)\n", jScored, jPassed)
		}
	}

	// Pipeline progress.
	if epTotal > 0 && generated > 0 {
		genPct := float64(generated) / float64(epTotal) * 100
		fmt.Printf("\n  Progress:           %.1f%% generated\n", genPct)
	}

	// Golden set context.
	var golden int
	err = db.conn.QueryRow("SELECT count(*) FROM golden_set").Scan(&golden)
	if err == nil {
		fmt.Printf("\n  Golden set:         %d / %d\n", golden, targetTotal)
		if generated > 0 {
			fmt.Printf("  Combined:           %d total examples\n", golden+generated)
		}
	}
}
