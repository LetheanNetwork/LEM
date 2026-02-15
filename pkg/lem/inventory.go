package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
)

// RunInventory is the CLI entry point for the inventory command.
// Shows row counts and summary stats for all tables in the DuckDB database.
func RunInventory(args []string) {
	fs := flag.NewFlagSet("inventory", flag.ExitOnError)
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

	counts, err := db.TableCounts()
	if err != nil {
		log.Fatalf("table counts: %v", err)
	}

	fmt.Printf("LEM Database Inventory (%s)\n", *dbPath)
	fmt.Println("============================================================")

	grandTotal := 0
	for table, count := range counts {
		detail := ""

		switch table {
		case "golden_set":
			pct := float64(count) / float64(targetTotal) * 100
			detail = fmt.Sprintf("  (%.1f%% of %d target)", pct, targetTotal)
		case "training_examples":
			var sources int
			db.conn.QueryRow("SELECT COUNT(DISTINCT source) FROM training_examples").Scan(&sources)
			detail = fmt.Sprintf("  (%d sources)", sources)
		case "prompts":
			var domains, voices int
			db.conn.QueryRow("SELECT COUNT(DISTINCT domain) FROM prompts").Scan(&domains)
			db.conn.QueryRow("SELECT COUNT(DISTINCT voice) FROM prompts").Scan(&voices)
			detail = fmt.Sprintf("  (%d domains, %d voices)", domains, voices)
		case "gemini_responses":
			rows, _ := db.conn.Query("SELECT source_model, count(*) FROM gemini_responses GROUP BY source_model")
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
					detail = fmt.Sprintf("  (%s)", joinStrings(parts, ", "))
				}
			}
		case "benchmark_results":
			var sources int
			db.conn.QueryRow("SELECT COUNT(DISTINCT source) FROM benchmark_results").Scan(&sources)
			detail = fmt.Sprintf("  (%d categories)", sources)
		}

		fmt.Printf("  %-25s %8d%s\n", table, count, detail)
		grandTotal += count
	}

	fmt.Printf("  %-25s\n", "────────────────────────────────────────")
	fmt.Printf("  %-25s %8d\n", "TOTAL", grandTotal)
}

func joinStrings(parts []string, sep string) string {
	result := ""
	for i, p := range parts {
		if i > 0 {
			result += sep
		}
		result += p
	}
	return result
}
