package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// RunCoverage is the CLI entry point for the coverage command.
// Analyzes seed coverage and shows underrepresented areas.
func RunCoverage(args []string) {
	fs := flag.NewFlagSet("coverage", flag.ExitOnError)
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

	var total int
	if err := db.conn.QueryRow("SELECT count(*) FROM seeds").Scan(&total); err != nil {
		log.Fatalf("No seeds table. Run: lem import-all first")
	}

	fmt.Println("LEM Seed Coverage Analysis")
	fmt.Println("==================================================")
	fmt.Printf("\nTotal seeds: %d\n", total)

	// Region distribution.
	fmt.Println("\nRegion distribution (underrepresented first):")
	rows, err := db.conn.Query(`
		SELECT
			CASE
				WHEN region LIKE '%cn%' THEN 'cn (Chinese)'
				WHEN region LIKE '%en-%' OR region LIKE '%en_para%' OR region LIKE '%para%' THEN 'en (English)'
				WHEN region LIKE '%ru%' THEN 'ru (Russian)'
				WHEN region LIKE '%de%' AND region NOT LIKE '%deten%' THEN 'de (German)'
				WHEN region LIKE '%es%' THEN 'es (Spanish)'
				WHEN region LIKE '%fr%' THEN 'fr (French)'
				WHEN region LIKE '%latam%' THEN 'latam (LatAm)'
				WHEN region LIKE '%africa%' THEN 'africa'
				WHEN region LIKE '%eu%' THEN 'eu (European)'
				WHEN region LIKE '%me%' AND region NOT LIKE '%premium%' THEN 'me (MidEast)'
				WHEN region LIKE '%multi%' THEN 'multilingual'
				WHEN region LIKE '%weak%' THEN 'weak-langs'
				ELSE 'other'
			END AS lang_group,
			count(*) AS n,
			count(DISTINCT domain) AS domains
		FROM seeds GROUP BY lang_group ORDER BY n ASC
	`)
	if err != nil {
		log.Fatalf("query regions: %v", err)
	}

	type regionRow struct {
		group   string
		n       int
		domains int
	}
	var regionRows []regionRow
	for rows.Next() {
		var r regionRow
		rows.Scan(&r.group, &r.n, &r.domains)
		regionRows = append(regionRows, r)
	}
	rows.Close()

	avg := float64(total) / float64(len(regionRows))
	for _, r := range regionRows {
		barLen := int(float64(r.n) / avg * 10)
		if barLen > 40 {
			barLen = 40
		}
		bar := strings.Repeat("#", barLen)
		gap := ""
		if float64(r.n) < avg*0.5 {
			gap = "  <- UNDERREPRESENTED"
		}
		fmt.Printf("  %-22s %6d  (%4d domains)  %s%s\n", r.group, r.n, r.domains, bar, gap)
	}

	// Top 10 domains.
	fmt.Println("\nTop 10 domains (most seeds):")
	topRows, err := db.conn.Query(`
		SELECT domain, count(*) AS n FROM seeds
		WHERE domain != '' GROUP BY domain ORDER BY n DESC LIMIT 10
	`)
	if err == nil {
		for topRows.Next() {
			var domain string
			var n int
			topRows.Scan(&domain, &n)
			fmt.Printf("  %-40s %5d\n", domain, n)
		}
		topRows.Close()
	}

	// Bottom 10 domains.
	fmt.Println("\nBottom 10 domains (fewest seeds, min 5):")
	bottomRows, err := db.conn.Query(`
		SELECT domain, count(*) AS n FROM seeds
		WHERE domain != '' GROUP BY domain HAVING count(*) >= 5 ORDER BY n ASC LIMIT 10
	`)
	if err == nil {
		for bottomRows.Next() {
			var domain string
			var n int
			bottomRows.Scan(&domain, &n)
			fmt.Printf("  %-40s %5d\n", domain, n)
		}
		bottomRows.Close()
	}

	fmt.Println("\nSuggested expansion areas:")
	fmt.Println("  - Japanese, Korean, Thai, Vietnamese (no seeds found)")
	fmt.Println("  - Hindi/Urdu, Bengali, Tamil (South Asian)")
	fmt.Println("  - Swahili, Yoruba, Amharic (Sub-Saharan Africa)")
	fmt.Println("  - Indigenous languages (Quechua, Nahuatl, Aymara)")
}
