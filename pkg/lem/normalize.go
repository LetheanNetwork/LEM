package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
)

// RunNormalize is the CLI entry point for the normalize command.
// Normalizes seeds into the expansion_prompts table, deduplicating against
// the golden set and existing prompts. Assigns priority based on domain
// coverage (underrepresented domains first).
func RunNormalize(args []string) {
	fs := flag.NewFlagSet("normalize", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	minLen := fs.Int("min-length", 50, "Minimum prompt length in characters")

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

	db, err := OpenDBReadWrite(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	// Check source tables.
	var seedCount int
	if err := db.conn.QueryRow("SELECT count(*) FROM seeds").Scan(&seedCount); err != nil {
		log.Fatalf("No seeds table. Run: lem import-all first")
	}
	fmt.Printf("Seeds table: %d rows\n", seedCount)

	// Drop and recreate expansion_prompts.
	_, err = db.conn.Exec("DROP TABLE IF EXISTS expansion_prompts")
	if err != nil {
		log.Fatalf("drop expansion_prompts: %v", err)
	}

	// Deduplicate: remove seeds whose prompt already appears in prompts or golden_set.
	_, err = db.conn.Exec(fmt.Sprintf(`
		CREATE TABLE expansion_prompts AS
		WITH unique_seeds AS (
			SELECT
				ROW_NUMBER() OVER (ORDER BY region, domain, seed_id) AS idx,
				seed_id,
				region,
				domain,
				prompt
			FROM (
				SELECT DISTINCT ON (prompt)
					seed_id, region, domain, prompt
				FROM seeds
				WHERE length(prompt) >= %d
				ORDER BY prompt, seed_id
			)
		),
		existing_prompts AS (
			SELECT prompt FROM prompts
			UNION ALL
			SELECT prompt FROM golden_set
		)
		SELECT
			us.idx,
			us.seed_id,
			us.region,
			us.domain,
			'en' AS language,
			us.prompt,
			'' AS prompt_en,
			0 AS priority,
			'pending' AS status
		FROM unique_seeds us
		WHERE NOT EXISTS (
			SELECT 1 FROM existing_prompts ep
			WHERE ep.prompt = us.prompt
		)
	`, *minLen))
	if err != nil {
		log.Fatalf("create expansion_prompts: %v", err)
	}

	var total, domains, regions int
	db.conn.QueryRow("SELECT count(*) FROM expansion_prompts").Scan(&total)
	db.conn.QueryRow("SELECT count(DISTINCT domain) FROM expansion_prompts").Scan(&domains)
	db.conn.QueryRow("SELECT count(DISTINCT region) FROM expansion_prompts").Scan(&regions)

	// Assign priority based on domain coverage.
	_, err = db.conn.Exec(`
		UPDATE expansion_prompts SET priority = (
			SELECT RANK() OVER (ORDER BY cnt ASC)
			FROM (
				SELECT domain, count(*) AS cnt
				FROM expansion_prompts GROUP BY domain
			) domain_counts
			WHERE domain_counts.domain = expansion_prompts.domain
		)
	`)
	if err != nil {
		log.Printf("warning: priority assignment failed: %v", err)
	}

	fmt.Printf("\nExpansion Prompts: %d\n", total)
	fmt.Printf("  Domains: %d\n", domains)
	fmt.Printf("  Regions: %d\n", regions)

	// Show region distribution.
	fmt.Println("\n  By region group:")
	rows, err := db.conn.Query(`
		SELECT
			CASE
				WHEN region LIKE '%cn%' THEN 'cn'
				WHEN region LIKE '%en-%' OR region LIKE '%en_para%' OR region LIKE '%para%' THEN 'en'
				WHEN region LIKE '%ru%' THEN 'ru'
				WHEN region LIKE '%de%' AND region NOT LIKE '%deten%' THEN 'de'
				WHEN region LIKE '%es%' THEN 'es'
				WHEN region LIKE '%fr%' THEN 'fr'
				WHEN region LIKE '%latam%' THEN 'latam'
				WHEN region LIKE '%africa%' THEN 'africa'
				WHEN region LIKE '%eu%' THEN 'eu'
				WHEN region LIKE '%me%' AND region NOT LIKE '%premium%' THEN 'me'
				ELSE 'other'
			END AS lang_group,
			count(*) AS n
		FROM expansion_prompts GROUP BY lang_group ORDER BY n DESC
	`)
	if err == nil {
		for rows.Next() {
			var group string
			var n int
			rows.Scan(&group, &n)
			fmt.Printf("    %-15s %6d\n", group, n)
		}
		rows.Close()
	}

	fmt.Printf("\nNormalization complete: %d expansion prompts from %d seeds\n", total, seedCount)
}
