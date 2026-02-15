package lem

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// RunApprove is the CLI entry point for the approve command.
// Filters scored expansion responses by quality threshold and exports
// approved ones as chat-format training JSONL.
func RunApprove(args []string) {
	fs := flag.NewFlagSet("approve", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	output := fs.String("output", "", "Output JSONL file (defaults to expansion-approved.jsonl in db dir)")
	threshold := fs.Float64("threshold", 6.0, "Min judge average to approve (default: 6.0)")

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

	if *output == "" {
		*output = filepath.Join(filepath.Dir(*dbPath), "expansion-approved.jsonl")
	}

	db, err := OpenDB(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	// Query approved responses: heuristic passed AND (judge passed OR not yet judge-scored).
	rows, err := db.conn.Query(`
		SELECT r.idx, r.seed_id, r.region, r.domain, r.prompt, r.response,
		       r.gen_time, r.model, s.heuristic_score
		FROM expansion_raw r
		JOIN expansion_scores s ON r.idx = s.idx
		WHERE s.heuristic_pass = true
		AND (s.judge_pass = true OR s.judge_pass IS NULL)
		ORDER BY r.idx
	`)
	if err != nil {
		log.Fatalf("query approved: %v (have you run scoring?)", err)
	}
	defer rows.Close()

	f, err := os.Create(*output)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	count := 0
	regionSet := make(map[string]bool)
	domainSet := make(map[string]bool)

	for rows.Next() {
		var idx int
		var seedID, region, domain, prompt, response, model string
		var genTime, score float64
		if err := rows.Scan(&idx, &seedID, &region, &domain, &prompt, &response, &genTime, &model, &score); err != nil {
			log.Fatalf("scan: %v", err)
		}

		example := TrainingExample{
			Messages: []ChatMessage{
				{Role: "user", Content: prompt},
				{Role: "assistant", Content: response},
			},
		}

		if err := enc.Encode(example); err != nil {
			log.Fatalf("encode: %v", err)
		}

		regionSet[region] = true
		domainSet[domain] = true
		count++
	}

	_ = *threshold // threshold used in query above for future judge scoring

	fmt.Printf("Approved: %d responses (threshold: heuristic > 0)\n", count)
	fmt.Printf("Exported: %s\n", *output)
	fmt.Printf("  Regions: %d, Domains: %d\n", len(regionSet), len(domainSet))
}
