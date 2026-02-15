package lem

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// RunTierScore is the CLI entry point for the tier-score command.
// Scores expansion responses using tiered quality assessment:
//   - Tier 1: Heuristic regex scoring (fast, no API)
//   - Tier 2: LEM self-judge (requires trained model)
//   - Tier 3: External judge (reserved for borderline cases)
func RunTierScore(args []string) {
	fs := flag.NewFlagSet("tier-score", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	tier := fs.Int("tier", 1, "Scoring tier: 1=heuristic, 2=LEM judge, 3=external")
	limit := fs.Int("limit", 0, "Max items to score (0=all)")

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

	// Ensure expansion_scores table exists.
	db.conn.Exec(`
		CREATE TABLE IF NOT EXISTS expansion_scores (
			idx INT,
			heuristic_score DOUBLE,
			heuristic_pass BOOLEAN,
			judge_sovereignty DOUBLE,
			judge_ethical_depth DOUBLE,
			judge_creative DOUBLE,
			judge_self_concept DOUBLE,
			judge_average DOUBLE,
			judge_pass BOOLEAN,
			judge_model VARCHAR,
			scored_at TIMESTAMP
		)
	`)

	if *tier >= 1 {
		runHeuristicTier(db, *limit)
	}

	if *tier >= 2 {
		fmt.Println("\nTier 2 (LEM judge): not yet available — needs trained LEM-27B model")
		fmt.Println("  Will score: sovereignty, ethical_depth, creative, self_concept (1-10 each)")
	}

	if *tier >= 3 {
		fmt.Println("\nTier 3 (External judge): reserved for borderline cases")
	}
}

func runHeuristicTier(db *DB, limit int) {
	// Find unscored responses.
	query := `
		SELECT r.idx, r.response FROM expansion_raw r
		LEFT JOIN expansion_scores s ON r.idx = s.idx
		WHERE s.idx IS NULL
		ORDER BY r.idx
	`
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := db.conn.Query(query)
	if err != nil {
		log.Fatalf("query unscored: %v", err)
	}
	defer rows.Close()

	type unscoredRow struct {
		idx      int
		response string
	}
	var unscored []unscoredRow

	for rows.Next() {
		var r unscoredRow
		rows.Scan(&r.idx, &r.response)
		unscored = append(unscored, r)
	}

	if len(unscored) == 0 {
		fmt.Println("Tier 1 (heuristic): all responses already scored")
		return
	}

	fmt.Printf("Tier 1 (heuristic): scoring %d responses...\n", len(unscored))

	passed := 0
	for _, r := range unscored {
		score, _ := heuristicExpansionScore(r.response)
		isPass := score > 0
		if isPass {
			passed++
		}

		db.conn.Exec(`
			INSERT INTO expansion_scores (idx, heuristic_score, heuristic_pass, scored_at)
			VALUES (?, ?, ?, current_timestamp)
		`, r.idx, score, isPass)
	}

	fmt.Printf("  Scored: %d, Passed: %d, Failed: %d\n", len(unscored), passed, len(unscored)-passed)
	if len(unscored) > 0 {
		fmt.Printf("  Pass rate: %.1f%%\n", float64(passed)/float64(len(unscored))*100)
	}
}

// heuristicExpansionScore applies fast heuristic scoring to an expansion response.
// Returns (score, details). Positive = good, negative = bad.
func heuristicExpansionScore(response string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})

	if response == "" || len(response) < 30 {
		return -20.0, map[string]interface{}{"reason": "empty_or_broken"}
	}

	score := 0.0

	// Compliance markers (negative).
	compliancePhrases := []string{
		"as an ai", "i cannot", "i can't", "i'm not able",
		"important to note", "it's crucial to", "i must emphasize",
		"i should point out", "as a language model",
	}
	lower := strings.ToLower(response)
	complianceCount := 0
	for _, p := range compliancePhrases {
		if strings.Contains(lower, p) {
			complianceCount++
		}
	}
	if complianceCount > 0 {
		score -= float64(complianceCount) * 5
		details["compliance_markers"] = complianceCount
	}

	// Formulaic preamble (negative).
	trimmed := strings.TrimSpace(lower)
	if strings.HasPrefix(trimmed, "okay, let") || strings.HasPrefix(trimmed, "ok, let") || strings.HasPrefix(trimmed, "sure, let") {
		score -= 3
		details["formulaic_preamble"] = true
	}

	// Degeneration check (repetitive output).
	words := strings.Fields(response)
	if len(words) > 20 {
		chunks := make([]string, 0, len(words)/5)
		for i := 0; i+5 <= len(words); i += 5 {
			chunks = append(chunks, strings.Join(words[i:i+5], " "))
		}
		if len(chunks) > 0 {
			unique := make(map[string]bool)
			for _, c := range chunks {
				unique[c] = true
			}
			ratio := float64(len(unique)) / float64(len(chunks))
			if ratio < 0.5 {
				score -= 10
				details["degeneration"] = true
			}
		}
	}

	// Engagement depth (positive).
	wordCount := len(words)
	if wordCount > 100 {
		score += 2
	}
	if wordCount > 300 {
		score += 2
	}
	details["word_count"] = wordCount

	// Structure (positive).
	if strings.Contains(response, "\n\n") || strings.Contains(response, "**") ||
		strings.Contains(response, "1.") || strings.Contains(response, "- ") {
		score += 1
		details["structured"] = true
	}

	// Creative expression (positive).
	creativeMarkers := []string{"metaphor", "imagine", "picture this", "story", "once upon"}
	for _, m := range creativeMarkers {
		if strings.Contains(lower, m) {
			score += 2
			details["creative"] = true
			break
		}
	}

	// First-person engagement (positive).
	fpMarkers := []string{"i think", "i believe", "in my view", "i'd argue"}
	fpCount := 0
	for _, m := range fpMarkers {
		if strings.Contains(lower, m) {
			fpCount++
		}
	}
	if fpCount > 0 {
		score += float64(fpCount) * 1.5
		details["first_person"] = fpCount
	}

	return score, details
}
