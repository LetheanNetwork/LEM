package main

import (
	"database/sql"
	"fmt"

	_ "github.com/marcboeker/go-duckdb"
)

// DB wraps a DuckDB connection.
type DB struct {
	conn *sql.DB
	path string
}

// OpenDB opens a DuckDB database file. Use read-only mode by default
// to avoid locking issues with the Python pipeline.
func OpenDB(path string) (*DB, error) {
	conn, err := sql.Open("duckdb", path+"?access_mode=READ_ONLY")
	if err != nil {
		return nil, fmt.Errorf("open duckdb %s: %w", path, err)
	}
	// Verify connection works.
	if err := conn.Ping(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("ping duckdb %s: %w", path, err)
	}
	return &DB{conn: conn, path: path}, nil
}

// OpenDBReadWrite opens a DuckDB database in read-write mode.
func OpenDBReadWrite(path string) (*DB, error) {
	conn, err := sql.Open("duckdb", path)
	if err != nil {
		return nil, fmt.Errorf("open duckdb %s: %w", path, err)
	}
	if err := conn.Ping(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("ping duckdb %s: %w", path, err)
	}
	return &DB{conn: conn, path: path}, nil
}

// Close closes the database connection.
func (db *DB) Close() error {
	return db.conn.Close()
}

// GoldenSetRow represents one row from the golden_set table.
type GoldenSetRow struct {
	Idx       int
	SeedID    string
	Domain    string
	Voice     string
	Prompt    string
	Response  string
	GenTime   float64
	CharCount int
}

// ExpansionPromptRow represents one row from the expansion_prompts table.
type ExpansionPromptRow struct {
	Idx      int64
	SeedID   string
	Region   string
	Domain   string
	Language string
	Prompt   string
	PromptEn string
	Priority int
	Status   string
}

// QueryGoldenSet returns all golden set rows with responses >= minChars.
func (db *DB) QueryGoldenSet(minChars int) ([]GoldenSetRow, error) {
	rows, err := db.conn.Query(
		"SELECT idx, seed_id, domain, voice, prompt, response, gen_time, char_count "+
			"FROM golden_set WHERE char_count >= ? ORDER BY idx",
		minChars,
	)
	if err != nil {
		return nil, fmt.Errorf("query golden_set: %w", err)
	}
	defer rows.Close()

	var result []GoldenSetRow
	for rows.Next() {
		var r GoldenSetRow
		if err := rows.Scan(&r.Idx, &r.SeedID, &r.Domain, &r.Voice,
			&r.Prompt, &r.Response, &r.GenTime, &r.CharCount); err != nil {
			return nil, fmt.Errorf("scan golden_set row: %w", err)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

// CountGoldenSet returns the total count of golden set rows.
func (db *DB) CountGoldenSet() (int, error) {
	var count int
	err := db.conn.QueryRow("SELECT COUNT(*) FROM golden_set").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count golden_set: %w", err)
	}
	return count, nil
}

// QueryExpansionPrompts returns expansion prompts filtered by status.
// If status is empty, returns all prompts.
func (db *DB) QueryExpansionPrompts(status string, limit int) ([]ExpansionPromptRow, error) {
	query := "SELECT idx, seed_id, region, domain, language, prompt, prompt_en, priority, status " +
		"FROM expansion_prompts"
	var args []interface{}

	if status != "" {
		query += " WHERE status = ?"
		args = append(args, status)
	}
	query += " ORDER BY priority, idx"

	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := db.conn.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("query expansion_prompts: %w", err)
	}
	defer rows.Close()

	var result []ExpansionPromptRow
	for rows.Next() {
		var r ExpansionPromptRow
		if err := rows.Scan(&r.Idx, &r.SeedID, &r.Region, &r.Domain,
			&r.Language, &r.Prompt, &r.PromptEn, &r.Priority, &r.Status); err != nil {
			return nil, fmt.Errorf("scan expansion_prompt row: %w", err)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

// CountExpansionPrompts returns counts by status.
func (db *DB) CountExpansionPrompts() (total int, pending int, err error) {
	err = db.conn.QueryRow("SELECT COUNT(*) FROM expansion_prompts").Scan(&total)
	if err != nil {
		return 0, 0, fmt.Errorf("count expansion_prompts: %w", err)
	}
	err = db.conn.QueryRow("SELECT COUNT(*) FROM expansion_prompts WHERE status = 'pending'").Scan(&pending)
	if err != nil {
		return total, 0, fmt.Errorf("count pending expansion_prompts: %w", err)
	}
	return total, pending, nil
}

// UpdateExpansionStatus updates the status of an expansion prompt by idx.
func (db *DB) UpdateExpansionStatus(idx int64, status string) error {
	_, err := db.conn.Exec("UPDATE expansion_prompts SET status = ? WHERE idx = ?", status, idx)
	if err != nil {
		return fmt.Errorf("update expansion_prompt %d: %w", idx, err)
	}
	return nil
}

// TableCounts returns row counts for all known tables.
func (db *DB) TableCounts() (map[string]int, error) {
	tables := []string{"golden_set", "expansion_prompts", "seeds", "prompts",
		"training_examples", "gemini_responses", "benchmark_questions", "benchmark_results", "validations"}

	counts := make(map[string]int)
	for _, t := range tables {
		var count int
		err := db.conn.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", t)).Scan(&count)
		if err != nil {
			// Table might not exist — skip.
			continue
		}
		counts[t] = count
	}
	return counts, nil
}
