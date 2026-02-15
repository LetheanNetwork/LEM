package lem

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// RunQuery is the CLI entry point for the query command.
// Runs ad-hoc SQL against the DuckDB database.
func RunQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	dbPath := fs.String("db", "", "DuckDB database path (defaults to LEM_DB env)")
	jsonOutput := fs.Bool("json", false, "Output as JSON instead of table")

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

	sql := strings.Join(fs.Args(), " ")
	if sql == "" {
		fmt.Fprintln(os.Stderr, "error: SQL query required as positional argument")
		fmt.Fprintln(os.Stderr, "  lem query --db path.duckdb \"SELECT * FROM golden_set LIMIT 5\"")
		fmt.Fprintln(os.Stderr, "  lem query --db path.duckdb \"domain = 'ethics'\"  (auto-wraps as WHERE clause)")
		os.Exit(1)
	}

	// Auto-wrap non-SELECT queries as WHERE clauses.
	trimmed := strings.TrimSpace(strings.ToUpper(sql))
	if !strings.HasPrefix(trimmed, "SELECT") && !strings.HasPrefix(trimmed, "SHOW") &&
		!strings.HasPrefix(trimmed, "DESCRIBE") && !strings.HasPrefix(trimmed, "EXPLAIN") {
		sql = "SELECT * FROM golden_set WHERE " + sql + " LIMIT 20"
	}

	db, err := OpenDB(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	rows, err := db.conn.Query(sql)
	if err != nil {
		log.Fatalf("query: %v", err)
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		log.Fatalf("columns: %v", err)
	}

	var results []map[string]interface{}

	for rows.Next() {
		values := make([]interface{}, len(cols))
		ptrs := make([]interface{}, len(cols))
		for i := range values {
			ptrs[i] = &values[i]
		}

		if err := rows.Scan(ptrs...); err != nil {
			log.Fatalf("scan: %v", err)
		}

		row := make(map[string]interface{})
		for i, col := range cols {
			v := values[i]
			// Convert []byte to string for readability.
			if b, ok := v.([]byte); ok {
				v = string(b)
			}
			row[col] = v
		}
		results = append(results, row)
	}

	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(results)
		return
	}

	// Table output.
	if len(results) == 0 {
		fmt.Println("(no results)")
		return
	}

	// Calculate column widths.
	widths := make(map[string]int)
	for _, col := range cols {
		widths[col] = len(col)
	}
	for _, row := range results {
		for _, col := range cols {
			s := fmt.Sprintf("%v", row[col])
			if len(s) > 60 {
				s = s[:57] + "..."
			}
			if len(s) > widths[col] {
				widths[col] = len(s)
			}
		}
	}

	// Print header.
	for i, col := range cols {
		if i > 0 {
			fmt.Print("  ")
		}
		fmt.Printf("%-*s", widths[col], col)
	}
	fmt.Println()

	// Print separator.
	for i, col := range cols {
		if i > 0 {
			fmt.Print("  ")
		}
		fmt.Print(strings.Repeat("─", widths[col]))
	}
	fmt.Println()

	// Print rows.
	for _, row := range results {
		for i, col := range cols {
			if i > 0 {
				fmt.Print("  ")
			}
			s := fmt.Sprintf("%v", row[col])
			if len(s) > 60 {
				s = s[:57] + "..."
			}
			fmt.Printf("%-*s", widths[col], s)
		}
		fmt.Println()
	}

	fmt.Printf("\n(%d rows)\n", len(results))
}
