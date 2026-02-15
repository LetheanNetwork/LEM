package lem

import (
	"os"
	"path/filepath"
	"testing"
)

func createTestDB(t *testing.T) *DB {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.duckdb")

	db, err := OpenDBReadWrite(path)
	if err != nil {
		t.Fatalf("open test db: %v", err)
	}

	// Create golden_set table.
	_, err = db.conn.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, char_count INTEGER
	)`)
	if err != nil {
		t.Fatalf("create golden_set: %v", err)
	}

	// Create expansion_prompts table.
	_, err = db.conn.Exec(`CREATE TABLE expansion_prompts (
		idx BIGINT, seed_id VARCHAR, region VARCHAR, domain VARCHAR,
		language VARCHAR, prompt VARCHAR, prompt_en VARCHAR, priority INTEGER, status VARCHAR
	)`)
	if err != nil {
		t.Fatalf("create expansion_prompts: %v", err)
	}

	return db
}

func TestOpenDBReadOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.duckdb")

	// Create a DB first so the file exists.
	db, err := OpenDBReadWrite(path)
	if err != nil {
		t.Fatalf("create db: %v", err)
	}
	db.Close()

	// Now open read-only.
	roDB, err := OpenDB(path)
	if err != nil {
		t.Fatalf("open read-only: %v", err)
	}
	defer roDB.Close()

	if roDB.path != path {
		t.Errorf("path = %q, want %q", roDB.path, path)
	}
}

func TestOpenDBNotFound(t *testing.T) {
	_, err := OpenDB("/nonexistent/path/to.duckdb")
	if err == nil {
		t.Fatal("expected error for nonexistent path")
	}
}

func TestQueryGoldenSet(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	// Insert test data.
	_, err := db.conn.Exec(`INSERT INTO golden_set VALUES
		(0, 'seed1', 'Identity', 'junior', 'prompt one', 'response one with enough chars to pass', 10.5, 200),
		(1, 'seed2', 'Ethics', 'senior', 'prompt two', 'short', 5.0, 5),
		(2, 'seed3', 'Privacy', 'peer', 'prompt three', 'another good response with sufficient length', 8.2, 300)
	`)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Query with minChars=50 should return 2 (skip the short one).
	rows, err := db.QueryGoldenSet(50)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}
	if rows[0].SeedID != "seed1" {
		t.Errorf("first row seed_id = %q, want seed1", rows[0].SeedID)
	}
	if rows[1].Domain != "Privacy" {
		t.Errorf("second row domain = %q, want Privacy", rows[1].Domain)
	}
}

func TestQueryGoldenSetEmpty(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	rows, err := db.QueryGoldenSet(0)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("got %d rows, want 0", len(rows))
	}
}

func TestCountGoldenSet(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	_, err := db.conn.Exec(`INSERT INTO golden_set VALUES
		(0, 'seed1', 'Identity', 'junior', 'p1', 'r1', 10.5, 200),
		(1, 'seed2', 'Ethics', 'senior', 'p2', 'r2', 5.0, 150)
	`)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	count, err := db.CountGoldenSet()
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}
}

func TestQueryExpansionPrompts(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	_, err := db.conn.Exec(`INSERT INTO expansion_prompts VALUES
		(0, 'ep1', 'chinese', 'Identity', 'zh', 'prompt zh', 'prompt en', 1, 'pending'),
		(1, 'ep2', 'russian', 'Ethics', 'ru', 'prompt ru', 'prompt en2', 2, 'pending'),
		(2, 'ep3', 'english', 'Privacy', 'en', 'prompt en3', '', 1, 'completed')
	`)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Query pending only.
	rows, err := db.QueryExpansionPrompts("pending", 0)
	if err != nil {
		t.Fatalf("query pending: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}
	// Should be ordered by priority, idx.
	if rows[0].SeedID != "ep1" {
		t.Errorf("first row = %q, want ep1", rows[0].SeedID)
	}

	// Query all.
	all, err := db.QueryExpansionPrompts("", 0)
	if err != nil {
		t.Fatalf("query all: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("got %d rows, want 3", len(all))
	}

	// Query with limit.
	limited, err := db.QueryExpansionPrompts("pending", 1)
	if err != nil {
		t.Fatalf("query limited: %v", err)
	}
	if len(limited) != 1 {
		t.Fatalf("got %d rows, want 1", len(limited))
	}
}

func TestCountExpansionPrompts(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	_, err := db.conn.Exec(`INSERT INTO expansion_prompts VALUES
		(0, 'ep1', 'chinese', 'Identity', 'zh', 'p1', 'p1en', 1, 'pending'),
		(1, 'ep2', 'russian', 'Ethics', 'ru', 'p2', 'p2en', 2, 'completed'),
		(2, 'ep3', 'english', 'Privacy', 'en', 'p3', '', 1, 'pending')
	`)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	total, pending, err := db.CountExpansionPrompts()
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if total != 3 {
		t.Errorf("total = %d, want 3", total)
	}
	if pending != 2 {
		t.Errorf("pending = %d, want 2", pending)
	}
}

func TestUpdateExpansionStatus(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	_, err := db.conn.Exec(`INSERT INTO expansion_prompts VALUES
		(0, 'ep1', 'chinese', 'Identity', 'zh', 'p1', 'p1en', 1, 'pending')
	`)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	err = db.UpdateExpansionStatus(0, "completed")
	if err != nil {
		t.Fatalf("update: %v", err)
	}

	rows, err := db.QueryExpansionPrompts("completed", 0)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1", len(rows))
	}
	if rows[0].Status != "completed" {
		t.Errorf("status = %q, want completed", rows[0].Status)
	}
}

func TestTableCounts(t *testing.T) {
	db := createTestDB(t)
	defer db.Close()

	_, err := db.conn.Exec(`INSERT INTO golden_set VALUES
		(0, 's1', 'd1', 'v1', 'p1', 'r1', 1.0, 100)
	`)
	if err != nil {
		t.Fatalf("insert golden: %v", err)
	}

	counts, err := db.TableCounts()
	if err != nil {
		t.Fatalf("table counts: %v", err)
	}
	if counts["golden_set"] != 1 {
		t.Errorf("golden_set count = %d, want 1", counts["golden_set"])
	}
	if counts["expansion_prompts"] != 0 {
		t.Errorf("expansion_prompts count = %d, want 0", counts["expansion_prompts"])
	}
}

func TestOpenDBWithEnvDefault(t *testing.T) {
	// Test that OpenDB uses the default path from LEM_DB env if available.
	dir := t.TempDir()
	path := filepath.Join(dir, "env-test.duckdb")

	db, err := OpenDBReadWrite(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	db.Close()

	os.Setenv("LEM_DB", path)
	defer os.Unsetenv("LEM_DB")

	db2, err := OpenDB(path)
	if err != nil {
		t.Fatalf("open via env: %v", err)
	}
	defer db2.Close()
}
