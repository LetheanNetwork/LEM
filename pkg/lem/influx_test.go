package lem

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestEscapeLp(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "no special chars", in: "hello", want: "hello"},
		{name: "spaces", in: "hello world", want: `hello\ world`},
		{name: "commas", in: "a,b,c", want: `a\,b\,c`},
		{name: "equals", in: "key=val", want: `key\=val`},
		{name: "all specials", in: "a b,c=d", want: `a\ b\,c\=d`},
		{name: "empty string", in: "", want: ""},
		{name: "multiple spaces", in: "a  b", want: `a\ \ b`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := escapeLp(tt.in)
			if got != tt.want {
				t.Errorf("escapeLp(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNewInfluxClientTokenFromEnv(t *testing.T) {
	t.Setenv("INFLUX_TOKEN", "env-token-123")

	client := NewInfluxClient("http://localhost:8181", "testdb")
	if client.token != "env-token-123" {
		t.Errorf("expected token 'env-token-123', got %q", client.token)
	}
	if client.url != "http://localhost:8181" {
		t.Errorf("expected url 'http://localhost:8181', got %q", client.url)
	}
	if client.db != "testdb" {
		t.Errorf("expected db 'testdb', got %q", client.db)
	}
}

func TestNewInfluxClientTokenFromFile(t *testing.T) {
	// Clear env var so file is used.
	t.Setenv("INFLUX_TOKEN", "")

	// Write a temp token file.
	tmpDir := t.TempDir()
	tokenFile := filepath.Join(tmpDir, ".influx_token")
	if err := os.WriteFile(tokenFile, []byte("file-token-456\n"), 0600); err != nil {
		t.Fatalf("write token file: %v", err)
	}

	// Override home dir so NewInfluxClient reads our temp file.
	t.Setenv("HOME", tmpDir)

	client := NewInfluxClient("", "")
	if client.token != "file-token-456" {
		t.Errorf("expected token 'file-token-456', got %q", client.token)
	}
}

func TestNewInfluxClientDefaults(t *testing.T) {
	t.Setenv("INFLUX_TOKEN", "tok")

	client := NewInfluxClient("", "")
	if client.url != "http://10.69.69.165:8181" {
		t.Errorf("expected default url, got %q", client.url)
	}
	if client.db != "training" {
		t.Errorf("expected default db 'training', got %q", client.db)
	}
}

func TestWriteLp(t *testing.T) {
	var gotBody string
	var gotAuth string
	var gotContentType string
	var gotPath string
	var gotQuery string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.Query().Get("db")
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		gotBody = string(body)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	lines := []string{
		"cpu,host=server01 value=0.64",
		"cpu,host=server02 value=0.72",
	}
	err := client.WriteLp(lines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotPath != "/api/v3/write_lp" {
		t.Errorf("expected path /api/v3/write_lp, got %q", gotPath)
	}
	if gotQuery != "testdb" {
		t.Errorf("expected db=testdb, got %q", gotQuery)
	}
	if gotAuth != "Bearer test-token" {
		t.Errorf("expected 'Bearer test-token', got %q", gotAuth)
	}
	if gotContentType != "text/plain" {
		t.Errorf("expected 'text/plain', got %q", gotContentType)
	}

	want := "cpu,host=server01 value=0.64\ncpu,host=server02 value=0.72"
	if gotBody != want {
		t.Errorf("expected body %q, got %q", want, gotBody)
	}
}

func TestWriteLpError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("invalid line protocol"))
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	err := client.WriteLp([]string{"bad data"})
	if err == nil {
		t.Fatal("expected error for 400 response, got nil")
	}
}

func TestQuerySQL(t *testing.T) {
	var gotBody map[string]string
	var gotAuth string
	var gotContentType string
	var gotPath string

	responseData := []map[string]interface{}{
		{"id": "row1", "score": float64(7.5)},
		{"id": "row2", "score": float64(8.2)},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Fatalf("unmarshal request body: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(responseData)
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	rows, err := client.QuerySQL("SELECT * FROM scores LIMIT 2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotPath != "/api/v3/query_sql" {
		t.Errorf("expected path /api/v3/query_sql, got %q", gotPath)
	}
	if gotAuth != "Bearer test-token" {
		t.Errorf("expected 'Bearer test-token', got %q", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("expected 'application/json', got %q", gotContentType)
	}
	if gotBody["db"] != "testdb" {
		t.Errorf("expected db 'testdb' in body, got %q", gotBody["db"])
	}
	if gotBody["q"] != "SELECT * FROM scores LIMIT 2" {
		t.Errorf("expected query in body, got %q", gotBody["q"])
	}

	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(rows))
	}
	if rows[0]["id"] != "row1" {
		t.Errorf("expected row 0 id 'row1', got %v", rows[0]["id"])
	}
	if rows[1]["score"] != 8.2 {
		t.Errorf("expected row 1 score 8.2, got %v", rows[1]["score"])
	}
}

func TestQuerySQLError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	_, err := client.QuerySQL("SELECT bad")
	if err == nil {
		t.Fatal("expected error for 500 response, got nil")
	}
}

func TestQuerySQLBadJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not valid json"))
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	_, err := client.QuerySQL("SELECT 1")
	if err == nil {
		t.Fatal("expected error for invalid JSON response, got nil")
	}
}

func TestWriteLpEmptyLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	client := NewInfluxClient(server.URL, "testdb")

	// Empty slice should still work -- sends empty body.
	err := client.WriteLp([]string{})
	if err != nil {
		t.Fatalf("unexpected error for empty lines: %v", err)
	}
}
