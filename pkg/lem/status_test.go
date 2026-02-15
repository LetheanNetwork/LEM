package lem

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// mockInfluxServer creates an httptest server that routes /api/v3/query_sql
// requests to the given handler function. The handler receives the parsed
// query body and writes the JSON response.
func mockInfluxServer(t *testing.T, handler func(q string) ([]map[string]interface{}, int)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v3/query_sql" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		var body struct {
			DB string `json:"db"`
			Q  string `json:"q"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		rows, status := handler(body.Q)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		json.NewEncoder(w).Encode(rows)
	}))
}

func TestPrintStatusFullOutput(t *testing.T) {
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "run_id": "run1", "status": "complete", "iteration": float64(1000), "total_iters": float64(1000), "pct": float64(100.0)},
				{"model": "gemma-3-12b", "run_id": "run2", "status": "training", "iteration": float64(340), "total_iters": float64(600), "pct": float64(56.7)},
				{"model": "gemma-3-27b", "run_id": "run3", "status": "pending", "iteration": float64(0), "total_iters": float64(400), "pct": float64(0.0)},
			}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "loss_type": "train", "loss": float64(1.434), "iteration": float64(1000), "tokens_per_sec": float64(512.3)},
				{"model": "gemma-3-12b", "loss_type": "train", "loss": float64(0.735), "iteration": float64(340), "tokens_per_sec": float64(128.5)},
			}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(15000), "target": float64(15000), "pct": float64(100.0)},
			}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(0), "target": float64(46331), "pct": float64(0.0)},
			}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Verify training section header.
	if !strings.Contains(output, "Training:") {
		t.Error("output missing 'Training:' header")
	}

	// Verify generation section header.
	if !strings.Contains(output, "Generation:") {
		t.Error("output missing 'Generation:' header")
	}

	// Verify model training rows.
	if !strings.Contains(output, "gemma-3-1b") {
		t.Error("output missing gemma-3-1b model")
	}
	if !strings.Contains(output, "complete") {
		t.Error("output missing 'complete' status")
	}
	if !strings.Contains(output, "1000/1000") {
		t.Error("output missing 1000/1000 progress")
	}
	if !strings.Contains(output, "100.0%") {
		t.Error("output missing 100.0% for gemma-3-1b")
	}
	if !strings.Contains(output, "loss=1.434") {
		t.Error("output missing loss=1.434")
	}

	if !strings.Contains(output, "gemma-3-12b") {
		t.Error("output missing gemma-3-12b model")
	}
	if !strings.Contains(output, "training") {
		t.Error("output missing 'training' status")
	}
	if !strings.Contains(output, "340/600") {
		t.Error("output missing 340/600 progress")
	}
	if !strings.Contains(output, "56.7%") {
		t.Error("output missing 56.7%")
	}
	if !strings.Contains(output, "loss=0.735") {
		t.Error("output missing loss=0.735")
	}

	if !strings.Contains(output, "gemma-3-27b") {
		t.Error("output missing gemma-3-27b model")
	}
	if !strings.Contains(output, "pending") {
		t.Error("output missing 'pending' status")
	}
	if !strings.Contains(output, "0/400") {
		t.Error("output missing 0/400 progress")
	}

	// Verify generation rows.
	if !strings.Contains(output, "golden") {
		t.Error("output missing golden generation row")
	}
	if !strings.Contains(output, "15000/15000") {
		t.Error("output missing 15000/15000 progress")
	}
	if !strings.Contains(output, "expansion") {
		t.Error("output missing expansion generation row")
	}
	if !strings.Contains(output, "0/46331") {
		t.Error("output missing 0/46331 progress")
	}
	if !strings.Contains(output, "m3-gpu0") {
		t.Error("output missing worker name m3-gpu0")
	}
}

func TestPrintStatusEmptyResults(t *testing.T) {
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		return []map[string]interface{}{}, http.StatusOK
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Should still have section headers even with no data.
	if !strings.Contains(output, "Training:") {
		t.Error("output missing 'Training:' header with empty data")
	}
	if !strings.Contains(output, "Generation:") {
		t.Error("output missing 'Generation:' header with empty data")
	}

	// Should indicate no data.
	if !strings.Contains(output, "no data") {
		t.Error("output should indicate 'no data' for empty results")
	}
}

func TestPrintStatusDeduplicatesModels(t *testing.T) {
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			// Two rows for same model — first should win (latest by time desc).
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "run_id": "run2", "status": "training", "iteration": float64(500), "total_iters": float64(1000), "pct": float64(50.0)},
				{"model": "gemma-3-1b", "run_id": "run1", "status": "complete", "iteration": float64(1000), "total_iters": float64(1000), "pct": float64(100.0)},
			}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			// Two rows for same model — first should win.
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "loss_type": "train", "loss": float64(0.8), "iteration": float64(500), "tokens_per_sec": float64(256.0)},
				{"model": "gemma-3-1b", "loss_type": "train", "loss": float64(1.5), "iteration": float64(200), "tokens_per_sec": float64(200.0)},
			}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(5000), "target": float64(15000), "pct": float64(33.3)},
				{"worker": "m3-gpu0", "completed": float64(3000), "target": float64(15000), "pct": float64(20.0)},
			}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Should show 500/1000 (latest) not 1000/1000 (older).
	if !strings.Contains(output, "500/1000") {
		t.Error("expected 500/1000 from latest row, deduplication may have failed")
	}
	if strings.Contains(output, "1000/1000") {
		t.Error("unexpected 1000/1000 — older row should be deduped out")
	}

	// Should show loss=0.800 (latest) not loss=1.500.
	if !strings.Contains(output, "loss=0.800") {
		t.Error("expected loss=0.800 from latest row")
	}

	// Golden should show 5000/15000 (latest).
	if !strings.Contains(output, "5000/15000") {
		t.Error("expected 5000/15000 from latest golden row")
	}
	if strings.Contains(output, "3000/15000") {
		t.Error("unexpected 3000/15000 — older golden row should be deduped out")
	}
}

func TestPrintStatusPartialData(t *testing.T) {
	// Training status exists but no loss data.
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			return []map[string]interface{}{
				{"model": "gemma-3-4b", "run_id": "run1", "status": "training", "iteration": float64(100), "total_iters": float64(500), "pct": float64(20.0)},
			}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			return []map[string]interface{}{}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu1", "completed": float64(7000), "target": float64(15000), "pct": float64(46.7)},
			}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Model should appear without loss data.
	if !strings.Contains(output, "gemma-3-4b") {
		t.Error("output missing gemma-3-4b")
	}
	if !strings.Contains(output, "100/500") {
		t.Error("output missing 100/500")
	}
	// Should NOT have a "loss=" for this model since no loss data.
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.Contains(line, "gemma-3-4b") && strings.Contains(line, "loss=") {
			t.Error("gemma-3-4b should not show loss when no loss data exists")
		}
	}

	// Generation: golden should exist, expansion should show no data.
	if !strings.Contains(output, "golden") {
		t.Error("output missing golden generation row")
	}
	if !strings.Contains(output, "7000/15000") {
		t.Error("output missing 7000/15000")
	}
}

func TestPrintStatusMultipleModels(t *testing.T) {
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "run_id": "r1", "status": "complete", "iteration": float64(1000), "total_iters": float64(1000), "pct": float64(100.0)},
				{"model": "gemma-3-4b", "run_id": "r2", "status": "training", "iteration": float64(250), "total_iters": float64(500), "pct": float64(50.0)},
				{"model": "gemma-3-12b", "run_id": "r3", "status": "pending", "iteration": float64(0), "total_iters": float64(600), "pct": float64(0.0)},
				{"model": "gemma-3-27b", "run_id": "r4", "status": "queued", "iteration": float64(0), "total_iters": float64(400), "pct": float64(0.0)},
			}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			return []map[string]interface{}{
				{"model": "gemma-3-1b", "loss_type": "train", "loss": float64(1.2), "iteration": float64(1000), "tokens_per_sec": float64(500.0)},
				{"model": "gemma-3-4b", "loss_type": "train", "loss": float64(2.1), "iteration": float64(250), "tokens_per_sec": float64(300.0)},
			}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(15000), "target": float64(15000), "pct": float64(100.0)},
			}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu1", "completed": float64(10000), "target": float64(46331), "pct": float64(21.6)},
			}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// All four models should appear.
	for _, model := range []string{"gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"} {
		if !strings.Contains(output, model) {
			t.Errorf("output missing model %s", model)
		}
	}

	// Both generation types should appear.
	if !strings.Contains(output, "golden") {
		t.Error("output missing golden generation")
	}
	if !strings.Contains(output, "expansion") {
		t.Error("output missing expansion generation")
	}
	if !strings.Contains(output, "10000/46331") {
		t.Error("output missing expansion progress 10000/46331")
	}
}

func TestPrintStatusQueryErrorGraceful(t *testing.T) {
	// When InfluxDB returns errors, status should degrade gracefully
	// and show "(no data)" instead of failing.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("database error"))
	}))
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("expected graceful degradation, got error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "(no data)") {
		t.Errorf("expected '(no data)' in output, got:\n%s", output)
	}
}

func TestPrintStatusModelOrdering(t *testing.T) {
	// Models should appear in a deterministic order (sorted by name).
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			return []map[string]interface{}{
				{"model": "zeta-model", "run_id": "r1", "status": "training", "iteration": float64(10), "total_iters": float64(100), "pct": float64(10.0)},
				{"model": "alpha-model", "run_id": "r2", "status": "complete", "iteration": float64(100), "total_iters": float64(100), "pct": float64(100.0)},
				{"model": "mid-model", "run_id": "r3", "status": "pending", "iteration": float64(0), "total_iters": float64(50), "pct": float64(0.0)},
			}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			return []map[string]interface{}{}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Check that alpha appears before mid, and mid before zeta.
	alphaIdx := strings.Index(output, "alpha-model")
	midIdx := strings.Index(output, "mid-model")
	zetaIdx := strings.Index(output, "zeta-model")

	if alphaIdx == -1 || midIdx == -1 || zetaIdx == -1 {
		t.Fatalf("not all models found in output:\n%s", output)
	}
	if alphaIdx >= midIdx {
		t.Errorf("alpha-model (%d) should appear before mid-model (%d)", alphaIdx, midIdx)
	}
	if midIdx >= zetaIdx {
		t.Errorf("mid-model (%d) should appear before zeta-model (%d)", midIdx, zetaIdx)
	}
}

func TestPrintStatusMultipleWorkers(t *testing.T) {
	// Multiple workers for golden — should deduplicate keeping latest per worker.
	server := mockInfluxServer(t, func(q string) ([]map[string]interface{}, int) {
		switch {
		case strings.Contains(q, "training_status"):
			return []map[string]interface{}{}, http.StatusOK
		case strings.Contains(q, "training_loss"):
			return []map[string]interface{}{}, http.StatusOK
		case strings.Contains(q, "golden_gen_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(8000), "target": float64(15000), "pct": float64(53.3)},
				{"worker": "m3-gpu1", "completed": float64(7000), "target": float64(15000), "pct": float64(46.7)},
			}, http.StatusOK
		case strings.Contains(q, "expansion_progress"):
			return []map[string]interface{}{
				{"worker": "m3-gpu0", "completed": float64(5000), "target": float64(46331), "pct": float64(10.8)},
			}, http.StatusOK
		default:
			return nil, http.StatusOK
		}
	})
	defer server.Close()

	t.Setenv("INFLUX_TOKEN", "test-token")
	influx := NewInfluxClient(server.URL, "training")

	var buf bytes.Buffer
	err := printStatus(influx, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Both workers should appear since they're different workers.
	if !strings.Contains(output, "m3-gpu0") {
		t.Error("output missing worker m3-gpu0")
	}
	if !strings.Contains(output, "m3-gpu1") {
		t.Error("output missing worker m3-gpu1")
	}
}
