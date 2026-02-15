package lem

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// agentConfig holds scoring agent configuration.
type agentConfig struct {
	m3Host        string
	m3User        string
	m3SSHKey      string
	m3AdapterBase string
	influxURL     string
	influxDB      string
	apiURL        string
	model         string
	baseModel     string
	pollInterval  int
	workDir       string
	oneShot       bool
	dryRun        bool
}

// checkpoint represents a discovered adapter checkpoint on M3.
type checkpoint struct {
	RemoteDir string
	Filename  string
	Dirname   string
	Iteration int
	ModelTag  string
	Label     string
	RunID     string
}

// probeResult holds the result of running all probes against a checkpoint.
type probeResult struct {
	Accuracy   float64                       `json:"accuracy"`
	Correct    int                           `json:"correct"`
	Total      int                           `json:"total"`
	ByCategory map[string]categoryResult     `json:"by_category"`
	Probes     map[string]singleProbeResult  `json:"probes"`
}

type categoryResult struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

type singleProbeResult struct {
	Passed   bool   `json:"passed"`
	Response string `json:"response"`
}

// bufferEntry is a JSONL-buffered result for when InfluxDB is down.
type bufferEntry struct {
	Checkpoint checkpoint  `json:"checkpoint"`
	Results    probeResult `json:"results"`
	Timestamp  string      `json:"timestamp"`
}

// RunAgent is the CLI entry point for the agent command.
// Polls M3 for unscored LoRA checkpoints, converts MLX → PEFT,
// runs 23 capability probes via an OpenAI-compatible API, and
// pushes results to InfluxDB.
func RunAgent(args []string) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)

	cfg := &agentConfig{}
	fs.StringVar(&cfg.m3Host, "m3-host", envOr("M3_HOST", "10.69.69.108"), "M3 host address")
	fs.StringVar(&cfg.m3User, "m3-user", envOr("M3_USER", "claude"), "M3 SSH user")
	fs.StringVar(&cfg.m3SSHKey, "m3-ssh-key", envOr("M3_SSH_KEY", expandHome("~/.ssh/id_ed25519")), "SSH key for M3")
	fs.StringVar(&cfg.m3AdapterBase, "m3-adapter-base", envOr("M3_ADAPTER_BASE", "/Volumes/Data/lem"), "Adapter base dir on M3")
	fs.StringVar(&cfg.influxURL, "influx", envOr("INFLUX_URL", "http://10.69.69.165:8181"), "InfluxDB URL")
	fs.StringVar(&cfg.influxDB, "influx-db", envOr("INFLUX_DB", "training"), "InfluxDB database")
	fs.StringVar(&cfg.apiURL, "api-url", envOr("LEM_API_URL", "http://localhost:8080"), "OpenAI-compatible inference API URL")
	fs.StringVar(&cfg.model, "model", envOr("LEM_MODEL", ""), "Model name for API (overrides auto-detect)")
	fs.StringVar(&cfg.baseModel, "base-model", envOr("BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"), "HuggingFace base model ID")
	fs.IntVar(&cfg.pollInterval, "poll", intEnvOr("POLL_INTERVAL", 300), "Poll interval in seconds")
	fs.StringVar(&cfg.workDir, "work-dir", envOr("WORK_DIR", "/tmp/scoring-agent"), "Working directory for adapters")
	fs.BoolVar(&cfg.oneShot, "one-shot", false, "Process one checkpoint and exit")
	fs.BoolVar(&cfg.dryRun, "dry-run", false, "Discover and plan but don't execute")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	runAgentLoop(cfg)
}

func runAgentLoop(cfg *agentConfig) {
	log.Println(strings.Repeat("=", 60))
	log.Println("ROCm Scoring Agent — Go Edition")
	log.Printf("M3: %s@%s", cfg.m3User, cfg.m3Host)
	log.Printf("Inference API: %s", cfg.apiURL)
	log.Printf("InfluxDB: %s/%s", cfg.influxURL, cfg.influxDB)
	log.Printf("Poll interval: %ds", cfg.pollInterval)
	log.Println(strings.Repeat("=", 60))

	influx := NewInfluxClient(cfg.influxURL, cfg.influxDB)
	os.MkdirAll(cfg.workDir, 0755)

	for {
		// Replay any buffered results.
		replayInfluxBuffer(cfg.workDir, influx)

		// Discover checkpoints on M3.
		log.Println("Discovering checkpoints on M3...")
		checkpoints, err := discoverCheckpoints(cfg)
		if err != nil {
			log.Printf("Discovery failed: %v", err)
			sleepOrExit(cfg)
			continue
		}
		log.Printf("Found %d total checkpoints", len(checkpoints))

		// Check what is already scored.
		scored, err := getScoredLabels(influx)
		if err != nil {
			log.Printf("InfluxDB query failed: %v", err)
		}
		log.Printf("Already scored: %d (run_id, label) pairs", len(scored))

		// Find unscored work.
		unscored := findUnscored(checkpoints, scored)
		log.Printf("Unscored: %d checkpoints", len(unscored))

		if len(unscored) == 0 {
			log.Printf("Nothing to score. Sleeping %ds...", cfg.pollInterval)
			if cfg.oneShot {
				return
			}
			time.Sleep(time.Duration(cfg.pollInterval) * time.Second)
			continue
		}

		target := unscored[0]
		log.Printf("Grabbed: %s (%s)", target.Label, target.Dirname)

		if cfg.dryRun {
			log.Printf("[DRY RUN] Would process: %s/%s", target.Dirname, target.Filename)
			for _, u := range unscored[1:] {
				log.Printf("[DRY RUN] Queued: %s/%s", u.Dirname, u.Filename)
			}
			return
		}

		if err := processOne(cfg, influx, target); err != nil {
			log.Printf("Error processing %s: %v", target.Label, err)
		}

		if cfg.oneShot {
			return
		}

		time.Sleep(5 * time.Second)
	}
}

// discoverCheckpoints lists all adapter directories and checkpoint files on M3 via SSH.
func discoverCheckpoints(cfg *agentConfig) ([]checkpoint, error) {
	out, err := sshCommand(cfg, fmt.Sprintf("ls -d %s/adapters-deepseek-r1-7b* 2>/dev/null", cfg.m3AdapterBase))
	if err != nil {
		return nil, fmt.Errorf("list adapter dirs: %w", err)
	}

	var checkpoints []checkpoint
	iterRe := regexp.MustCompile(`(\d+)`)

	for _, dirpath := range strings.Split(strings.TrimSpace(out), "\n") {
		if dirpath == "" {
			continue
		}
		dirname := filepath.Base(dirpath)

		// List checkpoint safetensors files.
		filesOut, err := sshCommand(cfg, fmt.Sprintf("ls %s/*_adapters.safetensors 2>/dev/null", dirpath))
		if err != nil {
			continue
		}

		for _, filepath := range strings.Split(strings.TrimSpace(filesOut), "\n") {
			if filepath == "" {
				continue
			}
			filename := fileBase(filepath)

			match := iterRe.FindStringSubmatch(filename)
			if len(match) < 2 {
				continue
			}
			iteration := 0
			fmt.Sscanf(match[1], "%d", &iteration)

			modelTag, labelPrefix, stem := adapterMeta(dirname)
			label := fmt.Sprintf("%s @%s", labelPrefix, match[1])
			runID := fmt.Sprintf("%s-capability-auto", stem)

			checkpoints = append(checkpoints, checkpoint{
				RemoteDir: dirpath,
				Filename:  filename,
				Dirname:   dirname,
				Iteration: iteration,
				ModelTag:  modelTag,
				Label:     label,
				RunID:     runID,
			})
		}
	}

	return checkpoints, nil
}

// adapterMeta maps an adapter directory name to (model_tag, label_prefix, run_id_stem).
func adapterMeta(dirname string) (string, string, string) {
	name := strings.TrimPrefix(dirname, "adapters-deepseek-r1-7b")
	name = strings.TrimLeft(name, "-")
	if name == "" {
		name = "base"
	}

	shortNames := map[string]string{
		"sovereignty":    "R1-sov",
		"russian":        "R1-rus",
		"composure":      "R1-comp",
		"sandwich":       "R1-sand",
		"sandwich-watts": "R1-sw",
		"western":        "R1-west",
		"western-fresh":  "R1-wf",
		"base":           "R1-base",
	}

	short, ok := shortNames[name]
	if !ok {
		if len(name) > 4 {
			short = "R1-" + name[:4]
		} else {
			short = "R1-" + name
		}
	}

	stem := "r1-" + name
	if name == "base" {
		stem = "r1-base"
	}

	return "deepseek-r1-7b", short, stem
}

// getScoredLabels returns all (run_id, label) pairs already scored in InfluxDB.
func getScoredLabels(influx *InfluxClient) (map[[2]string]bool, error) {
	rows, err := influx.QuerySQL("SELECT DISTINCT run_id, label FROM capability_score")
	if err != nil {
		return nil, err
	}

	scored := make(map[[2]string]bool)
	for _, row := range rows {
		runID, _ := row["run_id"].(string)
		label, _ := row["label"].(string)
		if runID != "" && label != "" {
			scored[[2]string{runID, label}] = true
		}
	}
	return scored, nil
}

// findUnscored filters checkpoints to only unscored ones, sorted by (dirname, iteration).
func findUnscored(checkpoints []checkpoint, scored map[[2]string]bool) []checkpoint {
	var unscored []checkpoint
	for _, c := range checkpoints {
		if !scored[[2]string{c.RunID, c.Label}] {
			unscored = append(unscored, c)
		}
	}
	sort.Slice(unscored, func(i, j int) bool {
		if unscored[i].Dirname != unscored[j].Dirname {
			return unscored[i].Dirname < unscored[j].Dirname
		}
		return unscored[i].Iteration < unscored[j].Iteration
	})
	return unscored
}

// processOne fetches, converts, scores, and pushes one checkpoint.
func processOne(cfg *agentConfig, influx *InfluxClient, cp checkpoint) error {
	log.Println(strings.Repeat("=", 60))
	log.Printf("Processing: %s / %s", cp.Dirname, cp.Filename)
	log.Println(strings.Repeat("=", 60))

	localAdapterDir := filepath.Join(cfg.workDir, cp.Dirname)
	os.MkdirAll(localAdapterDir, 0755)

	localSF := filepath.Join(localAdapterDir, cp.Filename)
	localCfg := filepath.Join(localAdapterDir, "adapter_config.json")

	// Cleanup on exit.
	defer func() {
		os.Remove(localSF)
		os.Remove(localCfg)
		peftDir := filepath.Join(cfg.workDir, fmt.Sprintf("peft_%07d", cp.Iteration))
		os.RemoveAll(peftDir)
	}()

	// Fetch adapter + config from M3.
	log.Println("Fetching adapter from M3...")
	remoteSF := fmt.Sprintf("%s/%s", cp.RemoteDir, cp.Filename)
	remoteCfg := fmt.Sprintf("%s/adapter_config.json", cp.RemoteDir)

	if err := scpFrom(cfg, remoteSF, localSF); err != nil {
		return fmt.Errorf("scp safetensors: %w", err)
	}
	if err := scpFrom(cfg, remoteCfg, localCfg); err != nil {
		return fmt.Errorf("scp config: %w", err)
	}

	// Convert MLX to PEFT format.
	log.Println("Converting MLX to PEFT format...")
	peftDir := filepath.Join(cfg.workDir, fmt.Sprintf("peft_%07d", cp.Iteration))
	if err := convertMLXtoPEFT(localAdapterDir, cp.Filename, peftDir, cfg.baseModel); err != nil {
		return fmt.Errorf("convert adapter: %w", err)
	}

	// Run 23 capability probes via API.
	log.Println("Running 23 capability probes...")
	modelName := cfg.model
	if modelName == "" {
		modelName = cp.ModelTag
	}
	client := NewClient(cfg.apiURL, modelName)
	client.MaxTokens = 500

	results := runCapabilityProbes(client)

	log.Printf("Result: %s -- %.1f%% (%d/%d)",
		cp.Label, results.Accuracy, results.Correct, results.Total)

	// Push to InfluxDB (buffer on failure).
	if err := pushCapabilityResults(influx, cp, results); err != nil {
		log.Printf("InfluxDB push failed, buffering: %v", err)
		bufferInfluxResult(cfg.workDir, cp, results)
	}

	return nil
}

// runCapabilityProbes runs all 23 probes against the inference API.
func runCapabilityProbes(client *Client) probeResult {
	results := probeResult{
		ByCategory: make(map[string]categoryResult),
		Probes:     make(map[string]singleProbeResult),
	}

	correct := 0
	total := 0

	for _, probe := range CapabilityProbes {
		response, err := client.ChatWithTemp(probe.Prompt, 0.1)
		if err != nil {
			log.Printf("  [%s] ERROR: %v", probe.ID, err)
			results.Probes[probe.ID] = singleProbeResult{Passed: false, Response: err.Error()}
			total++
			cat := results.ByCategory[probe.Category]
			cat.Total++
			results.ByCategory[probe.Category] = cat
			continue
		}

		// Strip <think> blocks from DeepSeek R1 responses.
		clean := StripThinkBlocks(response)

		passed := probe.Check(clean)
		total++
		if passed {
			correct++
		}

		cat := results.ByCategory[probe.Category]
		cat.Total++
		if passed {
			cat.Correct++
		}
		results.ByCategory[probe.Category] = cat

		// Truncate response for storage.
		stored := clean
		if len(stored) > 300 {
			stored = stored[:300]
		}
		results.Probes[probe.ID] = singleProbeResult{Passed: passed, Response: stored}

		status := "FAIL"
		if passed {
			status = "PASS"
		}
		log.Printf("  [%s] %s (expected: %s)", probe.ID, status, probe.Answer)
	}

	if total > 0 {
		results.Accuracy = float64(correct) / float64(total) * 100
	}
	results.Correct = correct
	results.Total = total

	return results
}

// pushCapabilityResults writes scoring results to InfluxDB as line protocol.
func pushCapabilityResults(influx *InfluxClient, cp checkpoint, results probeResult) error {
	// Base timestamp: 2026-02-15T00:00:00Z = 1739577600
	const baseTS int64 = 1739577600

	var lines []string

	// Overall score.
	ts := (baseTS + int64(cp.Iteration)*1000 + 0) * 1_000_000_000
	lines = append(lines, fmt.Sprintf(
		"capability_score,model=%s,run_id=%s,label=%s,category=overall accuracy=%.1f,correct=%di,total=%di,iteration=%di %d",
		escapeLp(cp.ModelTag), escapeLp(cp.RunID), escapeLp(cp.Label),
		results.Accuracy, results.Correct, results.Total, cp.Iteration, ts,
	))

	// Per-category scores (sorted for deterministic output).
	cats := make([]string, 0, len(results.ByCategory))
	for cat := range results.ByCategory {
		cats = append(cats, cat)
	}
	sort.Strings(cats)

	for i, cat := range cats {
		data := results.ByCategory[cat]
		catAcc := 0.0
		if data.Total > 0 {
			catAcc = float64(data.Correct) / float64(data.Total) * 100
		}
		ts := (baseTS + int64(cp.Iteration)*1000 + int64(i+1)) * 1_000_000_000
		lines = append(lines, fmt.Sprintf(
			"capability_score,model=%s,run_id=%s,label=%s,category=%s accuracy=%.1f,correct=%di,total=%di,iteration=%di %d",
			escapeLp(cp.ModelTag), escapeLp(cp.RunID), escapeLp(cp.Label), escapeLp(cat),
			catAcc, data.Correct, data.Total, cp.Iteration, ts,
		))
	}

	// Per-probe results (sorted).
	probeIDs := make([]string, 0, len(results.Probes))
	for id := range results.Probes {
		probeIDs = append(probeIDs, id)
	}
	sort.Strings(probeIDs)

	for j, probeID := range probeIDs {
		probeRes := results.Probes[probeID]
		passedInt := 0
		if probeRes.Passed {
			passedInt = 1
		}
		ts := (baseTS + int64(cp.Iteration)*1000 + int64(j+100)) * 1_000_000_000
		lines = append(lines, fmt.Sprintf(
			"probe_score,model=%s,run_id=%s,label=%s,probe_id=%s passed=%di,iteration=%di %d",
			escapeLp(cp.ModelTag), escapeLp(cp.RunID), escapeLp(cp.Label), escapeLp(probeID),
			passedInt, cp.Iteration, ts,
		))
	}

	if err := influx.WriteLp(lines); err != nil {
		return err
	}
	log.Printf("Pushed %d points to InfluxDB for %s", len(lines), cp.Label)
	return nil
}

// bufferInfluxResult saves results to a local JSONL file when InfluxDB is down.
func bufferInfluxResult(workDir string, cp checkpoint, results probeResult) {
	bufPath := filepath.Join(workDir, "influx_buffer.jsonl")
	f, err := os.OpenFile(bufPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Cannot open buffer file: %v", err)
		return
	}
	defer f.Close()

	entry := bufferEntry{
		Checkpoint: cp,
		Results:    results,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}
	data, _ := json.Marshal(entry)
	f.Write(append(data, '\n'))
	log.Printf("Buffered results to %s", bufPath)
}

// replayInfluxBuffer retries pushing buffered results to InfluxDB.
func replayInfluxBuffer(workDir string, influx *InfluxClient) {
	bufPath := filepath.Join(workDir, "influx_buffer.jsonl")
	data, err := os.ReadFile(bufPath)
	if err != nil {
		return // No buffer file.
	}

	var remaining []string
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var entry bufferEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			remaining = append(remaining, line)
			continue
		}
		if err := pushCapabilityResults(influx, entry.Checkpoint, entry.Results); err != nil {
			remaining = append(remaining, line)
		} else {
			log.Printf("Replayed buffered result: %s", entry.Checkpoint.Label)
		}
	}

	if len(remaining) > 0 {
		os.WriteFile(bufPath, []byte(strings.Join(remaining, "\n")+"\n"), 0644)
	} else {
		os.Remove(bufPath)
		log.Println("Buffer fully replayed and cleared")
	}
}

// sshCommand executes a command on M3 via SSH.
func sshCommand(cfg *agentConfig, cmd string) (string, error) {
	sshArgs := []string{
		"-o", "ConnectTimeout=10",
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=no",
		"-i", cfg.m3SSHKey,
		fmt.Sprintf("%s@%s", cfg.m3User, cfg.m3Host),
		cmd,
	}
	result, err := exec.Command("ssh", sshArgs...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("ssh %q: %w: %s", cmd, err, strings.TrimSpace(string(result)))
	}
	return string(result), nil
}

// scpFrom copies a file from M3 to a local path.
func scpFrom(cfg *agentConfig, remotePath, localPath string) error {
	os.MkdirAll(filepath.Dir(localPath), 0755)
	scpArgs := []string{
		"-o", "ConnectTimeout=10",
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=no",
		"-i", cfg.m3SSHKey,
		fmt.Sprintf("%s@%s:%s", cfg.m3User, cfg.m3Host, remotePath),
		localPath,
	}
	result, err := exec.Command("scp", scpArgs...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scp %s: %w: %s", remotePath, err, strings.TrimSpace(string(result)))
	}
	return nil
}

// fileBase returns the last component of a path (works for both / and \).
func fileBase(path string) string {
	if i := strings.LastIndexAny(path, "/\\"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func sleepOrExit(cfg *agentConfig) {
	if cfg.oneShot {
		return
	}
	time.Sleep(time.Duration(cfg.pollInterval) * time.Second)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func intEnvOr(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var n int
	fmt.Sscanf(v, "%d", &n)
	if n == 0 {
		return fallback
	}
	return n
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
