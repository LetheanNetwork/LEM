package lem

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// RunConsolidate is the CLI entry point for the consolidate command.
// Pulls all worker JSONLs from M3, merges them, deduplicates on idx,
// and writes a single merged file.
func RunConsolidate(args []string) {
	fs := flag.NewFlagSet("consolidate", flag.ExitOnError)
	remoteHost := fs.String("host", "m3", "SSH host for remote files")
	remotePath := fs.String("remote", "/Volumes/Data/lem/responses", "Remote directory for JSONL files")
	pattern := fs.String("pattern", "gold*.jsonl", "File glob pattern")
	outputDir := fs.String("output", "", "Output directory (defaults to ./responses)")
	merged := fs.String("merged", "", "Merged output file (defaults to gold-merged.jsonl in output dir)")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *outputDir == "" {
		*outputDir = "responses"
	}
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// List remote files.
	fmt.Println("Pulling responses from remote...")
	listCmd := exec.Command("ssh", *remoteHost, fmt.Sprintf("ls %s/%s", *remotePath, *pattern))
	listOutput, err := listCmd.Output()
	if err != nil {
		log.Fatalf("list remote files: %v", err)
	}

	remoteFiles := strings.Split(strings.TrimSpace(string(listOutput)), "\n")
	var validFiles []string
	for _, f := range remoteFiles {
		f = strings.TrimSpace(f)
		if f != "" {
			validFiles = append(validFiles, f)
		}
	}
	fmt.Printf("  Found %d JSONL files on %s\n", len(validFiles), *remoteHost)

	// Pull files.
	for _, rf := range validFiles {
		local := filepath.Join(*outputDir, filepath.Base(rf))
		scpCmd := exec.Command("scp", fmt.Sprintf("%s:%s", *remoteHost, rf), local)
		if err := scpCmd.Run(); err != nil {
			log.Printf("warning: failed to pull %s: %v", rf, err)
			continue
		}

		// Count lines.
		f, err := os.Open(local)
		if err != nil {
			continue
		}
		lines := 0
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			lines++
		}
		f.Close()
		fmt.Printf("  %s: %d records\n", filepath.Base(rf), lines)
	}

	// Merge and deduplicate on idx.
	seen := make(map[int]json.RawMessage)
	skipped := 0

	matches, _ := filepath.Glob(filepath.Join(*outputDir, *pattern))
	sort.Strings(matches)

	for _, local := range matches {
		f, err := os.Open(local)
		if err != nil {
			continue
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			var rec struct {
				Idx *int `json:"idx"`
			}
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				skipped++
				continue
			}
			if rec.Idx == nil {
				skipped++
				continue
			}
			if _, exists := seen[*rec.Idx]; !exists {
				seen[*rec.Idx] = json.RawMessage(line)
			}
		}
		f.Close()
	}

	if skipped > 0 {
		fmt.Printf("  Skipped %d records without idx\n", skipped)
	}

	// Sort by idx and write merged file.
	if *merged == "" {
		*merged = filepath.Join(*outputDir, "..", "gold-merged.jsonl")
	}

	idxs := make([]int, 0, len(seen))
	for idx := range seen {
		idxs = append(idxs, idx)
	}
	sort.Ints(idxs)

	f, err := os.Create(*merged)
	if err != nil {
		log.Fatalf("create merged file: %v", err)
	}
	for _, idx := range idxs {
		f.Write(seen[idx])
		f.WriteString("\n")
	}
	f.Close()

	fmt.Printf("\nMerged: %d unique examples → %s\n", len(seen), *merged)
}
