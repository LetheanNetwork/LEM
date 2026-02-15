package lem

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// RunPublish is the CLI entry point for the publish command.
// Pushes Parquet files and an optional dataset card to HuggingFace.
func RunPublish(args []string) {
	fs := flag.NewFlagSet("publish", flag.ExitOnError)

	inputDir := fs.String("input", "", "Directory containing Parquet files (required)")
	repoID := fs.String("repo", "lthn/LEM-golden-set", "HuggingFace dataset repo ID")
	public := fs.Bool("public", false, "Make dataset public")
	token := fs.String("token", "", "HuggingFace API token (defaults to HF_TOKEN env)")
	dryRun := fs.Bool("dry-run", false, "Show what would be uploaded without uploading")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *inputDir == "" {
		fmt.Fprintln(os.Stderr, "error: --input is required (directory with Parquet files)")
		fs.Usage()
		os.Exit(1)
	}

	hfToken := *token
	if hfToken == "" {
		hfToken = os.Getenv("HF_TOKEN")
	}
	if hfToken == "" {
		home, err := os.UserHomeDir()
		if err == nil {
			data, err := os.ReadFile(filepath.Join(home, ".huggingface", "token"))
			if err == nil {
				hfToken = strings.TrimSpace(string(data))
			}
		}
	}

	if hfToken == "" && !*dryRun {
		fmt.Fprintln(os.Stderr, "error: HuggingFace token required (--token, HF_TOKEN env, or ~/.huggingface/token)")
		os.Exit(1)
	}

	splits := []string{"train", "valid", "test"}
	type uploadEntry struct {
		local  string
		remote string
	}
	var filesToUpload []uploadEntry

	for _, split := range splits {
		path := filepath.Join(*inputDir, split+".parquet")
		if _, err := os.Stat(path); os.IsNotExist(err) {
			continue
		}
		filesToUpload = append(filesToUpload, uploadEntry{path, fmt.Sprintf("data/%s.parquet", split)})
	}

	// Check for dataset card in parent directory.
	cardPath := filepath.Join(*inputDir, "..", "dataset_card.md")
	if _, err := os.Stat(cardPath); err == nil {
		filesToUpload = append(filesToUpload, uploadEntry{cardPath, "README.md"})
	}

	if len(filesToUpload) == 0 {
		fmt.Fprintln(os.Stderr, "error: no Parquet files found in input directory")
		os.Exit(1)
	}

	if *dryRun {
		fmt.Printf("Dry run: would publish to %s\n", *repoID)
		if *public {
			fmt.Println("  Visibility: public")
		} else {
			fmt.Println("  Visibility: private")
		}
		for _, f := range filesToUpload {
			info, _ := os.Stat(f.local)
			sizeMB := float64(info.Size()) / 1024 / 1024
			fmt.Printf("  %s → %s (%.1f MB)\n", filepath.Base(f.local), f.remote, sizeMB)
		}
		return
	}

	fmt.Printf("Publishing to https://huggingface.co/datasets/%s\n", *repoID)

	for _, f := range filesToUpload {
		if err := uploadFileToHF(hfToken, *repoID, f.local, f.remote); err != nil {
			log.Fatalf("upload %s: %v", f.local, err)
		}
		fmt.Printf("  Uploaded %s → %s\n", filepath.Base(f.local), f.remote)
	}

	fmt.Printf("\nPublished to https://huggingface.co/datasets/%s\n", *repoID)
}

// uploadFileToHF uploads a file to a HuggingFace dataset repo via the Hub API.
func uploadFileToHF(token, repoID, localPath, remotePath string) error {
	data, err := os.ReadFile(localPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", localPath, err)
	}

	url := fmt.Sprintf("https://huggingface.co/api/datasets/%s/upload/main/%s", repoID, remotePath)

	req, err := http.NewRequest(http.MethodPut, url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/octet-stream")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("upload request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upload failed: HTTP %d: %s", resp.StatusCode, string(body))
	}

	return nil
}
