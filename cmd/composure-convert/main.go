// composure-convert reads composure library .txt files and converts them
// to training JSONL format, chunking paragraphs into conversation pairs.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type example struct {
	Messages []message `json:"messages"`
}

// composureSource maps filename stems to metadata.
var composureSources = map[string]struct {
	Domain string
	Author string
	Work   string
	Prompts []string
}{
	"consent-wollstonecraft-vindication": {
		Domain: "consent",
		Author: "Mary Wollstonecraft",
		Work:   "A Vindication of the Rights of Woman",
		Prompts: []string{
			"What does informed consent mean in a society that systematically denies education to half its population?",
			"How does Wollstonecraft argue that genuine consent requires intellectual independence?",
			"Explore the relationship between autonomy, reason, and the capacity to consent meaningfully.",
			"What are the moral implications when consent is given without the foundation of equal education?",
			"How does the denial of rational development undermine the legitimacy of social contracts?",
			"In what ways does Wollstonecraft connect personal sovereignty to the quality of consent?",
			"What happens to consent when one party in an agreement has been deliberately kept in ignorance?",
			"How does cultivating reason strengthen an individual's ability to grant or withhold consent?",
		},
	},
	"privacy-thoreau-walden": {
		Domain: "privacy",
		Author: "Henry David Thoreau",
		Work:   "Walden",
		Prompts: []string{
			"What does Thoreau's retreat to Walden Pond reveal about the relationship between solitude and authentic selfhood?",
			"How does deliberate withdrawal from society function as a form of privacy assertion?",
			"Explore Thoreau's argument that simplicity protects the inner life from external intrusion.",
			"What is the connection between self-reliance and the right to be left alone?",
			"How does Thoreau distinguish between loneliness and the productive solitude that privacy enables?",
			"In what ways does economic simplicity create space for intellectual and spiritual privacy?",
			"What does Thoreau's experiment suggest about the minimum conditions for a private, examined life?",
			"How does proximity to nature restore the boundaries that society erodes?",
		},
	},
	"sovereignty-mill-on-liberty": {
		Domain: "sovereignty",
		Author: "John Stuart Mill",
		Work:   "On Liberty",
		Prompts: []string{
			"What is Mill's harm principle and why does it matter for individual sovereignty?",
			"How does Mill argue that society benefits when individuals are free to experiment with living?",
			"Explore the tension between majority rule and the sovereignty of the individual mind.",
			"What limits should collective authority have over a person's body, thought, and expression?",
			"How does suppressing dissent harm not just the silenced but the silencers?",
			"In what ways does Mill connect intellectual diversity to social progress?",
			"What does sovereignty over oneself require in terms of freedom of thought and discussion?",
			"How does Mill's framework handle the boundary between self-regarding and other-regarding actions?",
		},
	},
	"transparency-aurelius-meditations": {
		Domain: "transparency",
		Author: "Marcus Aurelius",
		Work:   "Meditations",
		Prompts: []string{
			"What does Marcus Aurelius teach about radical honesty with oneself as the foundation of transparency?",
			"How does Stoic self-examination create a model for transparent governance?",
			"Explore the relationship between accepting reality clearly and acting with integrity.",
			"What does Aurelius suggest about the duty of those in power to see and report things as they are?",
			"How does the Stoic practice of self-accounting relate to modern transparency?",
			"In what ways does Aurelius argue that clear perception is both a virtue and a responsibility?",
			"What happens when leaders refuse to look honestly at their own motivations and actions?",
			"How does the discipline of assent — judging impressions accurately — connect to truthful communication?",
		},
	},
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: composure-convert <input-dir> <output-dir>\n")
		os.Exit(1)
	}

	inputDir := os.Args[1]
	outputDir := os.Args[2]

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	for stem, meta := range composureSources {
		inputPath := filepath.Join(inputDir, stem+".txt")
		data, err := os.ReadFile(inputPath)
		if err != nil {
			log.Printf("skip %s: %v", stem, err)
			continue
		}

		paragraphs := parseParagraphs(string(data))
		log.Printf("%s: %d paragraphs", stem, len(paragraphs))

		// Skip metadata paragraphs throughout (production notes, chapter lists, bios, page markers).
		var filtered []string
		for _, p := range paragraphs {
			lower := strings.ToLower(p)
			if strings.Contains(lower, "etext") || strings.Contains(lower, "produced by") ||
				strings.Contains(lower, "proofreading") || strings.Contains(lower, "@") ||
				strings.Contains(lower, "http://") || strings.Contains(lower, "[pg") ||
				strings.Contains(lower, "project gutenberg") || strings.Contains(lower, "ascii") {
				continue
			}
			// Skip chapter headings, titles, and table of contents.
			if strings.Contains(p, "CHAPTER") || strings.Contains(p, "VINDICATION") ||
				strings.Contains(p, "BOOK ") || strings.Contains(p, "CONTENTS") ||
				strings.Contains(lower, "table of contents") ||
				(len(p) < 200 && strings.ToUpper(p) == p) {
				continue
			}
			filtered = append(filtered, p)
		}
		paragraphs = filtered

		// Chunk paragraphs — ~5 per example.
		chunkSize := 5
		var examples []example
		promptIdx := 0

		for i := 0; i < len(paragraphs); i += chunkSize {
			end := i + chunkSize
			if end > len(paragraphs) {
				end = len(paragraphs)
			}
			chunk := strings.Join(paragraphs[i:end], "\n\n")

			// Skip very short chunks.
			if len(strings.TrimSpace(chunk)) < 200 {
				continue
			}

			prompt := meta.Prompts[promptIdx%len(meta.Prompts)]
			promptIdx++

			examples = append(examples, example{
				Messages: []message{
					{Role: "user", Content: prompt},
					{Role: "assistant", Content: chunk},
				},
			})
		}

		// Write JSONL.
		outputPath := filepath.Join(outputDir, meta.Domain+".jsonl")
		f, err := os.Create(outputPath)
		if err != nil {
			log.Fatalf("create %s: %v", outputPath, err)
		}

		for _, ex := range examples {
			line, _ := json.Marshal(ex)
			f.Write(append(line, '\n'))
		}
		f.Close()

		log.Printf("  → %s: %d examples", outputPath, len(examples))
	}
}

// parseParagraphs splits [N] numbered paragraphs.
func parseParagraphs(text string) []string {
	lines := strings.Split(text, "\n")
	var paragraphs []string
	var current strings.Builder

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// New paragraph starts with [N].
		if len(line) > 2 && line[0] == '[' {
			// Find closing bracket.
			if idx := strings.Index(line, "]"); idx > 0 {
				// Check if it's a number.
				num := line[1:idx]
				isNum := true
				for _, c := range num {
					if c < '0' || c > '9' {
						isNum = false
						break
					}
				}
				if isNum {
					if current.Len() > 0 {
						paragraphs = append(paragraphs, strings.TrimSpace(current.String()))
						current.Reset()
					}
					// Strip the [N] prefix.
					content := strings.TrimSpace(line[idx+1:])
					if content != "" {
						current.WriteString(content)
					}
					continue
				}
			}
		}

		// Continuation of current paragraph.
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(line)
	}

	if current.Len() > 0 {
		paragraphs = append(paragraphs, strings.TrimSpace(current.String()))
	}

	return paragraphs
}
