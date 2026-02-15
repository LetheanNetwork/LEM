package lem

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// RunConvert is the CLI entry point for the convert command.
// Converts MLX LoRA adapters to HuggingFace PEFT format:
//   - Key renaming: model.layers.N.module.lora_a → base_model.model.model.layers.N.module.lora_A.default.weight
//   - Transpose: MLX (in, rank) → PEFT (rank, in)
//   - Config generation: adapter_config.json with lora_alpha = scale × rank
func RunConvert(args []string) {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)

	safetensorsPath := fs.String("input", "", "Path to MLX .safetensors file (required)")
	configPath := fs.String("config", "", "Path to MLX adapter_config.json (required)")
	outputDir := fs.String("output", "./peft_output", "Output directory for PEFT adapter")
	baseModel := fs.String("base-model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "HuggingFace base model ID")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *safetensorsPath == "" || *configPath == "" {
		fmt.Fprintln(os.Stderr, "error: --input and --config are required")
		fs.Usage()
		os.Exit(1)
	}

	if err := convertMLXtoPEFT(*safetensorsPath, *configPath, *outputDir, *baseModel); err != nil {
		log.Fatalf("convert: %v", err)
	}

	fmt.Printf("Converted to: %s\n", *outputDir)
}

var (
	loraARe  = regexp.MustCompile(`\.lora_a$`)
	loraBRe  = regexp.MustCompile(`\.lora_b$`)
	layerRe  = regexp.MustCompile(`layers\.(\d+)`)
	moduleRe = regexp.MustCompile(`model\.layers\.\d+\.(.*?)\.lora_[ab]$`)
)

// renameMLXKey converts an MLX tensor key to PEFT format.
func renameMLXKey(mlxKey string) string {
	key := mlxKey
	key = loraARe.ReplaceAllString(key, ".lora_A.default.weight")
	key = loraBRe.ReplaceAllString(key, ".lora_B.default.weight")
	key = "base_model.model." + key
	return key
}

// safetensorsHeader represents the header of a safetensors file.
type safetensorsHeader struct {
	Metadata map[string]string                `json:"__metadata__,omitempty"`
	Tensors  map[string]safetensorsTensorInfo `json:"-"`
}

type safetensorsTensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// readSafetensors reads a safetensors file and returns tensor name→data+info pairs.
func readSafetensors(path string) (map[string]safetensorsTensorInfo, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read file: %w", err)
	}

	if len(data) < 8 {
		return nil, nil, fmt.Errorf("file too small")
	}

	headerSize := int(binary.LittleEndian.Uint64(data[:8]))
	if 8+headerSize > len(data) {
		return nil, nil, fmt.Errorf("invalid header size %d", headerSize)
	}

	headerJSON := data[8 : 8+headerSize]
	tensorData := data[8+headerSize:]

	// Parse header as a generic map since tensors are top-level keys.
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &rawHeader); err != nil {
		return nil, nil, fmt.Errorf("parse header: %w", err)
	}

	tensors := make(map[string]safetensorsTensorInfo)
	for key, raw := range rawHeader {
		if key == "__metadata__" {
			continue
		}
		var info safetensorsTensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, nil, fmt.Errorf("parse tensor %s: %w", key, err)
		}
		tensors[key] = info
	}

	return tensors, tensorData, nil
}

// getTensorData extracts raw bytes for a tensor from the data section.
func getTensorData(info safetensorsTensorInfo, allData []byte) []byte {
	return allData[info.DataOffsets[0]:info.DataOffsets[1]]
}

// transposeFloat32 transposes a (rows, cols) float32 matrix to (cols, rows).
func transposeFloat32(data []byte, rows, cols int) []byte {
	if len(data) != rows*cols*4 {
		return data // size mismatch, return as-is
	}

	result := make([]byte, len(data))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			srcOff := (r*cols + c) * 4
			dstOff := (c*rows + r) * 4
			copy(result[dstOff:dstOff+4], data[srcOff:srcOff+4])
		}
	}
	return result
}

// transposeFloat16 transposes a (rows, cols) float16 matrix to (cols, rows).
func transposeFloat16(data []byte, rows, cols int) []byte {
	if len(data) != rows*cols*2 {
		return data
	}

	result := make([]byte, len(data))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			srcOff := (r*cols + c) * 2
			dstOff := (c*rows + r) * 2
			copy(result[dstOff:dstOff+2], data[srcOff:srcOff+2])
		}
	}
	return result
}

// transposeBFloat16 transposes a (rows, cols) bfloat16 matrix to (cols, rows).
func transposeBFloat16(data []byte, rows, cols int) []byte {
	return transposeFloat16(data, rows, cols) // same element size
}

// writeSafetensors writes tensors to a safetensors file.
func writeSafetensors(path string, tensors map[string]safetensorsTensorInfo, tensorData map[string][]byte) error {
	// Sort keys for deterministic output.
	keys := make([]string, 0, len(tensors))
	for k := range tensors {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Compute offsets.
	offset := 0
	updatedTensors := make(map[string]safetensorsTensorInfo)
	for _, k := range keys {
		info := tensors[k]
		data := tensorData[k]
		info.DataOffsets = [2]int{offset, offset + len(data)}
		updatedTensors[k] = info
		offset += len(data)
	}

	// Build header JSON.
	headerMap := make(map[string]interface{})
	for k, info := range updatedTensors {
		headerMap[k] = info
	}

	headerJSON, err := json.Marshal(headerMap)
	if err != nil {
		return fmt.Errorf("marshal header: %w", err)
	}

	// Write file: 8-byte header size + header JSON + tensor data.
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	headerSizeBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBuf, uint64(len(headerJSON)))

	if _, err := f.Write(headerSizeBuf); err != nil {
		return err
	}
	if _, err := f.Write(headerJSON); err != nil {
		return err
	}

	for _, k := range keys {
		if _, err := f.Write(tensorData[k]); err != nil {
			return err
		}
	}

	return nil
}

// convertMLXtoPEFT converts an MLX LoRA adapter to PEFT format.
func convertMLXtoPEFT(safetensorsPath, configPath, outputDir, baseModelName string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Read MLX tensors.
	tensors, tensorData, err := readSafetensors(safetensorsPath)
	if err != nil {
		return fmt.Errorf("read safetensors: %w", err)
	}
	log.Printf("loaded %d tensors from %s", len(tensors), safetensorsPath)

	// Rename and transpose tensors.
	peftTensors := make(map[string]safetensorsTensorInfo)
	peftData := make(map[string][]byte)

	for mlxKey, info := range tensors {
		peftKey := renameMLXKey(mlxKey)
		data := getTensorData(info, tensorData)

		// Transpose: swap shape and transpose data.
		if len(info.Shape) == 2 {
			rows, cols := info.Shape[0], info.Shape[1]

			switch info.Dtype {
			case "F32":
				data = transposeFloat32(data, rows, cols)
			case "F16":
				data = transposeFloat16(data, rows, cols)
			case "BF16":
				data = transposeBFloat16(data, rows, cols)
			}

			info.Shape = []int{cols, rows}
		}

		peftTensors[peftKey] = info
		peftData[peftKey] = data
	}

	// Write PEFT safetensors.
	outSafetensors := filepath.Join(outputDir, "adapter_model.safetensors")
	if err := writeSafetensors(outSafetensors, peftTensors, peftData); err != nil {
		return fmt.Errorf("write safetensors: %w", err)
	}

	// Read MLX config for LoRA parameters.
	cfgData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}

	var mlxConfig struct {
		LoraParameters struct {
			Rank    int     `json:"rank"`
			Scale   float64 `json:"scale"`
			Dropout float64 `json:"dropout"`
		} `json:"lora_parameters"`
	}
	if err := json.Unmarshal(cfgData, &mlxConfig); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	rank := mlxConfig.LoraParameters.Rank
	if rank == 0 {
		rank = 8
	}
	scale := mlxConfig.LoraParameters.Scale
	if scale == 0 {
		scale = 20.0
	}

	// Determine target modules from tensor keys.
	modules := make(map[string]bool)
	layers := make(map[int]bool)
	for k := range tensors {
		if m := moduleRe.FindStringSubmatch(k); m != nil {
			parts := strings.Split(m[1], ".")
			modules[parts[len(parts)-1]] = true
		}
		if m := layerRe.FindStringSubmatch(k); m != nil {
			n, _ := strconv.Atoi(m[1])
			layers[n] = true
		}
	}

	sortedModules := make([]string, 0, len(modules))
	for m := range modules {
		sortedModules = append(sortedModules, m)
	}
	sort.Strings(sortedModules)

	sortedLayers := make([]int, 0, len(layers))
	for l := range layers {
		sortedLayers = append(sortedLayers, l)
	}
	sort.Ints(sortedLayers)

	// Write PEFT adapter_config.json.
	peftConfig := map[string]interface{}{
		"auto_mapping":           nil,
		"base_model_name_or_path": baseModelName,
		"bias":                   "none",
		"fan_in_fan_out":         false,
		"inference_mode":         true,
		"init_lora_weights":      true,
		"layers_pattern":         nil,
		"layers_to_transform":    sortedLayers,
		"lora_alpha":             math.Round(scale * float64(rank)),
		"lora_dropout":           mlxConfig.LoraParameters.Dropout,
		"modules_to_save":        nil,
		"peft_type":              "LORA",
		"r":                      rank,
		"revision":               nil,
		"target_modules":         sortedModules,
		"task_type":              "CAUSAL_LM",
	}

	cfgJSON, err := json.MarshalIndent(peftConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal peft config: %w", err)
	}

	if err := os.WriteFile(filepath.Join(outputDir, "adapter_config.json"), cfgJSON, 0644); err != nil {
		return fmt.Errorf("write adapter_config.json: %w", err)
	}

	log.Printf("converted %d tensors, %d layers, target modules: %v",
		len(peftTensors), len(sortedLayers), sortedModules)

	return nil
}
