package lem

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestRenameMLXKey(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{
			"model.layers.12.self_attn.q_proj.lora_a",
			"base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weight",
		},
		{
			"model.layers.0.self_attn.v_proj.lora_b",
			"base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
		},
		{
			"model.layers.5.mlp.gate_proj.lora_a",
			"base_model.model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
		},
	}

	for _, tt := range tests {
		got := renameMLXKey(tt.input)
		if got != tt.want {
			t.Errorf("renameMLXKey(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestTransposeFloat32(t *testing.T) {
	// 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
	data := make([]byte, 2*3*4)
	for i, v := range []float32{1, 2, 3, 4, 5, 6} {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	result := transposeFloat32(data, 2, 3)

	// Expected: 3x2 matrix: [[1, 4], [2, 5], [3, 6]]
	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, want := range expected {
		got := math.Float32frombits(binary.LittleEndian.Uint32(result[i*4:]))
		if got != want {
			t.Errorf("result[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestConvertMLXtoPEFT(t *testing.T) {
	dir := t.TempDir()

	// Create a minimal MLX safetensors file with one lora_a and one lora_b tensor.
	// Shape: lora_a is (in=4, rank=2), lora_b is (rank=2, out=4)
	tensors := map[string]safetensorsTensorInfo{
		"model.layers.0.self_attn.q_proj.lora_a": {Dtype: "F32", Shape: []int{4, 2}},
		"model.layers.0.self_attn.q_proj.lora_b": {Dtype: "F32", Shape: []int{2, 4}},
	}

	// Create tensor data: 4x2=8 floats and 2x4=8 floats.
	loraAData := make([]byte, 4*2*4)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(loraAData[i*4:], math.Float32bits(float32(i+1)))
	}
	loraBData := make([]byte, 2*4*4)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(loraBData[i*4:], math.Float32bits(float32(10+i)))
	}

	tensorData := make(map[string][]byte)
	tensorData["model.layers.0.self_attn.q_proj.lora_a"] = loraAData
	tensorData["model.layers.0.self_attn.q_proj.lora_b"] = loraBData

	sfPath := filepath.Join(dir, "adapters.safetensors")
	if err := writeSafetensors(sfPath, tensors, tensorData); err != nil {
		t.Fatalf("write test safetensors: %v", err)
	}

	// Create MLX config.
	mlxConfig := map[string]interface{}{
		"lora_parameters": map[string]interface{}{
			"rank":    8,
			"scale":   20.0,
			"dropout": 0.0,
		},
	}
	cfgData, _ := json.Marshal(mlxConfig)
	cfgPath := filepath.Join(dir, "adapter_config.json")
	os.WriteFile(cfgPath, cfgData, 0644)

	// Convert.
	outputDir := filepath.Join(dir, "peft_output")
	if err := convertMLXtoPEFT(sfPath, cfgPath, outputDir, "test-model"); err != nil {
		t.Fatalf("convert: %v", err)
	}

	// Check output files exist.
	if _, err := os.Stat(filepath.Join(outputDir, "adapter_model.safetensors")); err != nil {
		t.Error("missing adapter_model.safetensors")
	}
	if _, err := os.Stat(filepath.Join(outputDir, "adapter_config.json")); err != nil {
		t.Error("missing adapter_config.json")
	}

	// Read and verify PEFT config.
	peftCfgData, err := os.ReadFile(filepath.Join(outputDir, "adapter_config.json"))
	if err != nil {
		t.Fatalf("read peft config: %v", err)
	}

	var peftConfig map[string]interface{}
	if err := json.Unmarshal(peftCfgData, &peftConfig); err != nil {
		t.Fatalf("parse peft config: %v", err)
	}

	if peftConfig["peft_type"] != "LORA" {
		t.Errorf("peft_type = %v, want LORA", peftConfig["peft_type"])
	}
	if peftConfig["base_model_name_or_path"] != "test-model" {
		t.Errorf("base_model = %v, want test-model", peftConfig["base_model_name_or_path"])
	}

	// Check that lora_alpha = scale * rank = 20 * 8 = 160.
	if alpha, ok := peftConfig["lora_alpha"].(float64); !ok || alpha != 160 {
		t.Errorf("lora_alpha = %v, want 160", peftConfig["lora_alpha"])
	}

	// Verify converted safetensors has PEFT-format keys.
	peftTensors, _, err := readSafetensors(filepath.Join(outputDir, "adapter_model.safetensors"))
	if err != nil {
		t.Fatalf("read peft safetensors: %v", err)
	}

	expectedKeys := []string{
		"base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
		"base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
	}
	for _, k := range expectedKeys {
		if _, ok := peftTensors[k]; !ok {
			t.Errorf("missing expected PEFT key: %s", k)
		}
	}

	// Verify shapes are transposed: lora_a (4,2) → (2,4), lora_b (2,4) → (4,2).
	loraAInfo := peftTensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
	if loraAInfo.Shape[0] != 2 || loraAInfo.Shape[1] != 4 {
		t.Errorf("lora_A shape = %v, want [2, 4]", loraAInfo.Shape)
	}
}

func TestReadWriteSafetensorsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	original := map[string]safetensorsTensorInfo{
		"weight_a": {Dtype: "F32", Shape: []int{2, 3}},
	}
	data := map[string][]byte{
		"weight_a": make([]byte, 2*3*4),
	}
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint32(data["weight_a"][i*4:], math.Float32bits(float32(i)))
	}

	if err := writeSafetensors(path, original, data); err != nil {
		t.Fatalf("write: %v", err)
	}

	readTensors, readData, err := readSafetensors(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	if len(readTensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(readTensors))
	}

	info := readTensors["weight_a"]
	if info.Dtype != "F32" {
		t.Errorf("dtype = %s, want F32", info.Dtype)
	}
	if info.Shape[0] != 2 || info.Shape[1] != 3 {
		t.Errorf("shape = %v, want [2, 3]", info.Shape)
	}

	got := getTensorData(info, readData)
	if len(got) != 24 {
		t.Errorf("data length = %d, want 24", len(got))
	}
}
