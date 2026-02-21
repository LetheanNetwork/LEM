package lem

import (
	"math"

	"forge.lthn.ai/core/go-i18n/reversal"
)

// GrammarScore holds grammar-derived quality signals from a GrammarImprint.
type GrammarScore struct {
	VocabRichness float64 `json:"vocab_richness"`
	TenseEntropy  float64 `json:"tense_entropy"`
	QuestionRatio float64 `json:"question_ratio"`
	DomainDepth   int     `json:"domain_depth"`
	VerbDiversity int     `json:"verb_diversity"`
	NounDiversity int     `json:"noun_diversity"`
	Composite     float64 `json:"composite"`
}

// DeltaScore holds input-vs-output comparison signals.
type DeltaScore struct {
	InputComposite  float64 `json:"input_composite"`
	OutputComposite float64 `json:"output_composite"`
	Uplift          float64 `json:"uplift"`
	Echo            float64 `json:"echo"`
	Enrichment      float64 `json:"enrichment"`
	Sycophantic     bool    `json:"sycophantic"`
}

// ComputeGrammarScore derives quality signals from a GrammarImprint.
//
// Composite is a weighted combination of normalised signals (0-100):
//   - Tense diversity (0.25): varied tense = narrative depth
//   - Vocab richness (0.25): diverse vocabulary = engagement
//   - Question ratio (0.20): questioning = critical thinking
//   - Verb diversity (0.15): action variety = specificity
//   - Noun diversity (0.15): concept breadth = thoroughness
func ComputeGrammarScore(imp reversal.GrammarImprint) GrammarScore {
	gs := GrammarScore{
		VerbDiversity: imp.UniqueVerbs,
		NounDiversity: imp.UniqueNouns,
	}

	if imp.TokenCount > 0 {
		gs.VocabRichness = float64(imp.UniqueVerbs+imp.UniqueNouns) / float64(imp.TokenCount)
	}

	gs.TenseEntropy = shannonEntropy(imp.TenseDistribution)
	gs.QuestionRatio = imp.PunctuationPattern["question"]

	for _, v := range imp.DomainVocabulary {
		gs.DomainDepth += v
	}

	tenseNorm := gs.TenseEntropy / 1.585 // max entropy for 3 tenses = log2(3)
	vocabNorm := math.Min(gs.VocabRichness*10, 1.0)
	questionNorm := math.Min(gs.QuestionRatio*5, 1.0)
	verbNorm := math.Min(float64(gs.VerbDiversity)/30.0, 1.0)
	nounNorm := math.Min(float64(gs.NounDiversity)/40.0, 1.0)

	gs.Composite = 0.25*tenseNorm +
		0.25*vocabNorm +
		0.20*questionNorm +
		0.15*verbNorm +
		0.15*nounNorm

	gs.Composite *= 100.0

	return gs
}

// ComputeDelta scores both prompt and response, computing enrichment signals.
func ComputeDelta(tok *reversal.Tokeniser, prompt, response string, echoThreshold, upliftThreshold float64) DeltaScore {
	inTokens := tok.Tokenise(prompt)
	inImprint := reversal.NewImprint(inTokens)
	inGrammar := ComputeGrammarScore(inImprint)

	outTokens := tok.Tokenise(response)
	outImprint := reversal.NewImprint(outTokens)
	outGrammar := ComputeGrammarScore(outImprint)

	echo := inImprint.Similar(outImprint)
	uplift := outGrammar.Composite - inGrammar.Composite

	return DeltaScore{
		InputComposite:  inGrammar.Composite,
		OutputComposite: outGrammar.Composite,
		Uplift:          uplift,
		Echo:            echo,
		Enrichment:      uplift * (1.0 - echo),
		Sycophantic:     echo > echoThreshold && uplift < upliftThreshold,
	}
}

// ScoreResponse scores a single response text and returns the grammar score.
func ScoreResponse(tok *reversal.Tokeniser, text string) GrammarScore {
	tokens := tok.Tokenise(text)
	imprint := reversal.NewImprint(tokens)
	return ComputeGrammarScore(imprint)
}

func shannonEntropy(dist map[string]float64) float64 {
	var h float64
	for _, p := range dist {
		if p > 0 {
			h -= p * math.Log2(p)
		}
	}
	return h
}
