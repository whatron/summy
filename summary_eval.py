# summary_evaluation.py

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

# Download required resources for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def evaluate_summaries(generated_summaries, reference_summaries):
    reference_summary = """This lecture introduces the fundamentals of quantum computing, including key principles like superposition, entanglement, and the quantum circuit model. It explains how qubits differ from classical bits and how quantum gates (like Hadamard and CNOT) enable complex computations. Practical phenomena such as the double-slit experiment and quantum tunneling illustrate core quantum effects. Major challenges include error correction due to decoherence and scalability. Applications span cryptography, drug discovery, optimization, and machine learning. While quantum computers offer speedups for specific problems, they are not universally superior to classical systems. Understanding the field requires background in quantum mechanics, linear algebra, and classical computing. The talk concludes by highlighting industry progress and ongoing challenges in building practical, large-scale quantum computers."""
    assert len(generated_summaries) == len(reference_summaries), "Mismatched summary list lengths"

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    bleu_scores, meteor_scores = [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        # ROUGE
        scores = rouge.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

        # BLEU
        bleu = sentence_bleu(
            [ref.split()],
            gen.split(),
            smoothing_function=smooth_fn
        )
        bleu_scores.append(bleu)

        # METEOR
        meteor = meteor_score([ref], gen)
        meteor_scores.append(meteor)

    results = {
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores),
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores),
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores),
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
    }

    return results


if __name__ == "__main__":
    # Example usage
    generated = [
        "A fast brown fox leaps over a lazy dog",
        "She read the book in one night"
    ]
    reference = [
        "The quick brown fox jumps over the lazy dog",
        "She finished reading the book within one night"
    ]

    scores = evaluate_summaries(generated, reference)
    print("Evaluation Metrics:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")