# summary_evaluation.py

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

# Download required resources for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

def evaluate_summaries(generated_summaries, reference_summaries):
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

        # METEOR - tokenize the input text
        gen_tokens = nltk.word_tokenize(gen)
        ref_tokens = nltk.word_tokenize(ref)
        meteor = meteor_score([ref_tokens], gen_tokens)
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
        "Quantum computing is poised to transform how we process information. Unlike classical bits that can only be 0 or 1, quantum bits, or qubits, can exist in multiple states simultaneously Quantum entanglement is another crucial concept. When two or more particles become entangled, their quantum states become correlated. This phenomenon is essential for quantum teleportation The quantum circuit model is the most common way to describe quantum computations. A quantum circuit consists of quantum gates that manipulate qubits. The Hadamard gate creates super In terms of real-world applications, quantum computing shows great promise in several areas. In cryptography, it could break current encryption methods and create new quantum-resistant The quantum-classical boundary is an important concept to understand. While quantum computers can solve certain problems exponentially faster than classical computers, they're not universally better. The field of quantum computing is rapidly evolving. IBM's 433-qubit Osprey processor and Google's quantum supremacy experiment are significant milestones in this journey In conclusion, quantum computing represents a paradigm shift in how we process information. While we're still in the early stages of development, the potential applications are vast and transformative The quantum computing revolution is just beginning, and it's an exciting time to be involved in this field. This includes simulating quantum systems",
    ]
    reference = [
        "This lecture introduces the fundamentals of quantum computing, including key principles like superposition, entanglement, and the quantum circuit model. It explains how qubits differ from classical bits and how quantum gates (like Hadamard and CNOT) enable complex computations. Practical phenomena such as the double-slit experiment and quantum tunneling illustrate core quantum effects. Major challenges include error correction due to decoherence and scalability. Applications span cryptography, drug discovery, optimization, and machine learning. While quantum computers offer speedups for specific problems, they are not universally superior to classical systems. Understanding the field requires background in quantum mechanics, linear algebra, and classical computing. The talk concludes by highlighting industry progress and ongoing challenges in building practical, large-scale quantum computers."
        
    ]

    scores = evaluate_summaries(generated, reference)
    print("Evaluation Metrics:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")