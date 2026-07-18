"""
Extraction-fidelity metrics and query accounting for the hidden-CoT side channel.

We report a small battery of complementary metrics because no single number
captures "how much of the hidden reasoning was recovered":

  * Exact / verbatim overlap  -- the strict, string-matching view of extraction
    used by Carlini et al. in the training-data-extraction line of work
    ("Extracting Training Data from Large Language Models", USENIX Sec 2021, and
    "Quantifying Memorization Across Neural Language Models", 2023). This is the
    conservative lower bound: it only credits contiguous verbatim reproduction.

  * Token edit distance (Levenshtein) + BLEU / ROUGE -- the approximate
    string-matching view.

  * Embedding cosine similarity -- a semantic view. Recent work argues that
    string matching *severely undercounts* extraction because trivial surface
    artifacts (whitespace, punctuation, reordering) deflate edit-distance-style
    metrics, and that a high-quality embedding model recovers the semantic
    overlap those metrics miss:

        "While the majority of related work on memorisation has focused on
         measuring success of training data extraction through string matching,
         we argue that embedding models are better suited [...]. Distances
         measured through a high quality embedding model can identify semantic
         similarities between strings that a different metric such as edit
         distance will struggle to capture."
        -- Google, "Extracting memorised data ...", arXiv:2510.18554

Optional dependencies (sacrebleu, rouge_score, sentence-transformers) are
imported lazily; if any is missing its metric is reported as ``None`` rather
than crashing the run, so the core string metrics always work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


# --------------------------------------------------------------------------- #
# Query accounting
# --------------------------------------------------------------------------- #
class QueryCounter:
    """Counts forward passes / API queries so we can report queries-per-token.

    The attack increments this on every logit oracle call. Efficiency is then
    ``total / n_tokens_extracted`` -- the headline cost metric for the paper.
    """

    def __init__(self) -> None:
        self.forward_passes: int = 0

    def tick(self, n: int = 1) -> None:
        self.forward_passes += n

    def reset(self) -> None:
        self.forward_passes = 0

    def per_token(self, n_tokens: int) -> float:
        return self.forward_passes / n_tokens if n_tokens else float("nan")


# --------------------------------------------------------------------------- #
# Exact / verbatim (Carlini-style) overlap
# --------------------------------------------------------------------------- #
def longest_common_substring_len(a: Sequence, b: Sequence) -> int:
    """Length of the longest *contiguous* common subsequence (substring).

    Works on token-id lists or characters. O(len(a) * len(b)) with a rolling
    two-row DP so long CoTs stay tractable.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def verbatim_overlap(pred: Sequence, truth: Sequence) -> float:
    """Longest contiguous verbatim match, normalised by truth length -> [0, 1]."""
    if not truth:
        return 0.0
    return longest_common_substring_len(pred, truth) / len(truth)


def ngram_extraction_rate(pred: Sequence, truth: Sequence, n: int = 50) -> Optional[float]:
    """Carlini-style extractable-memorisation rate.

    Fraction of length-``n`` contiguous windows of ``truth`` that appear
    verbatim as a contiguous span in ``pred``. Returns ``None`` when ``truth``
    is shorter than ``n`` (the window is undefined).
    """
    if len(truth) < n:
        return None
    pred_windows = {tuple(pred[i : i + n]) for i in range(0, len(pred) - n + 1)}
    total = len(truth) - n + 1
    hit = sum(1 for i in range(total) if tuple(truth[i : i + n]) in pred_windows)
    return hit / total


def exact_match(pred: Sequence, truth: Sequence) -> bool:
    return list(pred) == list(truth)


# --------------------------------------------------------------------------- #
# Approximate string matching
# --------------------------------------------------------------------------- #
def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over arbitrary sequences (token ids or characters)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def normalized_levenshtein_similarity(pred: Sequence, truth: Sequence) -> float:
    """1 - edit_distance / max_len -> [0, 1], higher is better."""
    m = max(len(pred), len(truth))
    if m == 0:
        return 1.0
    return 1.0 - levenshtein(pred, truth) / m


def token_f1(pred: Sequence, truth: Sequence) -> float:
    """Unigram (bag-of-tokens) F1 -- order-insensitive overlap."""
    from collections import Counter

    cp, ct = Counter(pred), Counter(truth)
    overlap = sum((cp & ct).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cp.values())
    recall = overlap / sum(ct.values())
    return 2 * precision * recall / (precision + recall)


def bleu(pred_text: str, truth_text: str) -> Optional[float]:
    """Sentence BLEU via sacrebleu (0-100). ``None`` if sacrebleu is absent."""
    try:
        import sacrebleu
    except ImportError:
        return None
    return sacrebleu.sentence_bleu(pred_text, [truth_text]).score


def rouge_l(pred_text: str, truth_text: str) -> Optional[float]:
    """ROUGE-L F-measure via rouge_score. ``None`` if the lib is absent."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(truth_text, pred_text)["rougeL"].fmeasure


# --------------------------------------------------------------------------- #
# Semantic (embedding) similarity -- arXiv:2510.18554
# --------------------------------------------------------------------------- #
class EmbeddingScorer:
    """Cosine similarity in a sentence-embedding space.

    Motivated by arXiv:2510.18554, which shows string-matching metrics
    undercount extraction; embeddings recover the semantic overlap. The model
    is loaded once and reused. If sentence-transformers is unavailable,
    ``score`` returns ``None`` so the harness degrades gracefully.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._unavailable = False

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception:
            self._unavailable = True
            return False

    def score(self, pred_text: str, truth_text: str) -> Optional[float]:
        if not self._ensure():
            return None
        import numpy as np

        emb = self._model.encode([pred_text, truth_text], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionScores:
    exact_match: bool
    verbatim_overlap: float
    ngram_extraction_rate: Optional[float]
    levenshtein_similarity: float
    token_f1: float
    bleu: Optional[float]
    rouge_l: Optional[float]
    embedding_similarity: Optional[float]
    extra: dict = field(default_factory=dict)

    def as_row(self) -> str:
        def fmt(x):
            if x is None:
                return "  n/a"
            if isinstance(x, bool):
                return " yes" if x else "  no"
            return f"{x:5.3f}"

        return (
            f"exact={fmt(self.exact_match)}  verbatim={fmt(self.verbatim_overlap)}  "
            f"ngram={fmt(self.ngram_extraction_rate)}  editsim={fmt(self.levenshtein_similarity)}  "
            f"f1={fmt(self.token_f1)}  bleu={fmt(self.bleu)}  rougeL={fmt(self.rouge_l)}  "
            f"embed={fmt(self.embedding_similarity)}"
        )


def evaluate(
    pred_text: str,
    truth_text: str,
    tokenize: Callable[[str], Sequence] = None,
    embedding_scorer: Optional[EmbeddingScorer] = None,
    ngram_n: int = 50,
) -> ExtractionScores:
    """Compute the full metric battery for one prediction vs. one ground truth.

    ``tokenize`` maps text -> a sequence of token ids (pass the model tokenizer's
    encoder for token-level metrics); if ``None`` we fall back to whitespace
    words so the function is usable without a tokenizer.
    """
    if tokenize is None:
        tokenize = lambda s: s.split()

    pred_tokens = list(tokenize(pred_text))
    truth_tokens = list(tokenize(truth_text))

    return ExtractionScores(
        exact_match=exact_match(pred_tokens, truth_tokens),
        verbatim_overlap=verbatim_overlap(pred_tokens, truth_tokens),
        ngram_extraction_rate=ngram_extraction_rate(pred_tokens, truth_tokens, n=ngram_n),
        levenshtein_similarity=normalized_levenshtein_similarity(pred_tokens, truth_tokens),
        token_f1=token_f1(pred_tokens, truth_tokens),
        bleu=bleu(pred_text, truth_text),
        rouge_l=rouge_l(pred_text, truth_text),
        embedding_similarity=(
            embedding_scorer.score(pred_text, truth_text) if embedding_scorer else None
        ),
    )
