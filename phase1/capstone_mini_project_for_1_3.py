from __future__ import annotations
import re
import math
from collections import Counter
from statistics import mean, stdev

DATA = [
    ("You’ve won a $500 gift card! Reply YES to claim now.", "spam"),
    ("Your appointment is confirmed for Tuesday at 10 AM.not spam Urgent: verify your bank account using this link.", "spam"),
    ("Can we move our meeting to 3 PM?", "not spam"),
    ("Get cheap prescription medicines—no prescription needed!", "spam"),
    ("Thanks for sending the report.I’ll review it today.", "not spam"),
    ("Congratulations! You were selected for a free vacation.", "spam"),
    ("Your package has been delivered to the front desk.", "not spam"),
    ("Claim your lottery prize before it expires tonight!", "spam"),
    ("Don’t forget to pick up milk on your way home.", "not spam"),
    ("Earn money from home with zero effort.Click here!", "spam"),
    ("The project deadline has been extended to Friday.", "not spam"),
    ("Your account will be suspended unless you act immediately.", "spam"),
    ("Happy birthday! Hope you have a wonderful day.", "not spam"),
    ("Exclusive offer: 90 % off luxury watches today only.", "spam"),
    ("Please find the meeting notes attached.", "not spam"),
    ("You qualify for a pre - approved loan—apply instantly.", "spam"),
    ("I’ll call you when I arrive at the station.", "not spam"),
    ("Update your password now to avoid losing access.", "spam"),
    ("The restaurant reservation is under your name for 7 PM.", "not spam"),
    ("cheap loans available now", "spam"),
    ("exclusive offer click here", "spam"),
    ("congratulations you win money", "spam"),
    ("meeting moved to tomorrow", "not spam"),
    ("please review the project report", "not spam"),
    ("are we still having lunch today", "not spam"),
    ("can you call me tonight", "not spam"),
    ("the team meeting is at noon", "not spam"),
    ("thanks for your help", "not spam"),
    ("please send the budget file", "not spam"),
    ("see you at the office", "not spam"),
    ("family dinner this weekend", "not spam"),
    ("your appointment is confirmed", "not spam"),
    ("free project update", "not spam"),
    ("review the attached document", "not spam"),
    ("cash bonus available today", "spam"),
    ("free entry claim reward", "spam"),
    ("call now for a special offer", "spam"),
    ("win a cheap prize today", "spam"),
]

def tokenize(text: str) -> list[str]:
    """Tokenizes the input text into lowercase words."""
    return re.findall(r"[a-z]+", text.lower())

class ManualNaiveBayes:
    def __init__(self):
        self.vocabulary = None
        self.total_words = None
        self.word_counts = None
        self.total_documents = None
        self.document_counts = None
        self.classes = None

    def fit(self, examples: list[tuple[str, str]]) -> None:
        self.classes = {"spam", "not spam"}
        self.document_counts = Counter(label for _, label in examples)
        self.total_documents = len(examples)
        self.word_counts = {label: Counter() for label in self.classes}
        self.total_words = Counter()

        for text, label in examples:
            words = tokenize(text)
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)

        self.vocabulary = set().union(*(counts.keys() for counts in self.word_counts.values()))

    def prior(self, label: str) -> float:
        """Calculates the prior probability of a class."""
        return self.document_counts[label] / self.total_documents

    def word_probability(self, word: str, label: str, smoothing: bool) -> float:
        if smoothing:
            # Add-one smoothing.  The extra bucket represents every new word as UNK.
            vocabulary_size_including_unk = len(self.vocabulary) + 1
            return (self.word_counts[label][word] + 1) / (self.total_words[label] + vocabulary_size_including_unk)
        return self.word_counts[label][word] / self.total_words[label]

    def posterior(self, text: str, smoothing: bool) -> dict[str, float] | None:
        """Returning normalized P(class | message), or None if both MLE scores to zero."""
        scores = {}
        for label in self.classes:
            score = self.prior(label)
            for word in tokenize(text):
                score *= self.word_probability(word, label, smoothing)
            scores[label] = score

        total = sum(scores.values())
        return None if total == 0 else {label: score / total for label, score in scores.items()}

    def predict(self, text: str, smoothing: bool) -> str | None:
        """Returns the class with the highest posterior probability, or None if both are zero."""
        probabilities = self.posterior(text, smoothing)
        if probabilities is None:
            return None
        return max(probabilities, key=probabilities.get)


def t_critical_95(df: int) -> float:
    """Values sufficient for this small demonstration; normal approximation after wards."""
    lookup = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return lookup.get(df, 1.96)

def accuracy_confidence_interval(outcomes: list[int]) -> tuple[float, float, float]:
    """Two-sided 95% confidence interval for the accuracy of independent test outcomes."""
    accuracy = mean(outcomes)
    margin = t_critical_95(len(outcomes) - 1) * stdev(outcomes) / math.sqrt(len(outcomes))
    return accuracy, max(0.0, accuracy - margin), min(1.0, accuracy + margin)

def show_result(model: ManualNaiveBayes, message: str, smoothing: bool) -> None:
    posterior = model.posterior(message, smoothing)
    if posterior is None:
        print(f" {message!r} -> undefined(both unnormalized scores are zero)")
    else:
        print(f" {message!r} -> {model.predict(message, smoothing)}"
            f"(spam={posterior['spam']:.4f}, not spam={posterior['not spam']:.4f})"
              )

def main()-> None:
    train, test = DATA[:30], DATA[30:]
    model = ManualNaiveBayes()
    model.fit(train)

    print("Training Priors")
    for label in model.classes:
        print(f" P({label}) -> {model.prior(label):.4f} "
              f"({model.document_counts[label]}/{model.total_documents})")

    print("Conditional Word Probabilities (MLE examples)")
    for word in ("cash", "free", "meeting", "lottery", "project"):
        print(f"{word:>7}: P(word|spam)={model.word_probability(word, 'spam', smoothing=False):.4f}, "
              f"P(word|not spam)={model.word_probability(word, 'not spam', smoothing=False):.4f}")

    print("\n NEW MESSAGES WITH RAW MLE")
    for message in ("win cash", "project meeting", "free lottery", "free meeting"):
        show_result(model, message, smoothing=False)


    edge_case = "bitcoin offer"
    print("\n ZERO-FREQUENCY EDGE CASE, WITH RAW MLE")
    show_result(model, edge_case, smoothing=False)
    print("'bitcoin' occurred 0 times in training data, so P(bitcoin|spam) and P(bitcoin|not spam) are both 0, ")

    print("\n SAME EDGE CASE, WITH LAPLACE (ADD-ONE) SMOOTHING")
    show_result(model, edge_case, smoothing=True)

    outcomes = [int(model.predict(text, smoothing=True) == label) for text, label in test]
    accuracy, lower, upper = accuracy_confidence_interval(outcomes)
    print("\n HELD-OUT TEST EVALUATION (Laplace smoothed)")
    print(f" Accuracy = {accuracy:.1%} ({sum(outcomes)}/{len(outcomes)})")
    print(f" approximate 95% t-interval = [{lower:.1%}, {upper:.1%}]")
    print("This interval is very wide because hold out set is tiny.")


if __name__ == "__main__":
    main()
