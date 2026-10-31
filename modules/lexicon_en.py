"""Leksikon sentimen bahasa Inggris (untuk labeling otomatis / ground truth).

Sama seperti versi Indonesia: skor dihitung pada token hasil normalisasi
(belum di-stem), dengan dukungan negasi (jendela 3 token, negasi ganda
dihitung), bigram, dan intensifier.
"""
from .lexicon import label_from_score  # noqa: F401  (dipakai ulang)

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "best", "better", "nice",
    "love", "loved", "like", "liked", "happy", "glad", "pleased", "satisfied",
    "helpful", "help", "helps", "helped", "beneficial", "useful", "valuable",
    "support", "supports", "supported", "supportive", "agree", "agreed",
    "approve", "positive", "success", "successful", "succeed", "improve",
    "improved", "improvement", "progress", "benefit", "benefits", "win", "won",
    "strong", "smart", "brilliant", "perfect", "fantastic", "wonderful",
    "impressive", "impressed", "inspiring", "inspired", "thanks", "thank",
    "grateful", "appreciate", "appreciated", "respect", "fair", "honest",
    "trust", "trusted", "reliable", "safe", "healthy", "affordable", "cheap",
    "worth", "worthwhile", "recommend", "recommended", "congratulations",
    "proud", "hope", "hopeful", "optimistic", "promising", "delighted",
    "enjoy", "enjoyed", "enjoyable", "fun", "cool", "fine", "okay", "ok",
    "correct", "accurate", "clear", "effective", "efficient", "welcome",
}

NEGATIVE_WORDS = {
    "bad", "worse", "worst", "terrible", "awful", "horrible", "poor", "useless",
    "hate", "hated", "dislike", "angry", "upset", "sad", "disappointed",
    "disappointing", "disappointment", "fail", "failed", "failure", "fails",
    "problem", "problems", "issue", "issues", "wrong", "error", "broken",
    "corrupt", "corruption", "lie", "lies", "lying", "fake", "scam", "fraud",
    "waste", "wasted", "wasteful", "expensive", "costly", "burden", "harm",
    "harmful", "hurt", "damage", "damaged", "destroy", "destroyed", "ruin",
    "ruined", "confusing", "confused", "unclear", "unfair", "unjust",
    "disagree", "disagreed", "reject", "rejected", "oppose", "opposed",
    "protest", "criticize", "criticized", "criticism", "complain", "complaint",
    "negative", "weak", "stupid", "dumb", "boring", "tired", "exhausted",
    "worried", "worry", "fear", "afraid", "danger", "dangerous", "risky",
    "delay", "delayed", "late", "slow", "mess", "messy", "chaos", "chaotic",
    "shame", "shameful", "embarrassing", "regret", "regretted", "pointless",
    "meaningless", "ineffective", "inefficient", "misleading", "exaggerated",
    "incompetent", "careless", "neglect", "neglected", "abandoned",
    "stop", "cancel", "cancelled", "refuse", "refused", "deny", "denied",
}

POSITIVE_BIGRAMS = {"well done", "thank you", "great job", "high quality"}
NEGATIVE_BIGRAMS = {"not fair", "no good", "waste money", "waste of", "rip off"}

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "can't", "cant", "don't", "dont", "doesn't", "doesnt", "didn't",
    "didnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "won't", "wont", "wouldn't", "wouldnt",
    "shouldn't", "shouldnt", "couldn't", "couldnt", "hardly", "barely",
    "without", "lack", "lacks", "lacking",
}

INTENSIFIERS = {
    "very", "really", "so", "extremely", "highly", "totally", "completely",
    "absolutely", "super", "too", "quite", "especially", "truly", "incredibly",
    "much", "most", "more",
}

# Kata yang dipertahankan pada stopword removal
KEEP_WORDS = NEGATION_WORDS | INTENSIFIERS

# Kata bermuatan sentimen & kata penyusun bigram tidak di-stem / tidak dibuang
PROTECTED_WORDS = (
    set(POSITIVE_WORDS) | set(NEGATIVE_WORDS) | KEEP_WORDS
    | {w for p in (POSITIVE_BIGRAMS | NEGATIVE_BIGRAMS) for w in p.split()}
)

# Stopword bahasa Inggris (gabungan daftar umum, sengaja ringkas)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "might", "my", "myself", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "then", "there", "these", "they", "this",
    "those", "through", "to", "under", "until", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "able", "also", "even", "ever", "get", "got", "go", "going", "gone",
    "hey", "hi", "hello", "guys", "guy", "people", "person", "thing", "things",
    "think", "thought", "know", "knew", "see", "seen", "saw", "say", "said",
    "says", "one", "two", "many", "lot", "lots", "much", "like", "want",
    "need", "make", "made", "take", "took", "come", "came", "really2",
    "video", "channel", "subscribe", "upload", "share", "comment", "content",
    "youtube", "instagram", "telegram", "post", "wow", "lol", "lmao",
    "yes", "yeah", "yep", "nah", "well", "ok", "okay2", "please", "pls",
}

# Normalisasi singkatan/gaul Inggris
NORMALIZATION_DICT = {
    "u": "you", "ur": "your", "r": "are", "y": "why", "k": "ok",
    "pls": "please", "plz": "please", "thx": "thanks", "tnx": "thanks",
    "ty": "thank", "bc": "because", "bcoz": "because", "coz": "because",
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not",
    "werent": "were not", "cant": "cannot", "wont": "will not",
    "wouldnt": "would not", "shouldnt": "should not", "couldnt": "could not",
    "im": "i am", "ive": "i have", "id": "i would", "ill": "i will",
    "youre": "you are", "theyre": "they are", "thats": "that is",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of", "cuz": "because",
    "good": "good", "gr8": "great", "gud": "good", "b4": "before",
    "msg": "message", "plz2": "please", "smh": "disappointed",
    "tbh": "honestly", "imo": "opinion", "rn": "now", "asap": "soon",
}


# ============================== SKORING (algoritma sama dengan versi ID) ==============================
_WORD_WEIGHT = {}
for _w in POSITIVE_WORDS:
    _WORD_WEIGHT[_w] = 1
for _w in NEGATIVE_WORDS:
    _WORD_WEIGHT[_w] = -1

_BIGRAM_WEIGHT = {}
for _b in POSITIVE_BIGRAMS:
    _BIGRAM_WEIGHT[_b] = 1
for _b in NEGATIVE_BIGRAMS:
    _BIGRAM_WEIGHT[_b] = -1

del _w, _b


def _negated_en(tokens, i):
    """Negasi maks. 2 token ke belakang, berhenti di kata sentimen lain."""
    count = 0
    for j in range(i - 1, max(-1, i - 3), -1):
        tok = tokens[j]
        if tok in NEGATION_WORDS:
            count += 1
        elif tok in _WORD_WEIGHT:
            break
    return count % 2 == 1


def score_text_en(tokens):
    """Hitung skor sentimen Inggris (negasi, bigram, intensifier)."""
    score = 0
    n = len(tokens)
    i = 0
    while i < n:
        if i + 1 < n:
            wb = _BIGRAM_WEIGHT.get(f"{tokens[i]} {tokens[i + 1]}")
            if wb is not None:
                score += -wb if _negated_en(tokens, i) else wb
                i += 2
                continue
        w = tokens[i]
        base = _WORD_WEIGHT.get(w)
        if base is not None:
            neg = _negated_en(tokens, i)
            amp = 1
            if not neg and ((i > 0 and tokens[i - 1] in INTENSIFIERS) or
                            (i + 1 < n and tokens[i + 1] in INTENSIFIERS)):
                amp = 2
            score += (base * amp) if not neg else (-base)
        i += 1
    return score
