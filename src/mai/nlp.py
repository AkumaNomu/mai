import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[a-zA-Z']+")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "when",
    "while",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "onto",
    "to",
    "with",
    "without",
    "up",
    "down",
    "over",
    "under",
    "again",
    "once",
    "here",
    "there",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "their",
    "our",
    "mine",
    "yours",
    "hers",
    "theirs",
    "its",
    "what",
    "who",
    "whom",
    "which",
    "why",
    "how",
    "not",
    "no",
    "yes",
    "very",
    "really",
    "just",
    "still",
    "also",
    "so",
    "too",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "must",
    "will",
    "shall",
}

INTENSIFIERS = {
    "very": 1.3,
    "extremely": 1.6,
    "deeply": 1.4,
    "highly": 1.3,
    "so": 1.2,
    "really": 1.2,
    "incredibly": 1.6,
    "utterly": 1.5,
    "absolutely": 1.5,
    "totally": 1.4,
}

DOWNPLAYERS = {
    "slightly": 0.7,
    "somewhat": 0.8,
    "mildly": 0.8,
    "barely": 0.6,
    "hardly": 0.6,
    "a_bit": 0.8,
}

NEGATORS = {"not", "no", "never", "without"}

EMOTION_LEXICON = {
    "joy": {
        "joy",
        "happy",
        "delight",
        "smile",
        "laugh",
        "relief",
        "warm",
        "bright",
        "playful",
        "cheerful",
    },
    "sadness": {
        "sad",
        "lonely",
        "grief",
        "loss",
        "lost",
        "cry",
        "mourning",
        "empty",
        "heartbroken",
        "melancholy",
    },
    "anger": {
        "anger",
        "angry",
        "rage",
        "furious",
        "wrath",
        "frustration",
        "frustrated",
        "revenge",
        "vengeance",
        "hate",
    },
    "fear": {
        "fear",
        "afraid",
        "anxious",
        "anxiety",
        "nervous",
        "panic",
        "terror",
        "dread",
        "uneasy",
        "worried",
    },
    "disgust": {
        "disgust",
        "disgusted",
        "revulsion",
        "nausea",
        "gross",
        "filthy",
        "vile",
    },
    "surprise": {
        "surprise",
        "shocked",
        "sudden",
        "unexpected",
        "startled",
        "astonished",
    },
    "trust": {
        "trust",
        "safe",
        "secure",
        "calm",
        "gentle",
        "comfort",
        "tender",
        "steady",
    },
    "anticipation": {
        "anticipation",
        "anticipate",
        "expect",
        "prepare",
        "preparing",
        "ready",
        "upcoming",
        "await",
        "soon",
    },
}

SETTING_TAGS = {
    "interior": {"interior", "inside", "room", "hall", "kitchen", "office"},
    "exterior": {"exterior", "outside", "street", "rooftop", "forest", "field"},
    "day": {"day", "daylight", "morning", "afternoon", "sunrise", "sunny"},
    "night": {"night", "midnight", "moon", "dark", "streetlight", "neon"},
    "urban": {"city", "street", "alley", "subway", "downtown", "traffic"},
    "rural": {"farm", "field", "countryside", "barn", "meadow"},
    "nature": {"forest", "ocean", "river", "mountain", "desert", "lake"},
    "weather_rain": {"rain", "storm", "thunder", "lightning", "wet"},
    "weather_snow": {"snow", "ice", "winter", "frost"},
}

COLOR_WORDS = {
    "red",
    "crimson",
    "scarlet",
    "blue",
    "cyan",
    "teal",
    "green",
    "emerald",
    "yellow",
    "gold",
    "orange",
    "purple",
    "violet",
    "pink",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "neon",
    "pastel",
    "muted",
    "vivid",
    "monochrome",
}

LEXICON = {
    "positive": {
        "joy",
        "happy",
        "smile",
        "laugh",
        "love",
        "hope",
        "warm",
        "bright",
        "sweet",
        "tender",
        "peace",
        "calm",
        "safe",
        "glow",
        "sunlit",
        "sunny",
        "playful",
        "dreamy",
    },
    "negative": {
        "sad",
        "lonely",
        "fear",
        "afraid",
        "angry",
        "rage",
        "cold",
        "dark",
        "bleak",
        "pain",
        "grief",
        "cry",
        "broken",
        "empty",
        "guilt",
        "regret",
    },
    "high_energy": {
        "fast",
        "rush",
        "run",
        "chase",
        "fight",
        "explosion",
        "panic",
        "chaos",
        "storm",
        "loud",
        "intense",
        "clash",
        "attack",
        "escape",
    },
    "low_energy": {
        "slow",
        "quiet",
        "still",
        "soft",
        "gentle",
        "rest",
        "breathe",
        "calm",
        "serene",
        "sleep",
        "hush",
        "silent",
        "pause",
    },
    "tension": {
        "suspense",
        "threat",
        "danger",
        "risk",
        "mystery",
        "shadow",
        "secret",
        "trap",
        "hide",
        "search",
        "fear",
        "panic",
        "blood",
        "knife",
        "gun",
    },
    "romance": {
        "love",
        "kiss",
        "touch",
        "embrace",
        "heart",
        "romance",
        "date",
        "tender",
        "soft",
        "slow",
    },
    "awe": {
        "vast",
        "grand",
        "epic",
        "majestic",
        "wonder",
        "awe",
        "cosmic",
        "sky",
        "mountain",
        "ocean",
    },
    "whimsy": {
        "whimsical",
        "quirky",
        "odd",
        "playful",
        "fun",
        "light",
        "sparkle",
        "magic",
        "fantasy",
    },
    "noir": {
        "noir",
        "smoke",
        "neon",
        "rain",
        "alley",
        "night",
        "shadow",
        "grit",
        "crime",
        "city",
    },
    "grief": {
        "grief",
        "grieve",
        "loss",
        "lost",
        "death",
        "killed",
        "murder",
        "mourning",
        "funeral",
        "brother",
        "sister",
        "father",
        "mother",
    },
    "anger": {
        "anger",
        "angry",
        "rage",
        "furious",
        "wrath",
        "frustration",
        "frustrated",
        "revenge",
        "vengeance",
        "hate",
    },
    "anxiety": {
        "anxious",
        "anxiety",
        "nervous",
        "uneasy",
        "worried",
        "panic",
        "pressure",
        "stress",
        "tense",
        "dread",
    },
    "resolve": {
        "resolve",
        "determined",
        "determination",
        "focus",
        "focused",
        "training",
        "prepare",
        "preparing",
        "discipline",
        "grit",
        "drive",
    },
    "violence": {
        "fight",
        "punch",
        "boxing",
        "blood",
        "gun",
        "knife",
        "attack",
        "hit",
        "strike",
        "brutal",
    },
    "crime": {
        "mafia",
        "mob",
        "gang",
        "crime",
        "cartel",
        "illegal",
        "heist",
        "underworld",
        "betrayal",
    },
}


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def split_scenes(text: str, mode: str, scene_regex: str) -> list[str]:
    if mode == "lines":
        parts = [p.strip() for p in text.splitlines()]
        return [p for p in parts if p]
    if mode == "file":
        return [text.strip()] if text.strip() else []
    if mode == "blank":
        parts = re.split(r"\n\s*\n+", text)
        return [p.strip() for p in parts if p.strip()]
    if mode == "regex":
        if not scene_regex:
            raise ValueError("--scene-regex is required when --split=regex")
        chunks = re.split(scene_regex, text, flags=re.IGNORECASE | re.MULTILINE)
        scenes = []
        current = ""
        for chunk in chunks:
            if not chunk.strip():
                continue
            if re.match(scene_regex, chunk, flags=re.IGNORECASE):
                if current.strip():
                    scenes.append(current.strip())
                current = chunk
            else:
                if current:
                    current += "\n" + chunk
                else:
                    current = chunk
        if current.strip():
            scenes.append(current.strip())
        return scenes
    raise ValueError(f"Unknown split mode: {mode}")


def weighted_emotion_scores(tokens_list: list[str]) -> dict:
    total = max(len(tokens_list), 1)
    scores = {k: 0.0 for k in EMOTION_LEXICON.keys()}
    for i, tok in enumerate(tokens_list):
        weight = 1.0
        prev = tokens_list[i - 1] if i > 0 else ""
        if prev in INTENSIFIERS:
            weight *= INTENSIFIERS[prev]
        if prev in DOWNPLAYERS:
            weight *= DOWNPLAYERS[prev]
        if prev in NEGATORS:
            weight *= 0.5

        for emotion, words in EMOTION_LEXICON.items():
            if tok in words:
                scores[emotion] += weight

    for emotion in scores:
        scores[emotion] = scores[emotion] / total
    return scores


def compute_scene_features(tokens: list[str]) -> dict:
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)

    def score(keys: set[str]) -> float:
        return sum(counts[k] for k in keys) / total

    positive = score(LEXICON["positive"])
    negative = score(LEXICON["negative"])
    high_energy = score(LEXICON["high_energy"])
    low_energy = score(LEXICON["low_energy"])
    tension = score(LEXICON["tension"])

    emotion_scores = weighted_emotion_scores(tokens)
    emotion_sum = sum(emotion_scores.values())
    emotion_profile = {}
    for k, v in emotion_scores.items():
        if emotion_sum > 0:
            emotion_profile[k] = v / emotion_sum
        else:
            emotion_profile[k] = 0.0

    valence = (emotion_scores["joy"] + emotion_scores["trust"]) - (
        emotion_scores["sadness"]
        + emotion_scores["disgust"]
        + emotion_scores["fear"]
        + emotion_scores["anger"]
    )
    arousal = (
        emotion_scores["anger"]
        + emotion_scores["fear"]
        + emotion_scores["surprise"]
        + emotion_scores["anticipation"]
    ) - (emotion_scores["sadness"] + emotion_scores["trust"])

    tags = []
    scores = {}
    for name, words in LEXICON.items():
        s = score(words)
        scores[name] = s
        if name in {"positive", "negative", "high_energy", "low_energy", "tension"}:
            continue
        if s > 0:
            tags.append(name)

    setting = []
    for name, words in SETTING_TAGS.items():
        if score(words) > 0:
            setting.append(name)

    color_mentions = [w for w in tokens if w in COLOR_WORDS]
    color_mentions = list(dict.fromkeys(color_mentions))[:8]

    return {
        "valence": valence,
        "arousal": arousal,
        "tension": tension,
        "positive": positive,
        "negative": negative,
        "high_energy": high_energy,
        "low_energy": low_energy,
        "emotion_scores": emotion_scores,
        "emotion_profile": emotion_profile,
        "lexicon_scores": scores,
        "mood_tags": tags,
        "setting_tags": setting,
        "color_words": color_mentions,
    }


def top_keywords(tokens: list[str], limit: int) -> list[str]:
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(limit)]


def load_texts(input_path: Path) -> list[tuple[str, str]]:
    if input_path.is_file():
        return [(input_path.name, input_path.read_text(encoding="utf-8", errors="ignore"))]
    if not input_path.is_dir():
        raise ValueError(f"Input path not found: {input_path}")
    files = sorted(input_path.rglob("*.txt"))
    return [
        (path.name, path.read_text(encoding="utf-8", errors="ignore")) for path in files
    ]


def extract_scene_text(
    input_path: Path | None,
    input_text: str,
    split_mode: str,
    scene_regex: str,
    min_chars: int,
    keywords: int,
) -> dict:
    if input_text:
        texts = [("<inline>", input_text)]
        input_label = "<inline>"
    else:
        if input_path is None:
            raise ValueError("--input or --input-text is required")
        texts = load_texts(input_path)
        input_label = str(input_path)

    scenes_out = []
    for filename, text in texts:
        scenes = split_scenes(text, split_mode, scene_regex)
        for idx, scene_text in enumerate(scenes):
            if len(scene_text) < min_chars:
                continue
            tokens = tokenize(scene_text)
            features = compute_scene_features(tokens)
            keywords_out = top_keywords(tokens, keywords)
            scenes_out.append(
                {
                    "scene_id": f"{filename}:{idx}",
                    "source_file": filename,
                    "text_len": len(scene_text),
                    "text": scene_text,
                    "keywords": keywords_out,
                    "features": features,
                }
            )

    return {
        "input": input_label,
        "num_scenes": len(scenes_out),
        "scenes": scenes_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract mood/context tags from user-provided scene context."
    )
    parser.add_argument("--input", help="Text file or folder of .txt")
    parser.add_argument(
        "--input-text",
        default="",
        help="Direct scene context string (overrides --input)",
    )
    parser.add_argument(
        "--split",
        default="lines",
        choices=["lines", "blank", "file", "regex"],
        help="Context split strategy",
    )
    parser.add_argument(
        "--scene-regex",
        default=r"^\s*(scene|int\.|ext\.)",
        help="Regex for scene headings when --split=regex",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum characters for a scene to be kept",
    )
    parser.add_argument(
        "--keywords",
        type=int,
        default=8,
        help="Top keywords to include per scene",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")

    args = parser.parse_args()
    result = extract_scene_text(
        input_path=Path(args.input) if args.input else None,
        input_text=args.input_text,
        split_mode=args.split,
        scene_regex=args.scene_regex,
        min_chars=args.min_chars,
        keywords=args.keywords,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
