import sys

from .audio import main as audio_main
from .cv import main as cv_main
from .legacy_optimizer import main as legacy_main
from .nlp import main as nlp_main
from .recommend import main as recommend_main


def _print_help() -> None:
    print("Mai CLI")
    print("Usage: python main.py <command> [args]")
    print("")
    print("Commands:")
    print("  recommend         Scene -> song recommendations (NLP + CV + audio + Spotify genres)")
    print("  analyze-audio     Analyze local MP3/audio library")
    print("  extract-scene     Extract NLP features from scene context")
    print("  extract-palette   Extract color palette/metrics from scene frames")
    print("  legacy-optimize   Old playlist optimizer flow")


def main() -> None:
    if len(sys.argv) < 2:
        _print_help()
        return

    command = sys.argv[1].strip().lower()
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "recommend":
        recommend_main()
        return
    if command == "analyze-audio":
        audio_main()
        return
    if command == "extract-scene":
        nlp_main()
        return
    if command == "extract-palette":
        cv_main()
        return
    if command == "legacy-optimize":
        legacy_main()
        return

    _print_help()


if __name__ == "__main__":
    main()
