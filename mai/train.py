"""Unified training entry point: ``python -m mai.train``.

Trains the supervised transition model (with hard-negative mining and musical
interaction features) and the set-level arc model from a scraped positive
transitions CSV, then writes both artifacts. Reports cross-validated AUC so you
can see, before generating anything, whether the training data actually
separates good handoffs from plausible-but-unchosen ones.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config
from .data import load_csv_playlist
from .sequence_model import (
    DEFAULT_ARC,
    DEFAULT_ARC_MODEL_PATH,
    save_arc_model,
    train_arc_model,
)
from .transition_model import (
    DEFAULT_TRANSITION_MODEL_DEVICE,
    DEFAULT_TRANSITION_MODEL_HARD_FRACTION,
    DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO,
    DEFAULT_TRANSITION_MODEL_PATH,
    DEFAULT_TRANSITION_MODEL_RANDOM_STATE,
    save_transition_model,
    train_transition_model,
)


logger = logging.getLogger(__name__)


def _build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train Mai transition + arc models.')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--no-config', action='store_true')
    parser.add_argument(
        '--train-csv',
        default=get_config_value(config, 'training.output_path', 'data/training/positive_transitions.csv'),
        help='Positive transitions CSV produced by the scraper.',
    )
    parser.add_argument('--transition-model-out', default=DEFAULT_TRANSITION_MODEL_PATH)
    parser.add_argument(
        '--arc-model-out',
        default=get_config_value(config, 'training.arc_model_path', DEFAULT_ARC_MODEL_PATH),
    )
    parser.add_argument(
        '--negative-ratio',
        type=float,
        default=float(get_config_value(config, 'training.transition_model_negative_ratio', DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO)),
    )
    parser.add_argument(
        '--hard-fraction',
        type=float,
        default=float(get_config_value(config, 'training.transition_model_hard_fraction', DEFAULT_TRANSITION_MODEL_HARD_FRACTION)),
        help='Share of negatives drawn as close-but-unchosen hard negatives (0..1).',
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default=str(get_config_value(config, 'training.transition_model_device', DEFAULT_TRANSITION_MODEL_DEVICE)),
    )
    parser.add_argument(
        '--arc-profile',
        default=str(get_config_value(config, 'training.arc_profile', DEFAULT_ARC)),
    )
    parser.add_argument('--random-state', type=int, default=DEFAULT_TRANSITION_MODEL_RANDOM_STATE)
    parser.add_argument('--skip-arc-model', action='store_true', help='Train only the transition model.')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    bootstrap.add_argument('--no-config', action='store_true')
    known, _ = bootstrap.parse_known_args(argv)
    config = load_project_config(known.config, use_config=not known.no_config)

    args = _build_parser(config).parse_args(argv)

    training_df = load_csv_playlist(args.train_csv)
    if training_df.empty:
        raise SystemExit(f'training CSV is empty: {args.train_csv}')

    transition_model = train_transition_model(
        training_df,
        negative_ratio=args.negative_ratio,
        hard_fraction=args.hard_fraction,
        random_state=args.random_state,
        device=args.device,
    )
    save_transition_model(transition_model, args.transition_model_out)
    summary = transition_model.training_summary
    cv_auc = summary.get('cv_auc')
    print(
        f"Transition model -> {args.transition_model_out}\n"
        f"  positives={summary.get('positive_rows')} negatives={summary.get('negative_rows')} "
        f"features={summary.get('feature_count')} (musical={summary.get('musical_feature_count')})\n"
        f"  hard_fraction={summary.get('hard_fraction')} backend={summary.get('backend')} "
        f"device={summary.get('resolved_device')}\n"
        f"  cv_auc={'n/a' if cv_auc is None else f'{cv_auc:.3f}'}"
    )

    if not args.skip_arc_model:
        arc_model = train_arc_model(training_df, arc=args.arc_profile, random_state=args.random_state)
        save_arc_model(arc_model, args.arc_model_out)
        print(
            f"Arc model -> {args.arc_model_out}\n"
            f"  backend={arc_model.backend} arc={arc_model.arc} "
            f"{arc_model.training_summary or ''}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
