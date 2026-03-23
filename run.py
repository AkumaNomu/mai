"""Simple runner script that uses the `mai` package to reorder a playlist."""
import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from mai.cli_progress import CliProgressRenderer
from mai.config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config
from mai.data import load_csv_playlist, ensure_audio_columns, normalize_audio_feature_columns
from mai.features import add_log_tempo
from mai.playlist_generation import (
    build_transition_report,
    compute_transition_scores,
    generate_playlist_paths,
    ordered_playlist_paths_from_dataframe,
    playlists_to_dataframe,
)
from mai.transition_model import (
    DEFAULT_TRANSITION_MODEL_DEVICE,
    DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO,
    DEFAULT_TRANSITION_MODEL_PATH,
    DEFAULT_TRANSITION_MODEL_RANDOM_STATE,
    load_transition_model_if_exists,
    save_transition_model,
    train_transition_model,
)


logger = logging.getLogger(__name__)
TITLE_DISPLAY_COLUMNS = ['title', 'track_name', 'name', 'video_title']
ARTIST_DISPLAY_COLUMNS = ['artist', 'artists', 'channel_title', 'uploader']
DEFAULT_YOUTUBE_EXPORT_TITLE = 'mai reordered playlist'


def prepare_df(df):
    df = normalize_audio_feature_columns(df)
    df = ensure_audio_columns(df)
    df = add_log_tempo(df, tempo_col='tempo', out_col='log_tempo')
    return df


def playlist_title(base_title: str, playlist_index: int, total_playlists: int) -> str:
    return base_title if total_playlists == 1 else f'{base_title} {int(playlist_index):02d}'


def resolve_youtube_export_base_title(requested_title: str, source_playlist_title: str = '') -> str:
    normalized_requested = str(requested_title or '').strip()
    if normalized_requested.lower() != 'auto':
        return normalized_requested or DEFAULT_YOUTUBE_EXPORT_TITLE
    normalized_playlist_title = str(source_playlist_title or '').strip()
    if normalized_playlist_title:
        return f'{normalized_playlist_title} mai enhanced'
    return DEFAULT_YOUTUBE_EXPORT_TITLE


def default_transition_report_path(output_csv_path: str) -> str:
    output_path = Path(output_csv_path)
    return str(output_path.with_name(f'{output_path.stem}_transition_report.csv'))


def track_display_label(row: pd.Series) -> str:
    title = ''
    artist = ''
    for column in TITLE_DISPLAY_COLUMNS:
        if column in row.index and not pd.isna(row[column]):
            title = str(row[column]).strip()
            if title:
                break
    for column in ARTIST_DISPLAY_COLUMNS:
        if column in row.index and not pd.isna(row[column]):
            artist = str(row[column]).strip()
            if artist:
                break
    if title and artist:
        return f'{artist} - {title}'
    if title:
        return title
    if artist:
        return artist
    if 'video_id' in row.index and not pd.isna(row['video_id']):
        return str(row['video_id']).strip()
    return '<unknown track>'


def print_recommended_order(generated: pd.DataFrame) -> None:
    for playlist_name, playlist_df in generated.groupby('playlist_name', sort=False):
        print(f'Recommended order for {playlist_name}:')
        for _, row in playlist_df.iterrows():
            position = int(row['position']) if 'position' in row and not pd.isna(row['position']) else 0
            rating = row.get('transition_rating_from_previous', '')
            score = row.get('transition_score_from_previous', float('nan'))
            if pd.isna(score):
                transition_text = 'start'
            else:
                transition_text = f'{rating} {float(score):.3f}'
            print(f'  {position:02d}. {track_display_label(row)} [{transition_text}]')


def print_transition_summary(label: str, report_df: pd.DataFrame) -> None:
    if report_df.empty:
        print(f'{label}: not enough tracks to rate transitions.')
        return
    avg_score = float(report_df['transition_score'].mean())
    min_score = float(report_df['transition_score'].min())
    strong_share = float(report_df['transition_score'].ge(0.65).mean())
    print(
        f'{label}: avg {avg_score:.3f}, min {min_score:.3f}, '
        f'strong-or-better {strong_share:.0%} across {len(report_df)} transitions.'
    )


def _bootstrap_config(argv: Sequence[str] | None = None) -> tuple[dict, str, bool]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    bootstrap.add_argument('--no-config', action='store_true')
    known, _ = bootstrap.parse_known_args(argv)
    config = load_project_config(known.config, use_config=not known.no_config)
    return config, str(known.config), bool(known.no_config)


def _add_bool_override(
    parser: argparse.ArgumentParser,
    *,
    true_flag: str,
    false_flag: str,
    dest: str,
    default: bool,
    true_help: str,
    false_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action='store_true', default=bool(default), help=true_help)
    group.add_argument(false_flag, dest=dest, action='store_false', help=false_help)


def build_parser(config: dict, config_path: str, no_config: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='mai playlist reordering')
    parser.add_argument('--config', default=config_path, help='Path to project TOML config')
    parser.add_argument('--no-config', action='store_true', default=no_config, help='Ignore the TOML config and use CLI/default values only')
    parser.add_argument('--csv', help='Path to playlist CSV')
    parser.add_argument('--youtube-playlist', help='YouTube playlist ID or URL to fetch instead of CSV')
    parser.add_argument('--cache-dir', default=get_config_value(config, 'cache.root_dir', 'data/cache'), help='Directory for reusable metadata and feature caches')
    _add_bool_override(
        parser,
        true_flag='--refresh-cache',
        false_flag='--no-refresh-cache',
        dest='refresh_cache',
        default=bool(get_config_value(config, 'analysis.refresh_cache', False)),
        true_help='Ignore reusable caches and refetch/recompute',
        false_help='Reuse caches even if the config enables refresh mode',
    )
    parser.add_argument('--audio-cache', default=get_config_value(config, 'cache.audio_dir', 'data/audio_cache'), help='Directory for cached audio downloads')
    _add_bool_override(
        parser,
        true_flag='--no-audio-analysis',
        false_flag='--audio-analysis',
        dest='no_audio_analysis',
        default=bool(get_config_value(config, 'analysis.no_audio_analysis', False)),
        true_help='Skip audio analysis for YouTube playlists',
        false_help='Force audio analysis even if the config disables it',
    )
    parser.add_argument('--max-tracks', type=int, default=get_config_value(config, 'analysis.max_tracks', None), help='Limit number of tracks to analyze (for speed)')
    parser.add_argument('--edge-seconds', type=float, default=float(get_config_value(config, 'analysis.edge_seconds', 30.0)), help='Analyze first/last non-silent seconds')
    parser.add_argument('--silence-top-db', type=float, default=float(get_config_value(config, 'analysis.silence_top_db', 35.0)), help='Silence threshold for trimming (higher trims more)')
    parser.add_argument('--flow-profile', default=get_config_value(config, 'analysis.flow_profile', 'deep-dj'), choices=['standard', 'deep-dj'], help='Transition edge analysis depth')
    parser.add_argument('--resource-profile', default=get_config_value(config, 'analysis.resource_profile', 'default'), choices=['default', 'background'], help='Resource usage profile: `background` throttles audio analysis so it can run more gently in the background')
    parser.add_argument('--download-workers', type=int, default=int(get_config_value(config, 'analysis.download_workers', 4)), help='Concurrent worker count for YouTube audio downloads')
    parser.add_argument('--analysis-workers', type=int, default=int(get_config_value(config, 'analysis.analysis_workers', 4)), help='Concurrent worker count for CPU-heavy audio feature extraction')
    _add_bool_override(
        parser,
        true_flag='--delete-audio-after-analysis',
        false_flag='--keep-audio-cache',
        dest='delete_audio_after_analysis',
        default=bool(get_config_value(config, 'analysis.delete_audio_after_analysis', True)),
        true_help='Delete downloaded audio files after their features are persisted',
        false_help='Keep downloaded audio files after analysis',
    )
    parser.add_argument('--create-ytmusic', action='store_true', help='Create a YouTube Music playlist with the computed order')
    parser.add_argument('--ytmusic-auth', default=get_config_value(config, 'exports.ytmusic.auth_path', 'data/ytmusic_auth.json'), help='Path to ytmusicapi headers auth JSON')
    parser.add_argument('--ytmusic-title', default=get_config_value(config, 'exports.ytmusic.title', 'mai reordered playlist'), help='Title for created YouTube Music playlist')
    parser.add_argument('--ytmusic-privacy', default=get_config_value(config, 'exports.ytmusic.privacy', 'PRIVATE'), help='Playlist privacy: PRIVATE | PUBLIC | UNLISTED')
    parser.add_argument('--create-youtube', action='store_true', help='Create a standard YouTube playlist with the computed order')
    parser.add_argument('--youtube-client-secrets', default=get_config_value(config, 'exports.youtube.client_secrets_path', 'data/youtube_client_secret.json'), help='Path to Google OAuth desktop client secrets JSON')
    parser.add_argument('--youtube-token', default=get_config_value(config, 'exports.youtube.token_path', 'data/youtube_token.json'), help='Path to cached YouTube OAuth token JSON')
    parser.add_argument('--youtube-title', default=get_config_value(config, 'exports.youtube.title', 'auto'), help='Title for created standard YouTube playlist; use `auto` to derive it from the source playlist name when available')
    parser.add_argument('--youtube-privacy', default=get_config_value(config, 'exports.youtube.privacy', 'unlisted'), help='Playlist privacy: private | public | unlisted')
    parser.add_argument('--playlist-size', type=int, default=get_config_value(config, 'generation.playlist_size', None), help='Tracks per generated playlist (default: all tracks)')
    parser.add_argument('--num-playlists', type=int, default=int(get_config_value(config, 'generation.num_playlists', 1)), help='How many playlists to generate from the pool')
    _add_bool_override(
        parser,
        true_flag='--allow-reuse',
        false_flag='--no-allow-reuse',
        dest='allow_reuse',
        default=bool(get_config_value(config, 'generation.allow_reuse', False)),
        true_help='Allow the same song to appear in multiple generated playlists',
        false_help='Disallow song reuse even if the config enables it',
    )
    parser.add_argument('--genre-column', default=get_config_value(config, 'generation.genre_column', '') or None, help='Existing genre column to use for genre balancing')
    parser.add_argument('--genre-clusters', type=int, default=int(get_config_value(config, 'generation.genre_clusters', 8)), help='Inferred style clusters when no genre column exists')
    parser.add_argument('--beam-width', type=int, default=int(get_config_value(config, 'generation.beam_width', 8)), help='Beam width for playlist generation search')
    parser.add_argument('--candidate-width', '--k', dest='candidate_width', type=int, default=int(get_config_value(config, 'generation.candidate_width', 25)), help='Top transition candidates explored per step')
    parser.add_argument('--input-order-column', default=get_config_value(config, 'generation.input_order_column', '') or None, help='Column to use when rating the current playlist order (defaults to row order or existing `position`)')
    parser.add_argument('--train-transition-model', action='store_true', help='Train a supervised transition model from the positive transition CSV before playlist scoring')
    parser.add_argument('--transition-model-train-csv', default=get_config_value(config, 'training.output_path', 'data/training/positive_transitions.csv'), help='CSV file containing positive transition rows for model training')
    parser.add_argument('--transition-model-out', default=DEFAULT_TRANSITION_MODEL_PATH, help='Path where a trained transition model artifact should be written')
    parser.add_argument('--transition-model-path', default='', help='Path to a saved transition model artifact to load for scoring')
    parser.add_argument('--transition-model-weight', type=float, default=0.0, help='Weight to give the transition model component when blending transition scores')
    parser.add_argument('--transition-model-negative-ratio', type=float, default=DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO, help='Number of synthetic negatives to create per positive training row')
    parser.add_argument('--transition-model-random-state', type=int, default=DEFAULT_TRANSITION_MODEL_RANDOM_STATE, help='Random seed used when synthesizing negative training rows')
    parser.add_argument('--transition-model-device', choices=['cuda', 'cpu', 'auto'], default=get_config_value(config, 'training.transition_model_device', DEFAULT_TRANSITION_MODEL_DEVICE), help='Training device for transition model (`cuda` uses GPU, `cpu` uses CPU, `auto` picks CUDA when available)')
    _add_bool_override(
        parser,
        true_flag='--rate-transitions',
        false_flag='--no-rate-transitions',
        dest='rate_transitions',
        default=bool(get_config_value(config, 'generation.rate_transitions', False)),
        true_help='Write a CSV report rating transitions for the input order and recommended order',
        false_help='Skip transition report generation even if the config enables it',
    )
    parser.add_argument('--transition-report-out', default=get_config_value(config, 'generation.transition_report_out', '') or None, help='Path for the transition report CSV')
    _add_bool_override(
        parser,
        true_flag='--print-recommended-order',
        false_flag='--no-print-recommended-order',
        dest='print_recommended_order',
        default=bool(get_config_value(config, 'generation.print_recommended_order', False)),
        true_help='Print the recommended order and per-transition ratings to the terminal',
        false_help='Suppress recommended-order printing even if the config enables it',
    )
    parser.add_argument('--log-level', default=get_config_value(config, 'logging.level', 'INFO'), choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'], help='Verbosity')
    return parser


def main(argv: Sequence[str] | None = None):
    config, config_path, no_config = _bootstrap_config(argv)
    parser = build_parser(config, config_path=config_path, no_config=no_config)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s: %(message)s', force=True)
    progress = CliProgressRenderer()

    try:
        transition_model = None
        if args.train_transition_model:
            progress.section('Training transition model', args.transition_model_train_csv)
            training_df = load_csv_playlist(args.transition_model_train_csv)
            training_df = prepare_df(training_df)
            if training_df.empty:
                raise SystemExit(f'transition model training CSV is empty: {args.transition_model_train_csv}')
            transition_model = train_transition_model(
                training_df,
                negative_ratio=float(args.transition_model_negative_ratio),
                random_state=int(args.transition_model_random_state),
                device=str(args.transition_model_device),
            )
            save_transition_model(transition_model, args.transition_model_out)
            progress.success(
                'Transition model trained',
                f"{transition_model.training_summary.get('positive_rows', 0)} positive rows, "
                f"{transition_model.training_summary.get('negative_rows', 0)} synthetic negatives",
            )
            print('Wrote transition model to', args.transition_model_out)

        source_playlist_title = ''
        if args.youtube_playlist:
            try:
                from mai.youtube_integration import parse_youtube_playlist_id, fetch_youtube_playlist_tracks
                from mai.audio_analysis import analyze_youtube_playlist_audio
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    'YouTube playlist input requires yt-dlp and librosa-related dependencies. Install requirements.txt and retry.'
                ) from exc
            playlist_id = parse_youtube_playlist_id(args.youtube_playlist)
            logger.info('Fetching YouTube playlist metadata: %s', playlist_id)
            df = fetch_youtube_playlist_tracks(
                playlist_id,
                limit=args.max_tracks,
                cache_dir=os.path.join(args.cache_dir, 'youtube_playlists'),
                refresh=args.refresh_cache,
            )
            source_playlist_title = str(df.attrs.get('source_playlist_title') or '').strip()
            logger.info('Fetched %d tracks.', len(df))
            if not args.no_audio_analysis:
                logger.info(
                    'Analyzing audio (edge_seconds=%.1f, silence_top_db=%.1f, flow_profile=%s, resource_profile=%s)...',
                    args.edge_seconds,
                    args.silence_top_db,
                    args.flow_profile,
                    args.resource_profile,
                )
                df = analyze_youtube_playlist_audio(
                    df,
                    audio_cache_dir=args.audio_cache,
                    feature_cache_dir=os.path.join(args.cache_dir, 'audio_features.sqlite'),
                    max_tracks=args.max_tracks,
                    edge_seconds=args.edge_seconds,
                    silence_top_db=args.silence_top_db,
                    flow_profile=args.flow_profile,
                    resource_profile=args.resource_profile,
                    refresh_cache=args.refresh_cache,
                    download_workers=args.download_workers,
                    analysis_workers=args.analysis_workers,
                    delete_audio_after_analysis=args.delete_audio_after_analysis,
                    progress_callback=progress.update,
                )
        elif args.csv:
            logger.info('Loading CSV: %s', args.csv)
            df = load_csv_playlist(args.csv)
        else:
            if args.train_transition_model:
                return
            raise SystemExit('Provide --csv or --youtube-playlist')

        df = prepare_df(df)
        playlist_size = args.playlist_size or len(df)
        if playlist_size > len(df) and not args.allow_reuse:
            raise SystemExit('playlist-size cannot exceed the available track count unless --allow-reuse is set')

        logger.info('Computing directed transition scores for %d tracks...', len(df))
        if transition_model is None and args.transition_model_path:
            transition_model = load_transition_model_if_exists(args.transition_model_path)
            if transition_model is None:
                logger.warning('Transition model artifact not found at %s; continuing without the ML component.', args.transition_model_path)
        transition_model_weight = float(args.transition_model_weight) if transition_model is not None else 0.0
        transition_scores, scored_df = compute_transition_scores(
            df,
            flow_profile=args.flow_profile,
            transition_model=transition_model,
            transition_model_weight=transition_model_weight,
            progress_callback=progress.update,
        )
        try:
            input_paths_with_labels = ordered_playlist_paths_from_dataframe(scored_df, order_column=args.input_order_column)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        input_labels = [label for label, _ in input_paths_with_labels]
        input_paths = [path for _, path in input_paths_with_labels]
        input_transition_report = build_transition_report(
            scored_df,
            transition_scores,
            input_paths,
            playlist_labels=input_labels,
            report_name='input_order',
        )
        logger.info(
            'Generating %d playlist(s) of size %d with beam_width=%d and candidate_width=%d...',
            args.num_playlists,
            playlist_size,
            args.beam_width,
            args.candidate_width,
        )
        paths, scored_df, resolved_genre_column = generate_playlist_paths(
            scored_df,
            transition_scores,
            playlist_size=playlist_size,
            num_playlists=args.num_playlists,
            genre_column=args.genre_column,
            genre_clusters=args.genre_clusters,
            beam_width=args.beam_width,
            candidate_width=args.candidate_width,
            allow_reuse=args.allow_reuse,
            progress_callback=progress.update,
        )
        logger.info('Genre balancing uses column: %s', resolved_genre_column)

        generated = playlists_to_dataframe(scored_df, paths, transition_scores=transition_scores)
        is_full_reorder = args.num_playlists == 1 and playlist_size == len(scored_df) and not args.allow_reuse
        out_path = 'data/Playlist_reordered.csv' if is_full_reorder else 'data/Generated_playlists.csv'
        generated.to_csv(out_path, index=False)
        print('Wrote playlist CSV to', out_path)
        recommended_labels = [f'playlist_{playlist_index:02d}' for playlist_index in range(1, len(paths) + 1)]
        recommended_transition_report = build_transition_report(
            scored_df,
            transition_scores,
            paths,
            playlist_labels=recommended_labels,
            report_name='recommended_order',
        )
        if is_full_reorder:
            print_transition_summary('Input order', input_transition_report)
            print_transition_summary('Recommended order', recommended_transition_report)
        if args.rate_transitions:
            transition_report = pd.concat(
                [report for report in [input_transition_report, recommended_transition_report] if not report.empty],
                ignore_index=True,
            )
            if not transition_report.empty:
                transition_report_out = args.transition_report_out or default_transition_report_path(out_path)
                Path(transition_report_out).parent.mkdir(parents=True, exist_ok=True)
                transition_report.to_csv(transition_report_out, index=False)
                print('Wrote transition report to', transition_report_out)
            else:
                print('No transition report was written because there were not enough tracks to rate.')
        if args.print_recommended_order:
            print_recommended_order(generated)

        if args.create_ytmusic or args.create_youtube:
            if 'video_id' not in generated.columns:
                raise SystemExit('Cannot create YouTube playlist exports: missing video_id column')

        grouped_playlists = list(generated.groupby('playlist_index', sort=True))

        if args.create_ytmusic:
            try:
                from mai.ytmusic_integration import load_ytmusic, create_reordered_playlist
            except ModuleNotFoundError as exc:
                raise SystemExit('YouTube Music export requires ytmusicapi. Install requirements.txt and retry.') from exc
            ytm = load_ytmusic(args.ytmusic_auth)
            created_urls = []
            for completed, (playlist_index, playlist_df) in enumerate(grouped_playlists, start=1):
                video_ids = playlist_df['video_id'].dropna().astype(str).tolist()
                title = playlist_title(args.ytmusic_title, int(playlist_index), len(grouped_playlists))
                progress.update('Exporting YouTube Music', completed - 1, len(grouped_playlists), title)
                logger.info('Creating YouTube Music playlist: %s (%s)', title, args.ytmusic_privacy)
                playlist_id = create_reordered_playlist(
                    ytm,
                    title=title,
                    video_ids=video_ids,
                    privacy_status=args.ytmusic_privacy
                )
                created_urls.append(f'https://music.youtube.com/playlist?list={playlist_id}')
                progress.update('Exporting YouTube Music', completed, len(grouped_playlists), title)
            for url in created_urls:
                print('Created YouTube Music playlist:', url)

        if args.create_youtube:
            try:
                from mai.youtube_export import load_youtube_service, create_youtube_playlist
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    'Standard YouTube export requires google-api-python-client, google-auth-oauthlib, and google-auth-httplib2. '
                    'Install requirements.txt and retry.'
                ) from exc
            youtube = load_youtube_service(
                client_secrets_path=args.youtube_client_secrets,
                token_path=args.youtube_token,
            )
            youtube_export_base_title = resolve_youtube_export_base_title(
                args.youtube_title,
                source_playlist_title=source_playlist_title,
            )
            for completed, (playlist_index, playlist_df) in enumerate(grouped_playlists, start=1):
                video_ids = playlist_df['video_id'].dropna().astype(str).tolist()
                title = playlist_title(youtube_export_base_title, int(playlist_index), len(grouped_playlists))
                progress.update('Exporting YouTube', completed - 1, len(grouped_playlists), title)
                logger.info('Creating standard YouTube playlist: %s (%s)', title, args.youtube_privacy)
                result = create_youtube_playlist(
                    youtube,
                    title=title,
                    video_ids=video_ids,
                    privacy_status=args.youtube_privacy,
                )
                print('Created YouTube playlist:', result['playlist_url'])
                if result['failed_video_ids']:
                    print('Skipped YouTube playlist items:', ', '.join(result['failed_video_ids']))
                progress.update('Exporting YouTube', completed, len(grouped_playlists), title)
    finally:
        progress.close()

if __name__ == '__main__':
    main()
