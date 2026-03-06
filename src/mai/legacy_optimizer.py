import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
import sys
from tqdm import tqdm
import time

# Fix Windows UTF-8 encoding issue
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

# Load environment variables
load_dotenv()

# Initialize Spotify client with user authentication for creating playlists
sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri='http://127.0.0.1:8888/callback',
    scope='playlist-modify-private'
))


def get_input_source():
    """
    Ask user for a CSV file path.
    
    Returns:
        Tuple of (source_type, path, name)
    """
    print_header("LOAD PLAYLIST")
    
    while True:
        csv_path = input(f"{Colors.BOLD}Enter path to CSV file (Exportify format): {Colors.END}").strip()
        if os.path.exists(csv_path):
            print_success(f"Found CSV: {csv_path}")
            # Extract name from filename
            name = os.path.basename(csv_path).replace('.csv', '')
            return 'csv', csv_path, name
        else:
            print_error(f"File not found: {csv_path}")
            print_info("Get CSV files from: https://exportify.net/")

def fetch_playlist_tracks(csv_path):
    """
    Load tracks from an exported CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (DataFrame with track info, list of track URIs)
    """
    print_info("Loading playlist from CSV...")
    
    df = pd.read_csv(csv_path)
    print_success(f"Loaded {len(df)} tracks")
    
    # Verify we have track URIs
    if 'Track URI' not in df.columns:
        print_error("Track URI column not found in CSV")
        return None, None
    
    track_uris = df['Track URI'].tolist()
    
    return df, track_uris


def normalize(df, column):
    """Normalize a column to 0-1 range."""
    col_max = df[column].max()
    col_min = df[column].min()
    df[column] = (df[column] - col_min) / (col_max - col_min)

def key_similarity(key1, mode1, key2, mode2):
    """
    Calculate similarity between two musical keys (0-1 scale, 1 = identical).
    Uses circle of fifths harmonic distance.
    Handles both numeric keys (0-11 from Spotify) and string keys.
    
    Args:
        key1, key2: int or str - Key (0-11 for numeric, or 'C', 'F#', 'Bb' for strings)
        mode1, mode2: int or str - 'major'(1) or 'minor'(0)
    
    Returns:
        float: Similarity score (0 = maximally distant, 1 = identical)
    """
    
    # Convert minor keys to their relative major (3 semitones up)
    # On circle of fifths: minor = 3 positions counter-clockwise from relative major
    def to_major_position(key, mode):
        pos = key
        if mode == 0:
            # A minor -> C major (shift 3 positions clockwise)
            pos = (pos + 3) % 12
        return pos
    
    pos1 = to_major_position(key1, mode1)
    pos2 = to_major_position(key2, mode2)
    
    # Calculate shortest distance around the circle (0-6)
    distance = abs(pos1 - pos2)
    distance = min(distance, 12 - distance)
    
    # Convert to similarity score (0-1)
    # Distance 0 = identical (similarity 1.0)
    # Distance 6 = opposite (similarity 0.0)
    similarity = 1 - (distance / 6)
    
    return similarity


features = ['Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo']

def vector_mod(row):
    s = 0.0
    for col in features:
        s += row[col] ** 2
    return s ** 0.5

def dot_product(row1, row2):
    s = 0.0
    for col in features:
        s += row1[col] * row2[col]
    return s

def cosine_similarity(row1, row2):
    dp = dot_product(row1, row2)
    mod1 = vector_mod(row1)
    mod2 = vector_mod(row2)
    if mod1 == 0 or mod2 == 0:
        return 0.0
    return dp / (mod1 * mod2)

def similarity_score(df, song1, song2):
    """
    Calculate comprehensive similarity between two songs.
    Optimized for smooth playlist transitions.
    
    Considers:
    - Harmonic compatibility (30%)
    - Audio feature similarity (40%)
    - Mood compatibility (Valence/Energy) (15%)
    - Tempo smoothness (15%)
    """
    first = df.iloc[song1]
    second = df.iloc[song2]
    
    # 1. Harmonic similarity (30% weight)
    key_sim = key_similarity(first['Key'], first['Mode'], second['Key'], second['Mode'])
    
    # 2. Audio feature similarity (40% weight)
    # Features: Danceability, Energy, Speechiness, Acousticness, Liveness, Valence, Tempo
    cosine_sim = cosine_similarity(first, second)
    
    # 3. Mood compatibility (15% weight)
    # Valence (1=positive/happy, 0=negative) and Energy should match somewhat
    mood_sim = 1.0 - abs(first.get('Valence', 0.5) - second.get('Valence', 0.5)) * 0.3
    energy_sim = 1.0 - abs(first.get('Energy', 0.5) - second.get('Energy', 0.5)) * 0.3
    mood_compatibility = (mood_sim + energy_sim) / 2
    
    # 4. Tempo smoothness (15% weight)
    # Prefer similar tempos for natural flow
    tempo_diff = abs(first['Tempo'] - second['Tempo'])
    max_tempo = max(first['Tempo'], second['Tempo'])
    tempo_smoothness = 1.0 - min(tempo_diff / max(max_tempo, 1), 1.0) * 0.5
    
    # Combined score with optimized weights
    overall_score = (
        0.30 * key_sim +
        0.40 * cosine_sim +
        0.15 * mood_compatibility +
        0.15 * tempo_smoothness
    )
    
    return overall_score

def calculate_edges(df):
    """Calculate similarity edges between all pairs of songs."""
    print_info("Calculating song similarities...")
    edges = []
    
    for i in tqdm(range(len(df)), desc="Comparing", unit=" songs"):
        for j in range(i+1, len(df)):
            sim = similarity_score(df, i, j)
            edges.append((i, j, sim))
    
    return edges

import networkx as nx

def optimize_playlist_order(G, method='2opt_multistart'):
    """
    Optimized playlist ordering with smooth transitions
    
    Args:
        G: NetworkX graph with similarity weights
        method: 'greedy', 'multistart', '2opt', '2opt_multistart', 'mst', 'mst_2opt'
    """
    if method == '2opt_multistart':
        # Best approach: multiple starts + local optimization
        path, score = find_best_path_multistart(G, num_starts=20)
        path = two_opt_improve(G, path)
        return path
    # ... other methods

def find_best_path_multistart(G, num_starts=None):
    """
    Try multiple starting points with smart heuristics.
    Uses high-weight nodes as starting points for better initial solutions.
    """
    all_nodes = list(G.nodes())
    if num_starts is None:
        num_starts = min(max(5, len(all_nodes) // 10), 20)
    
    # Find high-weight nodes (good starting points)
    node_avg_weights = {}
    for node in all_nodes:
        neighbors = G[node]
        if neighbors:
            avg_weight = sum(neighbors[n]['weight'] for n in neighbors) / len(neighbors)
            node_avg_weights[node] = avg_weight
    
    # Sort by average neighbor weight - start with well-connected nodes
    sorted_nodes = sorted(node_avg_weights.items(), key=lambda x: x[1], reverse=True)
    start_nodes = [node for node, _ in sorted_nodes[:num_starts]]
    
    best_path = None
    best_score = -float('inf')
    
    for start_node in tqdm(start_nodes, desc="Finding best start", unit=" path"):
        path = greedy_path(G, start_node, all_nodes)
        score = calculate_path_score(G, path)
        
        if score > best_score:
            best_score = score
            best_path = path
    
    return best_path, best_score

def greedy_path(G, start, all_nodes):
    """Build path greedily from start node"""
    path = [start]
    visited = {start}
    current = start
    
    while len(visited) < len(all_nodes):
        neighbors = [(n, G[current][n]['weight']) for n in all_nodes 
                     if n not in visited]
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda x: x[1])[0]
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    
    return path

def two_opt_improve(G, path, max_iterations=None):
    """
    Iteratively improve path by reversing segments (2-opt algorithm).
    Thorough optimization for best results regardless of runtime.
    """
    if max_iterations is None:
        # No iteration limit - optimize until no improvements found
        max_iterations = float('inf')
    
    improved = True
    iteration = 0
    best_score = calculate_path_score(G, path)
    
    with tqdm(total=max_iterations, desc="Optimizing", unit=" iteration") as pbar:
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(path) - 2):
                for j in range(i + 2, min(i + 15, len(path))):  # Limit search window for efficiency
                    if j >= len(path):
                        continue
                    
                    # Calculate gain from reversing path[i:j]
                    current_edges = (
                        G[path[i-1]][path[i]]['weight'] + 
                        G[path[j-1]][path[j]]['weight'] if j < len(path) else 
                        G[path[i-1]][path[i]]['weight']
                    )
                    
                    new_edges = (
                        G[path[i-1]][path[j-1]]['weight'] + 
                        G[path[i]][path[j]]['weight'] if j < len(path) else
                        G[path[i-1]][path[j-1]]['weight']
                    )
                    
                    if new_edges > current_edges:
                        path[i:j] = list(reversed(path[i:j]))
                        new_score = calculate_path_score(G, path)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
            
            pbar.update(1)
    
    return path

def calculate_path_score(G, path):
    """Calculate total similarity across all transitions"""
    return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

def create_optimized_playlist(playlist_name, track_uris):
    """
    Create a new playlist on Spotify with the optimized track order.
    
    Args:
        playlist_name: Name for the new playlist
        track_uris: List of track URIs in optimized order
    
    Returns:
        Playlist URL
    """
    print_info("Creating new playlist...")
    
    # Get current user
    user = sp.current_user()
    user_id = user['id']
    
    # Create playlist
    playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
    playlist_id = playlist['id']
    
    print_success(f"Created playlist: {playlist_name}")
    
    # Add tracks in optimized order (max 100 per request)
    print_info("Adding tracks in optimized order...")
    
    for i in tqdm(range(0, len(track_uris), 100), desc="Adding tracks", unit=" batch"):
        batch = track_uris[i:i+100]
        sp.playlist_add_items(playlist_id, batch)
    
    print_success(f"Added {len(track_uris)} tracks")
    
    return playlist['external_urls']['spotify']

def main():
    """Main application flow."""
    print_header("SPOTIFY PLAYLIST OPTIMIZER")
    
    # Get input source from user
    source_type, source_path, playlist_name = get_input_source()
    
    print_header(f"LOADING: {playlist_name}")
    
    # Load from CSV
    df, track_uris = fetch_playlist_tracks(source_path)
    
    if df is None:
        return
    
    print_header(f"ANALYZING: {playlist_name}")
    print_info(f"Playlist size: {len(df)} tracks")
    
    # Show original stats
    print_info(f"Avg Tempo: {df['Tempo'].mean():.1f} BPM")
    print_info(f"Avg Energy: {df['Energy'].mean():.2f}")
    print_info(f"Avg Valence: {df['Valence'].mean():.2f}")
    
    # Normalize tempo
    normalize(df, 'Tempo')
    
    # Calculate similarities
    edges = calculate_edges(df)
    
    # Build graph and optimize
    print_header("OPTIMIZING PLAYLIST")
    print_info("Building song similarity graph...")
    
    import networkx as nx
    G = nx.Graph()
    
    for i in range(len(df)):
        G.add_node(i)
    
    for i, j, weight in edges:
        G.add_edge(i, j, weight=weight)
    
    # Run optimization
    print_info("Finding optimal track order (this may take a moment)...")
    optimized_path = optimize_playlist_order(G)
    
    # Calculate flow quality
    flow_score = calculate_path_score(G, optimized_path)
    avg_similarity = flow_score / (len(optimized_path) - 1) if len(optimized_path) > 1 else 0
    
    # Display results
    print_header("OPTIMIZED PLAYLIST ORDER")
    print_info(f"Flow Quality Score: {flow_score:.2f}")
    print_info(f"Average Track Similarity: {avg_similarity:.2f}/1.00\n")
    
    print(f"\n{Colors.BOLD}{'#':<4} {'Track Name':<45} {'Artist':<30}{Colors.END}")
    print("-" * 80)
    
    for idx, song_idx in enumerate(optimized_path, 1):
        track_name = df.iloc[song_idx].get('Track Name', f'Track {song_idx}')[:42]
        artist_name = df.iloc[song_idx].get('Artist Name(s)', 'Unknown')[:27]
        print(f"{idx:<4} {track_name:<45} {artist_name:<30}")
    
    # Show transition quality option
    print()
    show_transitions = input(f"{Colors.BOLD}Show transition scores? (y/n): {Colors.END}").lower()
    if show_transitions == 'y':
        print(f"\n{Colors.BOLD}{'Transition':<30} {'Score':<10}{Colors.END}")
        print("-" * 40)
        for i in range(len(optimized_path) - 1):
            from_idx = optimized_path[i]
            to_idx = optimized_path[i + 1]
            score = G[from_idx][to_idx]['weight']
            from_name = df.iloc[from_idx]['Track Name'][:27]
            to_name = df.iloc[to_idx]['Track Name'][:27]
            print(f"{from_name:<30} → {score:.3f}")
    
    # Create new playlist
    print()
    while True:
        create = input(f"{Colors.BOLD}Create optimized playlist on Spotify? (y/n): {Colors.END}").lower()
        if create == 'y':
            # Get ordered track URIs
            ordered_uris = [track_uris[i] for i in optimized_path]
            new_playlist_name = f"{playlist_name} (Optimized)"
            
            try:
                playlist_url = create_optimized_playlist(new_playlist_name, ordered_uris)
                
                print_header("SUCCESS!")
                print_success(f"Playlist created: {new_playlist_name}")
                print_info(f"Open in Spotify: {playlist_url}")
            except Exception as e:
                print_error(f"Failed to create playlist: {e}")
                print_info("Your optimized track order is shown above.")
            break
        elif create == 'n':
            print_info("Skipped playlist creation.")
            break
        else:
            print_error("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()
