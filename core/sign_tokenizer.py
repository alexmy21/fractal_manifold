"""
Tokenization and N-gram Generation for Uninflected Sign Systems.

This module provides tokenization utilities for sign systems where each symbol
is a complete, unmodifiable semantic unit. This is distinct from inflected
languages (like English) where words change form.

Examples of uninflected sign systems:
- Chinese/Japanese/Korean characters (hieroglyphs)
- Musical notation symbols
- Mathematical notation
- Chemical formulas
- Emoji sequences
- Traffic signs
- Braille

The key property: signs combine to form meaning but each sign is atomic.
Unlike "running" → "run" + "-ing" in English, Chinese "学习" is just "学" + "习".

Usage:
    from core.sign_tokenizer import tokenize_signs, generate_sign_ngrams
    
    # Tokenize any sign sequence
    signs = tokenize_signs("学习很有趣")  # ['学', '习', '很', '有', '趣']
    
    # Generate n-grams
    ngrams = generate_sign_ngrams(signs, max_n=3)
"""

import unicodedata
from typing import List, Tuple, Callable, Optional

# Boundary markers (same as mf_algebra)
START: Tuple[str, ...] = ("<START>",)
END: Tuple[str, ...] = ("<END>",)

# Import cascading_disambiguate for query_signs (lazy import to avoid circular dependency)
_cascading_disambiguate = None

def _get_cascading_disambiguate():
    """Lazy import of cascading_disambiguate to avoid circular imports."""
    global _cascading_disambiguate
    if _cascading_disambiguate is None:
        from core.mf_algebra import cascading_disambiguate
        _cascading_disambiguate = cascading_disambiguate
    return _cascading_disambiguate


def tokenize_signs(
    text: str,
    skip_whitespace: bool = True,
    skip_punctuation: bool = True,
    custom_filter: Optional[Callable[[str], bool]] = None
) -> List[str]:
    """
    Tokenize text into individual signs (atomic units).
    
    For uninflected sign systems, each character is typically an atomic unit.
    This is the default behavior. Custom filters can be provided for specific
    sign systems.
    
    Args:
        text: Input text/sign sequence
        skip_whitespace: Skip whitespace characters
        skip_punctuation: Skip punctuation (Unicode category 'P*')
        custom_filter: Optional function(char) -> bool, True to include
        
    Returns:
        List of sign strings (usually single characters)
        
    Examples:
        # Chinese
        tokenize_signs("机器学习")  # ['机', '器', '学', '习']
        
        # Music (if encoded as text)
        tokenize_signs("♩♪♫♬")  # ['♩', '♪', '♫', '♬']
        
        # Emoji
        tokenize_signs("🎵🎶🎼")  # ['🎵', '🎶', '🎼']
    """
    result = []
    
    for char in text:
        # Apply custom filter first if provided
        if custom_filter is not None:
            if custom_filter(char):
                result.append(char)
            continue
        
        # Default filtering
        if skip_whitespace and char.isspace():
            continue
        if skip_punctuation and unicodedata.category(char).startswith('P'):
            continue
        
        result.append(char)
    
    return result


def generate_sign_ngrams(
    signs: List[str],
    max_n: int = 3,
    include_boundaries: bool = True
) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from a sequence of signs.
    
    Pattern: START → 1-grams → 2-grams → ... → n-grams → END
    
    For uninflected sign systems, each sign is already atomic,
    so 1-grams are individual signs, 2-grams are sign pairs, etc.
    
    Args:
        signs: List of signs (from tokenize_signs)
        max_n: Maximum n-gram size (default 3 for trigrams)
        include_boundaries: Include START and END markers
        
    Returns:
        List of n-gram tuples
        
    Example:
        signs = ['学', '习', '是']
        ngrams = generate_sign_ngrams(signs, max_n=2)
        # [START, ('学',), ('学', '习'), ('习',), ('习', '是'), ('是',), END]
    """
    ngrams = []
    
    if include_boundaries:
        ngrams.append(START)
    
    for i in range(len(signs)):
        for n in range(1, min(max_n + 1, len(signs) - i + 1)):
            ngram = tuple(signs[i:i+n])
            ngrams.append(ngram)
    
    if include_boundaries:
        ngrams.append(END)
    
    return ngrams


def ngram_to_string(ngram: Tuple[str, ...], separator: str = '') -> str:
    """
    Convert n-gram tuple to display string.
    
    Args:
        ngram: N-gram tuple
        separator: String to join signs (empty for Chinese, space for words)
        
    Returns:
        Display string
    """
    if ngram == START:
        return "<START>"
    if ngram == END:
        return "<END>"
    return separator.join(ngram)


# =============================================================================
# COVER BUILDING FROM W LATTICE
# =============================================================================

def build_cover_from_rows(
    query_indices: set,
    W: dict,
    max_depth: int = 2
) -> set:
    """
    Build forward cover: all indices reachable FROM query via W rows.
    
    W[layer][row][col] = P(row → col)
    This follows OUTGOING transitions.
    
    Args:
        query_indices: Starting set of indices
        W: Transition matrix W[layer][row][col] = probability
        max_depth: BFS depth limit
        
    Returns:
        Set of indices reachable from query
    """
    cover = set(query_indices)
    frontier = set(query_indices)
    
    for depth in range(max_depth):
        new_frontier = set()
        for idx in frontier:
            for layer in W.values():
                if idx in layer:  # idx is a row (source)
                    for target_idx in layer[idx].keys():
                        if target_idx not in cover:
                            new_frontier.add(target_idx)
                            cover.add(target_idx)
        frontier = new_frontier
        if not frontier:
            break
    
    return cover


def build_cover_from_cols(
    query_indices: set,
    W: dict,
    max_depth: int = 2
) -> set:
    """
    Build backward cover: all indices that REACH query via W columns.
    
    W[layer][row][col] = P(row → col)
    This follows INCOMING transitions (reverse lookup).
    
    Args:
        query_indices: Target set of indices
        W: Transition matrix W[layer][row][col] = probability
        max_depth: BFS depth limit
        
    Returns:
        Set of indices that can reach query
    """
    cover = set(query_indices)
    frontier = set(query_indices)
    
    for depth in range(max_depth):
        new_frontier = set()
        for idx in frontier:
            # Find all sources that transition TO idx
            for layer in W.values():
                for src_idx, targets in layer.items():
                    if idx in targets and src_idx not in cover:
                        new_frontier.add(src_idx)
                        cover.add(src_idx)
        frontier = new_frontier
        if not frontier:
            break
    
    return cover


def build_cover_from_w(
    query_indices: set,
    W: dict,
    max_depth: int = 2,
    strategy: str = 'intersection'
) -> set:
    """
    Build cover HLLSet from W lattice transitions.
    
    Args:
        query_indices: Starting indices
        W: Transition matrix W[layer][row][col] = probability
        max_depth: BFS depth limit
        strategy: Cover strategy:
            - 'intersection': row_cover ∩ col_cover (most precise)
            - 'union': row_cover ∪ col_cover (broader recall)
            - 'rows_only': Forward reachability only
            - 'cols_only': Backward reachability only
    
    Returns:
        Cover set of related indices
        
    The INTERSECTION strategy finds indices that are:
    - Reachable from query (forward context via rows)
    - AND can reach query (backward context via cols)
    
    This is more precise than union, as it finds indices
    that are truly "between" the query and its context.
    """
    row_cover = build_cover_from_rows(query_indices, W, max_depth)
    col_cover = build_cover_from_cols(query_indices, W, max_depth)
    
    if strategy == 'intersection':
        # Indices in BOTH forward and backward traversal
        # Always include query indices themselves
        return (row_cover & col_cover) | query_indices
    elif strategy == 'union':
        return row_cover | col_cover
    elif strategy == 'rows_only':
        return row_cover
    elif strategy == 'cols_only':
        return col_cover
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# QUERY PIPELINE
# =============================================================================

def query_signs(
    query_text: str,
    lut,  # LookupTable
    am,   # SparseAM3D
    W: dict,
    layer_hllsets,  # LayerHLLSets
    max_n: int = 3,
    top_k: int = 10,
    cover_strategy: str = 'intersection',
    verbose: bool = True
) -> list:
    """
    Complete query pipeline for uninflected sign systems.
    
    Steps:
    1. Tokenize query into signs
    2. Generate ALL n-grams from query signs
    3. Look up indices for ALL n-grams in LUT
    4. Build cover from W using row/col intersection
    5. Disambiguate cover using cascading algorithm
    6. Recover tokens from LUT and order by layer
    
    Args:
        query_text: Query string
        lut: LookupTable with indexed n-tokens
        am: SparseAM3D adjacency matrix
        W: Transition probability matrix
        layer_hllsets: Layer HLLSets for disambiguation
        max_n: Maximum n-gram size
        top_k: Number of results to return
        cover_strategy: 'intersection', 'union', 'rows_only', 'cols_only'
        verbose: Print progress
        
    Returns:
        List of result dicts with 'token', 'layer', 'index', etc.
    """
    # Get cascading_disambiguate via lazy import
    cascading_disambiguate = _get_cascading_disambiguate()
    if verbose:
        print(f"Query: {query_text} (strategy={cover_strategy})")
    
    # Step 1: Tokenize → indices IMMEDIATELY (HLLSet-early principle)
    # Convert input to index set as fast as possible, defer token resolution to end
    signs = tokenize_signs(query_text)
    
    # Step 2: Generate n-grams and convert to indices (not storing n-gram tuples)
    query_ngrams = generate_sign_ngrams(signs, max_n=max_n)
    
    # Step 3: Build index set (pure indices, no token info carried forward)
    query_indices = set()
    found_count = 0
    
    for ng in query_ngrams:
        if ng in (START, END):
            continue
        idx = lut.get_ntoken_index(ng)
        if idx is not None and (idx in am.all_active_rows or idx in am.all_active_cols):
            query_indices.add(idx)
            found_count += 1
    
    if verbose:
        print(f"  Signs: {len(signs)}, Indices: {len(query_indices)} (from {found_count} n-grams)")
    
    if not query_indices:
        if verbose:
            print("  ⚠️ Query not found in corpus")
        return []
    
    # Step 4: Build cover
    row_cover = build_cover_from_rows(query_indices, W, max_depth=2)
    col_cover = build_cover_from_cols(query_indices, W, max_depth=2)
    cover = build_cover_from_w(query_indices, W, max_depth=2, strategy=cover_strategy)
    
    if verbose:
        print(f"  Cover: rows={len(row_cover)}, cols={len(col_cover)}, "
              f"intersection={len(row_cover & col_cover)}, using={len(cover)}")
    
    # Step 5: Disambiguate
    results = cascading_disambiguate(cover, am, layer_hllsets, W, lut)
    
    # Step 6: Recover tokens
    recovered = []
    seen_tokens = set()
    
    for result in results:
        ntokens = lut.get_ntokens_at_index(result.index)
        for layer, ntoken in ntokens:
            if ntoken not in (START, END):
                display = ngram_to_string(ntoken)
                if display not in seen_tokens:
                    seen_tokens.add(display)
                    recovered.append({
                        'token': display,
                        'layer': layer,
                        'index': result.index,
                        'constituents': len(result.constituent_indices),
                        'ntoken': ntoken
                    })
    
    # Order by layer (higher = more context)
    recovered.sort(key=lambda x: (-x['layer'], x['index']))
    
    if verbose:
        print(f"  Results: {len(recovered)} tokens")
        for layer in [2, 1, 0]:
            layer_results = [r for r in recovered if r['layer'] == layer]
            if layer_results:
                samples = [r['token'] for r in layer_results[:3]]
                print(f"    L{layer}: {samples}{'...' if len(layer_results) > 3 else ''}")
    
    return recovered[:top_k]
