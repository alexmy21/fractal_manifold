"""
Perceptrons and Actuators Module - Sense-Process-Act Loop.

This module provides:
- Perceptron: Base class for input sensing (files → HLLSet)
- PromptPerceptron: Process user queries
- Actuator: Base class for actions
- ResponseActuator: Generate responses with feedback loop
- QueryContext: Mutable context for interactive querying
- ask(): Interactive query with feedback loop
- create_query_context(): Initialize query environment

Sense-Process-Act Loop:
    Perceptron (sense) → Pipeline (process) → Actuator (act)
    
Feedback Loop (co-adaptive learning):
    Query → Response → HLLSet → Commit → (shapes future responses)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..sparse_hrt_3d import SparseAM3D, SparseHRT3D, Sparse3DConfig, SparseLattice3D
from .lut import LookupTable, START, END
from .processing import ProcessingResult, unified_process, build_w_from_am
from .stores import Commit, CommitStore
from .disambiguation import LayerHLLSets, DisambiguationResult, cascading_disambiguate
from .sparse_matrices import Sparse3DMatrix
from .operations import project_layer, reachable_from


# ═══════════════════════════════════════════════════════════════════════════
# PERCEPTRON BASE CLASS - Sense Phase
# ═══════════════════════════════════════════════════════════════════════════

class Perceptron(ABC):
    """
    Base class for perceptrons - sense input and convert to HLLSet.
    
    Each perceptron:
    1. Finds/accepts its input type
    2. Extracts text content
    3. Processes via unified pipeline
    4. Commits after processing
    
    Part of sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    """
    
    def __init__(self, name: str, extensions: List[str], config: Sparse3DConfig):
        self.name = name
        self.extensions = extensions
        self.config = config
        self.lut: Optional[LookupTable] = None
        self.files_processed = 0
        self.total_tokens = 0
    
    def initialize(self, lut: LookupTable):
        """Initialize with shared LUT."""
        self.lut = lut
    
    def find_files(self, root: Path, exclude_dirs: Set[str] = None) -> Iterator[Path]:
        """Find all files matching extensions."""
        exclude_dirs = exclude_dirs or {'__pycache__', '.git', 'build', '.ipynb_checkpoints', 'deprecated'}
        
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in self.extensions:
                if not any(ex in path.parts for ex in exclude_dirs):
                    yield path
    
    @abstractmethod
    def extract_text(self, path: Path) -> str:
        """Extract text content from input."""
        pass
    
    def process_file(
        self,
        path: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit]]:
        """
        Process a single file and commit.
        
        Returns (new_hrt, new_W, commit)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        text = self.extract_text(path)
        if not text.strip():
            return current_hrt, current_W, None
        
        # Unified processing
        result = unified_process(
            text,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        # Update state
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit
        commit = store.commit(new_hrt, new_W, str(path), self.name)
        
        self.files_processed += 1
        self.total_tokens = len(self.lut.ntoken_to_index)
        
        return new_hrt, new_W, commit
    
    def process_all(
        self,
        root: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_files: Optional[int] = None,
        max_n: int = 3,
        verbose: bool = True
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Process all files of this type.
        
        Args:
            root: Root directory to search
            max_files: Limit number of files (None = all)
            max_n: Maximum n-gram size
            verbose: Print progress
        """
        files = list(self.find_files(root))
        if max_files:
            files = files[:max_files]
        
        if verbose:
            print(f"[{self.name}] Processing {len(files)} files")
        
        for path in files:
            try:
                current_hrt, current_W, commit = self.process_file(
                    path, current_hrt, current_W, store, max_n
                )
                if verbose and commit:
                    print(f"  ✓ {path.name} [{current_hrt.nnz} edges]")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {path.name}: {e}")
        
        return current_hrt, current_W


class PromptPerceptron(Perceptron):
    """
    Perceptron for user prompts/queries.
    
    Treats user input exactly like file input:
    - Goes through unified pipeline
    - Gets committed
    - Contributes to manifold (learning from queries!)
    """
    
    def __init__(self, config: Sparse3DConfig):
        super().__init__("p_prompt", [], config)
        self.prompt_history: List[str] = []
    
    def extract_text(self, path: Path) -> str:
        """Not used - prompts come directly as text."""
        return ""
    
    def process_prompt(
        self,
        prompt: str,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit], Optional[ProcessingResult]]:
        """
        Process a user prompt and commit.
        
        Returns (new_hrt, new_W, commit, processing_result)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        if not prompt.strip():
            return current_hrt, current_W, None, None
        
        self.prompt_history.append(prompt)
        
        # Unified processing (same as files!)
        result = unified_process(
            prompt,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit with prompt ID as source
        prompt_id = f"prompt_{len(self.prompt_history)}"
        commit = store.commit(new_hrt, new_W, prompt_id, self.name)
        
        self.files_processed += 1
        
        return new_hrt, new_W, commit, result


# ═══════════════════════════════════════════════════════════════════════════
# ACTUATOR BASE CLASS - Act Phase
# ═══════════════════════════════════════════════════════════════════════════

class Actuator(ABC):
    """
    Base class for actuators - turn processed data into action.
    
    Completes the sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    
    Key insight: Actuator output can feed back into the manifold!
    """
    
    def __init__(self, name: str):
        self.name = name
        self.actions_taken = 0
    
    @abstractmethod
    def act(self, commit: Commit, result: ProcessingResult, **kwargs) -> str:
        """
        Perform action based on processed result.
        
        Returns action summary string.
        """
        pass


class ResponseActuator(Actuator):
    """
    Actuator for query responses with FEEDBACK LOOP.
    
    The response itself is ingested back into the manifold!
    This creates co-adaptive learning:
        Query → Response → HLLSet → Commit → (shapes future responses)
    """
    
    def __init__(self):
        super().__init__("a_response")
        self.responses: List[Dict[str, Any]] = []
    
    def act(
        self,
        commit: Commit,
        result: ProcessingResult,
        query_results: List[Tuple[Any, float]] = None,
        hrt: SparseHRT3D = None,
        W: Dict[int, Dict[int, Dict[int, float]]] = None,
        store: CommitStore = None,
        lut: LookupTable = None,
        config: Sparse3DConfig = None,
        ingest_response: bool = True,
        max_n: int = 3,
        **kwargs
    ) -> Tuple[str, SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Generate response and optionally ingest it back.
        
        Returns:
            (response_text, updated_hrt, updated_W)
        """
        import time
        from datetime import datetime
        
        # Build response text
        lines = [
            f"Query: {commit.source}",
            f"Commit: {commit.id[:8]}",
            f"Results ({len(query_results or [])} found):",
        ]
        
        for i, (ntoken, score) in enumerate(query_results or [], 1):
            lines.append(f"  {i:2d}. [{score:5.1f}] {ntoken}")
        
        response_text = "\n".join(lines)
        
        # Track response
        response_record = {
            "timestamp": datetime.fromtimestamp(commit.timestamp).isoformat(),
            "prompt": commit.source,
            "commit_id": commit.id[:8],
            "response": response_text,
            "ingested": False,
        }
        
        new_hrt = hrt
        new_W = W
        
        # FEEDBACK LOOP: Ingest response back into manifold
        if ingest_response and hrt and store and lut and config:
            response_result = unified_process(
                response_text,
                hrt,
                W,
                config,
                lut,
                max_n
            )
            
            new_hrt = response_result.merged_hrt
            new_W = build_w_from_am(new_hrt.am, config)
            
            # Commit response as its own entry
            response_id = f"response_{len(self.responses) + 1}"
            store.commit(new_hrt, new_W, response_id, self.name)
            
            response_record["ingested"] = True
            response_record["response_commit"] = response_id
        
        self.responses.append(response_record)
        self.actions_taken += 1
        
        return response_text, new_hrt, new_W
    
    def history(self) -> List[Dict[str, Any]]:
        """Get response history."""
        return self.responses


# ═══════════════════════════════════════════════════════════════════════════
# QUERY INTERFACE - ask() function
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryContext:
    """
    Mutable context for interactive querying.
    
    Holds the current state that gets updated with each query.
    """
    hrt: SparseHRT3D
    W: Dict[int, Dict[int, Dict[int, float]]]
    config: Sparse3DConfig
    lut: LookupTable
    store: CommitStore
    layer_hllsets: LayerHLLSets
    prompt_perceptron: PromptPerceptron
    response_actuator: ResponseActuator
    max_n: int = 3


def ask(
    prompt: str,
    ctx: QueryContext,
    top_k: int = 10,
    learn: bool = True
) -> Tuple[str, List[DisambiguationResult]]:
    """
    Interactive query with feedback loop.
    
    Full sense-process-act-feedback cycle:
    1. Query → HLLSet → HRT → Commit (SENSE)
    2. Find related concepts (PROCESS)
    3. Disambiguate to tokens (PROCESS)
    4. Generate response (ACT)
    5. Response → HLLSet → HRT → Commit (FEEDBACK!)
    
    The manifold learns from BOTH the question AND its own answer.
    
    Args:
        prompt: User query text
        ctx: QueryContext with current state
        top_k: Number of results to return
        learn: If True, ingest both query AND response
    
    Returns:
        (response_text, disambiguation_results)
    """
    # SENSE: Process query through prompt perceptron
    new_hrt, new_W, commit, result = ctx.prompt_perceptron.process_prompt(
        prompt,
        ctx.hrt,
        ctx.W,
        ctx.store,
        ctx.max_n
    )
    
    if not commit:
        return "No results (empty query)", []
    
    if learn:
        ctx.hrt = new_hrt
        ctx.W = new_W
    
    # PROCESS: Get query indices
    query_indices = set()
    if result:
        for basic in result.input_basics:
            query_indices.add(basic.to_index(ctx.config))
    
    # Find reachable concepts
    AM = Sparse3DMatrix.from_am(ctx.hrt.am, ctx.config)
    layer0 = project_layer(AM, 0)
    reachable = reachable_from(layer0, query_indices, hops=1)
    
    # Score by connectivity
    layer0_dict = layer0.to_dict()
    scores = {}
    for idx in reachable:
        if idx in layer0_dict:
            scores[idx] = sum(layer0_dict[idx].values())
    
    # Get top-k results
    top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    query_results = []
    for idx, score in top:
        ntokens = ctx.lut.index_to_ntokens.get(idx, set())
        if ntokens:
            # Get first ntoken
            _, ntoken = next(iter(ntokens))
            query_results.append((ntoken, score))
        else:
            query_results.append((f"<idx:{idx}>", score))
    
    # DISAMBIGUATE: Full cascading disambiguation
    disamb_results = cascading_disambiguate(
        query_indices=query_indices,
        am=ctx.hrt.am,
        layer_hllsets=ctx.layer_hllsets,
        W=ctx.W,
        lut=ctx.lut
    )
    
    # ACT + FEEDBACK: Generate response and ingest it back
    response_text, final_hrt, final_W = ctx.response_actuator.act(
        commit,
        result,
        query_results=query_results,
        hrt=ctx.hrt,
        W=ctx.W,
        store=ctx.store,
        lut=ctx.lut,
        config=ctx.config,
        ingest_response=learn,
        max_n=ctx.max_n
    )
    
    if learn:
        ctx.hrt = final_hrt
        ctx.W = final_W
    
    return response_text, disamb_results


def create_query_context(
    config: Sparse3DConfig,
    lut: Optional[LookupTable] = None
) -> QueryContext:
    """
    Create a new QueryContext for interactive querying.
    
    Initializes all components needed for ask().
    """
    if lut is None:
        lut = LookupTable(config=config)
        lut.add_ntoken(START)
        lut.add_ntoken(END)
    
    # Empty initial structures
    empty_am = SparseAM3D.from_edges(config, [])
    empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
    empty_hrt = SparseHRT3D(
        am=empty_am,
        lattice=empty_lattice,
        config=config,
        lut=frozenset(),
        step=0
    )
    
    empty_W: Dict[int, Dict[int, Dict[int, float]]] = {n: {} for n in range(config.max_n)}
    
    # Components
    store = CommitStore()
    layer_hllsets = LayerHLLSets.empty(config.p_bits)
    
    prompt_perceptron = PromptPerceptron(config)
    prompt_perceptron.initialize(lut)
    
    response_actuator = ResponseActuator()
    
    return QueryContext(
        hrt=empty_hrt,
        W=empty_W,
        config=config,
        lut=lut,
        store=store,
        layer_hllsets=layer_hllsets,
        prompt_perceptron=prompt_perceptron,
        response_actuator=response_actuator,
        max_n=config.max_n
    )
