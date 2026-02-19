"""
Graph Visualization Tool for Lattice Structures

Converts HLLSet lattices into property graphs and provides visualization capabilities.
Tests consistency of lattice representations across different structures.

Supports:
- Adjacency Matrix (AM)
- W Lattice (row/column HLLSets)
- Hash Relational Tensor (HRT)
- ManifoldOS storage structures
- Custom graphs
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib


@dataclass
class NodeMetadata:
    """Metadata for a graph node."""
    node_id: str
    node_type: str  # 'hllset', 'token', 'table', 'column', etc.
    label: str
    sha1: Optional[str] = None  # SHA1 hash (content-addressed artifact)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'id': self.node_id,
            'type': self.node_type,
            'label': self.label,
            **self.properties
        }
        if self.sha1:
            result['sha1'] = self.sha1
        return result


@dataclass
class EdgeMetadata:
    """Metadata for a lattice edge (partial order relation)."""
    source: str  # Source node ID
    target: str  # Target node ID
    source_sha1: Optional[str] = None  # SHA1 hash of source node
    target_sha1: Optional[str] = None  # SHA1 hash of target node
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def sha1(self) -> str:
        """Content-addressed hash of edge (source + target + properties)."""
        edge_repr = json.dumps({
            'source': self.source,
            'target': self.target,
            'source_sha1': self.source_sha1,
            'target_sha1': self.target_sha1,
            'weight': self.weight,
            'properties': sorted(self.properties.items())
        }, sort_keys=True)
        return hashlib.sha1(edge_repr.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'sha1': self.sha1,
            **self.properties
        }
        if self.source_sha1:
            result['source_sha1'] = self.source_sha1
        if self.target_sha1:
            result['target_sha1'] = self.target_sha1
        return result


class LatticeGraphBuilder:
    """
    Builds property graphs from various lattice structures.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, NodeMetadata] = {}
        self.edges: List[EdgeMetadata] = []
    
    def clear(self):
        """Clear current graph."""
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
    
    def add_node(self, metadata: NodeMetadata):
        """Add node with metadata."""
        self.nodes[metadata.node_id] = metadata
        self.graph.add_node(
            metadata.node_id,
            **metadata.to_dict()
        )
    
    def add_edge(self, metadata: EdgeMetadata):
        """Add edge with metadata."""
        self.edges.append(metadata)
        self.graph.add_edge(
            metadata.source,
            metadata.target,
            **metadata.to_dict()
        )
    
    def from_adjacency_matrix(self, am: 'AdjacencyMatrix', name: str = "AM"):
        """
        Build graph from Adjacency Matrix.
        
        Nodes: tokens
        Edges: adjacency relationships
        """
        from core.hrt import AdjacencyMatrix
        
        self.clear()
        
        # Add token nodes
        for i, token in enumerate(am.tokens):
            node_id = f"token_{i}_{token}"
            self.add_node(NodeMetadata(
                node_id=node_id,
                node_type='token',
                label=token,
                properties={
                    'index': i,
                    'token': token
                }
            ))
        
        # Add adjacency edges
        n = len(am.tokens)
        for i in range(n):
            for j in range(n):
                if am.matrix[i, j] > 0:
                    source = f"token_{i}_{am.tokens[i]}"
                    target = f"token_{j}_{am.tokens[j]}"
                    self.add_edge(EdgeMetadata(
                        source=source,
                        target=target,
                        weight=float(am.matrix[i, j]),
                        properties={
                            'relation': 'adjacent',
                            'value': int(am.matrix[i, j])
                        }
                    ))
        
        return self
    
    def from_w_lattice(self, w_lattice: 'WLattice', name: str = "W"):
        """
        Build graph from W Lattice.
        
        Nodes: HLLSets (rows, columns, cells)
        Edges: containment relationships
        """
        self.clear()
        
        # Add row nodes
        for i, row_hllset in enumerate(w_lattice.rows):
            node_id = f"row_{i}"
            self.add_node(NodeMetadata(
                node_id=node_id,
                node_type='row_hllset',
                label=f"Row {i}",
                properties={
                    'index': i,
                    'cardinality': row_hllset.cardinality(),
                    'sha1': row_hllset.short_name
                }
            ))
        
        # Add column nodes
        for j, col_hllset in enumerate(w_lattice.columns):
            node_id = f"col_{j}"
            self.add_node(NodeMetadata(
                node_id=node_id,
                node_type='col_hllset',
                label=f"Col {j}",
                properties={
                    'index': j,
                    'cardinality': col_hllset.cardinality(),
                    'sha1': col_hllset.short_name
                }
            ))
        
        # Add cell nodes and edges
        for i in range(len(w_lattice.rows)):
            for j in range(len(w_lattice.columns)):
                cell_hllset = w_lattice.rows[i].intersect(w_lattice.columns[j])
                if cell_hllset.cardinality() > 0:
                    cell_id = f"cell_{i}_{j}"
                    self.add_node(NodeMetadata(
                        node_id=cell_id,
                        node_type='cell_hllset',
                        label=f"Cell ({i},{j})",
                        properties={
                            'row': i,
                            'col': j,
                            'cardinality': cell_hllset.cardinality(),
                            'sha1': cell_hllset.short_name
                        }
                    ))
                    
                    # Edge: row contains cell (lattice: row ⊇ cell)
                    self.add_edge(EdgeMetadata(
                        source=f"row_{i}",
                        target=cell_id,
                        weight=cell_hllset.cardinality(),
                        properties={'relation': 'row_contains'}
                    ))
                    
                    # Edge: column contains cell (lattice: col ⊇ cell)
                    self.add_edge(EdgeMetadata(
                        source=f"col_{j}",
                        target=cell_id,
                        weight=cell_hllset.cardinality(),
                        properties={'relation': 'col_contains'}
                    ))
        
        return self
    
    def from_database_hierarchy(self, ingestion_result: Dict[str, Any], manifold: 'ManifoldOS'):
        """
        Build graph from database ingestion result.
        
        Nodes: database, tables, columns
        Edges: containment, entanglement
        """
        from core.hllset import HLLSet
        
        self.clear()
        
        # Add database node
        db_info = ingestion_result['database']
        db_id = 'database'
        self.add_node(NodeMetadata(
            node_id=db_id,
            node_type='database',
            label='Database',
            sha1=db_info['sha1'],
            properties={
                'cardinality': db_info['cardinality'],
                'artifact_id': db_info['data_id']
            }
        ))
        
        # Add table and column nodes
        for table_name, table_data in ingestion_result['tables'].items():
            # Table node
            table_id = f"table_{table_name}"
            
            # Load table HLLSet to get cardinality
            table_bytes = manifold.retrieve_artifact(table_data['data_id'])
            table_hllset = HLLSet.from_roaring(table_bytes)
            
            self.add_node(NodeMetadata(
                node_id=table_id,
                node_type='table',
                label=table_name,
                sha1=table_data['sha1'],
                properties={
                    'name': table_name,
                    'cardinality': table_hllset.cardinality(),
                    'artifact_id': table_data['data_id'],
                    'column_count': len(table_data['columns'])
                }
            ))
            
            # Edge: database contains table (lattice: db ⊇ table)
            self.add_edge(EdgeMetadata(
                source=db_id,
                target=table_id,
                source_sha1=ingestion_result['database']['sha1'],
                target_sha1=table_data['sha1'],
                weight=table_hllset.cardinality(),
                properties={'relation': 'contains'}
            ))
            
            # Add column nodes
            for col_name, col_data in table_data['columns'].items():
                col_id = f"column_{table_name}_{col_name}"
                
                # Load column HLLSet
                col_bytes = manifold.retrieve_artifact(col_data['data_id'])
                col_hllset = HLLSet.from_roaring(col_bytes)
                
                # Load column metadata
                meta_bytes = manifold.retrieve_artifact(col_data['metadata_id'])
                metadata = json.loads(meta_bytes.decode())
                
                self.add_node(NodeMetadata(
                    node_id=col_id,
                    node_type='column',
                    label=f"{table_name}.{col_name}",
                    sha1=col_data['sha1'],
                    properties={
                        'table': table_name,
                        'column': col_name,
                        'data_type': metadata['data_type'],
                        'cardinality': col_hllset.cardinality(),
                        'distinct_count': metadata.get('distinct_count', 0),
                        'artifact_id': col_data['data_id']
                    }
                ))
                
                # Edge: table contains column (lattice: table ⊇ column)
                self.add_edge(EdgeMetadata(
                    source=table_id,
                    target=col_id,
                    source_sha1=table_data['sha1'],
                    target_sha1=col_data['sha1'],
                    weight=col_hllset.cardinality(),
                    properties={'relation': 'contains'}
                ))
        
        # Add entanglement edges
        for table_name, table_data in ingestion_result['tables'].items():
            for col_name, col_data in table_data['columns'].items():
                col_id = f"column_{table_name}_{col_name}"
                
                # Check for entanglements in ManifoldOS
                entanglements = manifold.get_entanglements(col_data['data_id'])
                
                for entangled_id, strength in entanglements.items():
                    # Find what this artifact represents
                    if entangled_id == col_data['metadata_id']:
                        # Metadata entanglement - add edge to metadata node if we want
                        # For now, just note it in properties
                        pass
        
        return self
    
    def detect_potential_foreign_keys(self, threshold: float = 0.7):
        """
        Analyze column nodes and add FK edges based on similarity.
        
        Only works after from_database_hierarchy() has been called.
        """
        column_nodes = [
            (nid, meta) for nid, meta in self.nodes.items() 
            if meta.node_type == 'column'
        ]
        
        # Compare all column pairs
        fk_candidates = []
        
        for i, (id1, meta1) in enumerate(column_nodes):
            for id2, meta2 in column_nodes[i+1:]:
                # Skip if same table
                if meta1.properties['table'] == meta2.properties['table']:
                    continue
                
                # Check for high similarity (would need to load HLLSets)
                # For now, use heuristics: name matching, cardinality similarity
                name1 = meta1.properties['column'].lower()
                name2 = meta2.properties['column'].lower()
                
                # Simple heuristic: if one ends with _id and matches other's table name
                table1 = meta1.properties['table'].lower()
                table2 = meta2.properties['table'].lower()
                
                if ('_id' in name1 and table2 in name1) or \
                   ('_id' in name2 and table1 in name2) or \
                   (name1 == 'id' and name2.endswith('_id')) or \
                   (name2 == 'id' and name1.endswith('_id')):
                    
                    fk_candidates.append((id1, id2, 0.9))  # High confidence
                
                # Or if names are very similar
                elif name1 == name2 and 'id' in name1:
                    fk_candidates.append((id1, id2, 0.7))  # Medium confidence
        
        # Add FK edges (discovered relationships, not lattice structure)
        for source, target, confidence in fk_candidates:
            source_meta = self.nodes.get(source)
            target_meta = self.nodes.get(target)
            
            self.add_edge(EdgeMetadata(
                source=source,
                target=target,
                source_sha1=source_meta.sha1 if source_meta else None,
                target_sha1=target_meta.sha1 if target_meta else None,
                weight=confidence,
                properties={
                    'relation': 'potential_fk',
                    'confidence': confidence
                }
            ))
        
        return len(fk_candidates)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {},
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }
        
        # Count by type
        for node_id, metadata in self.nodes.items():
            node_type = metadata.node_type
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        for edge in self.edges:
            edge_relation = edge.properties.get('relation', 'unknown')
            stats['edge_types'][edge_relation] = stats['edge_types'].get(edge_relation, 0) + 1
        
        # Degree statistics
        if stats['node_count'] > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats
    
    def export_graphml(self, filepath: Path):
        """Export to GraphML format (property graph)."""
        nx.write_graphml(self.graph, str(filepath))
    
    def export_dot(self, filepath: Path):
        """Export to DOT format (Graphviz)."""
        nx.drawing.nx_pydot.write_dot(self.graph, str(filepath))
    
    def export_json(self, filepath: Path):
        """Export to JSON format."""
        data = {
            'nodes': [meta.to_dict() for meta in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class LatticeVisualizer:
    """
    Visualization utilities for lattice graphs.
    """
    
    def __init__(self, builder: LatticeGraphBuilder):
        self.builder = builder
    
    def plot(self, 
             figsize: Tuple[int, int] = (12, 8),
             layout: str = 'spring',
             node_color_by: str = 'type',
             show_labels: bool = True,
             save_path: Optional[Path] = None):
        """
        Plot the graph.
        
        Args:
            figsize: Figure size
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai')
            node_color_by: Color nodes by property ('type', 'cardinality', etc.)
            show_labels: Show node labels
            save_path: Save to file instead of showing
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.builder.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.builder.graph)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(self.builder.graph, prog='dot')
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.builder.graph)
        else:
            pos = nx.spring_layout(self.builder.graph)
        
        # Color nodes by type
        node_colors = []
        color_map = {
            'database': '#FF6B6B',
            'table': '#4ECDC4',
            'column': '#95E1D3',
            'token': '#FFA07A',
            'row_hllset': '#87CEEB',
            'col_hllset': '#FFB6C1',
            'cell_hllset': '#E6E6FA',
            'hllset': '#DDA0DD'
        }
        
        for node_id in self.builder.graph.nodes():
            if node_id in self.builder.nodes:
                node_type = self.builder.nodes[node_id].node_type
                node_colors.append(color_map.get(node_type, '#CCCCCC'))
            else:
                node_colors.append('#CCCCCC')
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.builder.graph,
            pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges with varying widths based on weight
        edges = self.builder.graph.edges()
        weights = [self.builder.graph[u][v].get('weight', 1.0) for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [2 * (w / max_weight) for w in weights]
        
        # Color edges by type
        edge_colors = []
        edge_color_map = {
            'contains': '#999999',
            'adjacent': '#666666',
            'entangled': '#FF00FF',
            'potential_fk': '#FF0000',
            'morphism': '#0000FF'
        }
        
        for u, v in edges:
            edge_type = self.builder.graph[u][v].get('type', 'unknown')
            edge_colors.append(edge_color_map.get(edge_type, '#CCCCCC'))
        
        nx.draw_networkx_edges(
            self.builder.graph,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.5,
            arrows=True,
            arrowsize=10,
            ax=ax
        )
        
        # Draw labels
        if show_labels:
            labels = {
                node_id: self.builder.nodes[node_id].label
                for node_id in self.builder.graph.nodes()
                if node_id in self.builder.nodes
            }
            nx.draw_networkx_labels(
                self.builder.graph,
                pos,
                labels,
                font_size=8,
                ax=ax
            )
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=node_type)
            for node_type, color in color_map.items()
            if any(meta.node_type == node_type for meta in self.builder.nodes.values())
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        ax.set_title(f"Lattice Graph ({self.builder.graph.number_of_nodes()} nodes, "
                    f"{self.builder.graph.number_of_edges()} edges)")
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
    
    def plot_subgraph(self, 
                     node_ids: List[str],
                     depth: int = 1,
                     **kwargs):
        """
        Plot subgraph around specified nodes.
        
        Args:
            node_ids: Center nodes
            depth: How many hops to include
            **kwargs: Passed to plot()
        """
        # Get subgraph
        nodes_to_include = set(node_ids)
        
        for _ in range(depth):
            new_nodes = set()
            for node in nodes_to_include:
                # Add neighbors
                if node in self.builder.graph:
                    new_nodes.update(self.builder.graph.predecessors(node))
                    new_nodes.update(self.builder.graph.successors(node))
            nodes_to_include.update(new_nodes)
        
        subgraph = self.builder.graph.subgraph(nodes_to_include)
        
        # Create temporary builder with subgraph
        temp_builder = LatticeGraphBuilder()
        temp_builder.graph = subgraph
        temp_builder.nodes = {
            nid: meta for nid, meta in self.builder.nodes.items()
            if nid in nodes_to_include
        }
        temp_builder.edges = [
            edge for edge in self.builder.edges
            if edge.source in nodes_to_include and edge.target in nodes_to_include
        ]
        
        # Plot
        temp_viz = LatticeVisualizer(temp_builder)
        temp_viz.plot(**kwargs)
    
    def print_statistics(self):
        """Print graph statistics."""
        stats = self.builder.get_statistics()
        
        print("=" * 60)
        print("LATTICE GRAPH STATISTICS")
        print("=" * 60)
        print(f"Nodes: {stats['node_count']}")
        print(f"Edges: {stats['edge_count']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Weakly connected: {stats['is_connected']}")
        
        if 'avg_degree' in stats:
            print(f"\nDegree statistics:")
            print(f"  Average: {stats['avg_degree']:.2f}")
            print(f"  Min: {stats['min_degree']}")
            print(f"  Max: {stats['max_degree']}")
        
        print(f"\nNode types:")
        for node_type, count in sorted(stats['node_types'].items()):
            print(f"  {node_type:20s}: {count:4d}")
        
        print(f"\nEdge types:")
        for edge_type, count in sorted(stats['edge_types'].items()):
            print(f"  {edge_type:20s}: {count:4d}")
        
        print("=" * 60)


def test_consistency(builder: LatticeGraphBuilder) -> Dict[str, Any]:
    """
    Test consistency of lattice representation.
    
    Checks:
    - All edges reference existing nodes
    - No self-loops (unless expected)
    - Containment hierarchy is acyclic
    - Node IDs are unique
    - Property types are consistent
    
    Returns:
        Dictionary with test results
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    # Check node ID uniqueness
    node_ids = list(builder.nodes.keys())
    if len(node_ids) != len(set(node_ids)):
        results['errors'].append("Duplicate node IDs found")
        results['passed'] = False
    
    # Check edges reference existing nodes
    for edge in builder.edges:
        if edge.source not in builder.nodes:
            results['errors'].append(f"Edge source '{edge.source}' not in nodes")
            results['passed'] = False
        if edge.target not in builder.nodes:
            results['errors'].append(f"Edge target '{edge.target}' not in nodes")
            results['passed'] = False
    
    # Check for self-loops
    self_loops = [e for e in builder.edges if e.source == e.target]
    if self_loops:
        results['warnings'].append(f"Found {len(self_loops)} self-loops")
    
    # Check containment hierarchy is acyclic
    containment_edges = [
        (e.source, e.target) 
        for e in builder.edges 
        if e.properties.get('relation') == 'contains'
    ]
    if containment_edges:
        containment_graph = nx.DiGraph(containment_edges)
        if not nx.is_directed_acyclic_graph(containment_graph):
            results['errors'].append("Containment hierarchy contains cycles")
            results['passed'] = False
    
    # Check property consistency
    node_type_properties = {}
    for node_id, metadata in builder.nodes.items():
        node_type = metadata.node_type
        if node_type not in node_type_properties:
            node_type_properties[node_type] = set(metadata.properties.keys())
        else:
            current_props = set(metadata.properties.keys())
            expected_props = node_type_properties[node_type]
            if current_props != expected_props:
                results['warnings'].append(
                    f"Node '{node_id}' ({node_type}) has inconsistent properties: "
                    f"expected {expected_props}, got {current_props}"
                )
    
    return results
