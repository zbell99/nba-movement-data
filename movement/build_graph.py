"""
Build a spatiotemporal graph from NBA tracking data.

Graph Structure:
- Frame nodes: one per unique (game_clock, event_id, quarter)
  Attributes: game_clock, event_id, quarter
  
- Entity state nodes: one per entity per unique game_clock
  Attributes: player_id, team_id, x_loc, y_loc, radius, entity_type (player|ball)
  Node ID: "{entity_id}_{game_clock}"
  
- Spatial edges: entity-to-entity within the same frame
  Connects player-player and player-ball pairs
  Attributes: distance, entity_pair (player-player|player-ball)
  
- Frame-entity edges: link frame to all its entities
  Attributes: edge_type (contains)
  
- Temporal edges: entity-to-entity across consecutive frames within same event
  Connects entity at t to entity at t+1 if same entity and same event_id
  Attributes: time_delta, displacement, temporal_edge (True)
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image, ImageDraw

from court import draw_court, COURT_LENGTH, COURT_WIDTH
from animate_movement_pil import create_court_background_image, draw_circle_with_border, hex_to_rgb


class TrackingGraphBuilder:
    """Build a spatiotemporal graph from NBA tracking data."""
    
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.frame_nodes = set()
        self.entity_nodes = set()
        self.frames_by_entity = {}  # {entity_id: [game_clocks in order]}
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load tracking data from CSV."""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tracking records from {csv_path}")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique game_clocks: {df['game_clock'].nunique()}")
        print(f"Unique event_ids: {df['event_id'].nunique()}")
        return df
    
    def build(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """Build the graph from tracking data."""
        print("\n--- Building Graph ---")
        
        # Step 1: Create frame nodes
        print("\n[1/4] Creating frame nodes...")
        self._create_frame_nodes(df)
        print(f"Created {len(self.frame_nodes)} frame nodes")
        
        # Step 2: Create entity state nodes
        print("\n[2/4] Creating entity state nodes...")
        self._create_entity_nodes(df)
        print(f"Created {len(self.entity_nodes)} entity state nodes")
        
        # Step 3: Create spatial edges (entity-to-entity within frames)
        print("\n[3/4] Creating spatial edges...")
        print("  (This step is computationally intensive for large datasets)")
        spatial_edges = self._create_spatial_edges(df)
        print(f"Created {spatial_edges} spatial edges")
        
        # Step 4: Create temporal edges (entity across consecutive frames in same event)
        print("\n[4/4] Creating temporal edges...")
        temporal_edges = self._create_temporal_edges(df)
        print(f"Created {temporal_edges} temporal edges")
        
        return self.G
    
    def _create_frame_nodes(self, df: pd.DataFrame):
        """Create one node per unique (game_clock, event_id, quarter)."""
        frames = df.groupby(['game_clock', 'event_id', 'quarter']).size()
        for (game_clock, event_id, quarter), _ in frames.items():
            node_id = self._frame_node_id(game_clock, event_id)
            self.G.add_node(
                node_id,
                node_type='frame',
                game_clock=game_clock,
                event_id=int(event_id),
                quarter=int(quarter)
            )
            self.frame_nodes.add(node_id)
    
    def _create_entity_nodes(self, df: pd.DataFrame):
        """Create one node per entity per unique game_clock."""
        # Group by entity and game_clock to get unique positions
        unique_states = df.groupby(['player_id', 'team_id', 'game_clock']).first().reset_index()
        
        for _, row in unique_states.iterrows():
            entity_id = self._entity_id(row['player_id'], row['team_id'])
            node_id = self._entity_node_id(entity_id, row['game_clock'])
            
            # Track frame membership for temporal edges
            if entity_id not in self.frames_by_entity:
                self.frames_by_entity[entity_id] = []
            self.frames_by_entity[entity_id].append(row['game_clock'])
            
            entity_type = 'ball' if row['player_id'] == -1 else 'player'
            self.G.add_node(
                node_id,
                node_type='entity_state',
                entity_id=entity_id,
                player_id=int(row['player_id']),
                team_id=int(row['team_id']),
                x_loc=float(row['x_loc']),
                y_loc=float(row['y_loc']),
                radius=float(row['radius']),
                game_clock=float(row['game_clock']),
                entity_type=entity_type
            )
            self.entity_nodes.add(node_id)
            
            # Add edge from frame to entity
            frame_id = self._frame_node_id(row['game_clock'], row['event_id'])
            self.G.add_edge(frame_id, node_id, edge_type='contains')
    
    def _create_spatial_edges(self, df: pd.DataFrame) -> int:
        """
        Create spatial edges between all entity pairs within each frame.
        Connect player-player and player-ball pairs.
        """
        edge_count = 0
        
        # Group by frame and get entity data
        frames = df.groupby(['game_clock', 'event_id', 'quarter'])
        
        for (game_clock, event_id, quarter), frame_df in frames:
            # Get unique entities in this frame
            entities_list = []
            for idx, row in frame_df.iterrows():
                entities_list.append((
                    self._entity_id(row['player_id'], row['team_id']),
                    float(row['x_loc']),
                    float(row['y_loc']),
                    int(row['player_id'])
                ))
            
            # Create edges for all pairs (optimized iteration)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    entity1_id, x1, y1, player1 = entities_list[i]
                    entity2_id, x2, y2, player2 = entities_list[j]
                    
                    node1 = self._entity_node_id(entity1_id, game_clock)
                    node2 = self._entity_node_id(entity2_id, game_clock)
                    
                    # Calculate distance
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    
                    # Classify edge type
                    type1 = 'ball' if player1 == -1 else 'player'
                    type2 = 'ball' if player2 == -1 else 'player'
                    entity_pair = f"{type1}-{type2}" if type1 < type2 else f"{type2}-{type1}"
                    
                    # Add bidirectional spatial edges
                    self.G.add_edge(
                        node1, node2,
                        edge_type='spatial',
                        distance=distance,
                        entity_pair=entity_pair
                    )
                    self.G.add_edge(
                        node2, node1,
                        edge_type='spatial',
                        distance=distance,
                        entity_pair=entity_pair
                    )
                    edge_count += 2
        
        return edge_count
    
    def _create_temporal_edges(self, df: pd.DataFrame) -> int:
        """
        Create temporal edges between consecutive observations of the same entity
        within the same event_id.
        """
        edge_count = 0
        
        # Group by entity and event, then sort by game_clock
        for (player_id, team_id, event_id), group in df.groupby(['player_id', 'team_id', 'event_id']):
            entity_id = self._entity_id(player_id, team_id)
            
            # Sort by game_clock (descending, since game time counts down)
            group = group.sort_values('game_clock', ascending=False).reset_index(drop=True)
            
            # Connect consecutive frames
            for i in range(len(group) - 1):
                row_t = group.iloc[i]
                row_t1 = group.iloc[i + 1]
                
                game_clock_t = row_t['game_clock']
                game_clock_t1 = row_t1['game_clock']
                
                node_t = self._entity_node_id(entity_id, game_clock_t)
                node_t1 = self._entity_node_id(entity_id, game_clock_t1)
                
                # Calculate displacement
                displacement = np.sqrt(
                    (row_t['x_loc'] - row_t1['x_loc']) ** 2 +
                    (row_t['y_loc'] - row_t1['y_loc']) ** 2
                )
                
                # Time delta (positive means backward in game time)
                time_delta = game_clock_t - game_clock_t1
                
                # Add edge from t+1 to t (following game time progression)
                self.G.add_edge(
                    node_t1, node_t,
                    edge_type='temporal',
                    time_delta=time_delta,
                    displacement=displacement,
                    temporal_edge=True
                )
                edge_count += 1
        
        return edge_count
    
    def _entity_id(self, player_id: int, team_id: int) -> str:
        """Create unique entity identifier."""
        if player_id == -1:
            return "ball_-1"
        return f"player_{int(player_id)}"
    
    def _entity_node_id(self, entity_id: str, game_clock: float) -> str:
        """Create entity state node ID."""
        return f"{entity_id}_{game_clock}"
    
    def _frame_node_id(self, game_clock: float, event_id: int) -> str:
        """Create frame node ID."""
        return f"frame_{int(event_id)}_{game_clock}"
    
    def visualize_frame(self, game_clock: float, event_id: int, output_path: str = None):
        """
        Visualize a frame on the court using existing court visualization helpers.
        
        Args:
            game_clock: game clock time
            event_id: event ID
            output_path: optional path to save figure
        """
        frame_id = self._frame_node_id(game_clock, event_id)
        
        if frame_id not in self.G:
            print(f"Frame {frame_id} not found in graph")
            return
        
        # Create court background with coordinate mapping
        court_img, x_scale, y_scale, x_offset, y_offset = create_court_background_image()
        
        # Get entities in this frame
        entity_nodes = list(self.G.successors(frame_id))
        
        players = []
        ball = None
        team_ids = set()
        
        for entity_node in entity_nodes:
            entity_data = self.G.nodes[entity_node]
            x = entity_data.get('x_loc', 0)
            y = entity_data.get('y_loc', 0)
            entity_type = entity_data.get('entity_type', 'unknown')
            team_id = entity_data.get('team_id', -1)
            
            if entity_type == 'ball':
                ball = (x, y)
            else:
                players.append((x, y, team_id))
                team_ids.add(team_id)
        
        # Assign team colors (hex format for hex_to_rgb)
        team_colors = {}
        team_list = sorted([t for t in team_ids if t != -1])
        if len(team_list) >= 1:
            team_colors[team_list[0]] = '#DC143C'  # Crimson
        if len(team_list) >= 2:
            team_colors[team_list[1]] = '#0064C8'  # Dodger blue
        
        # Create canvas and draw
        canvas = court_img.copy()
        draw = ImageDraw.Draw(canvas)
        
        # Draw players using the shared helper
        for x, y, team_id in players:
            px = x * x_scale + x_offset
            py = y * y_scale + y_offset
            color_hex = team_colors.get(team_id, 'gray')
            rgb_color = hex_to_rgb(color_hex)
            draw_circle_with_border(draw, px, py, radius=6, 
                                   fill_color=rgb_color, border_color=(0, 0, 0), border_width=1)
        
        # Draw ball
        if ball:
            px = ball[0] * x_scale + x_offset
            py = ball[1] * y_scale + y_offset
            rgb_color = hex_to_rgb('#FFA500')  # Orange
            draw_circle_with_border(draw, px, py, radius=4,
                                   fill_color=rgb_color, border_color=(0, 0, 0), border_width=1)
        
        if output_path:
            canvas.save(output_path)
            print(f"Saved visualization to {output_path}")
        
        return canvas
    
    def validate(self) -> Dict:
        """Validate graph structure and return metrics."""
        print("\n--- Validation ---")
        metrics = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'frame_nodes': len(self.frame_nodes),
            'entity_nodes': len(self.entity_nodes),
            'spatial_edges': sum(1 for _, _, data in self.G.edges(data=True) if data.get('edge_type') == 'spatial'),
            'temporal_edges': sum(1 for _, _, data in self.G.edges(data=True) if data.get('edge_type') == 'temporal'),
            'frame_entity_edges': sum(1 for _, _, data in self.G.edges(data=True) if data.get('edge_type') == 'contains'),
        }
        
        for key, val in metrics.items():
            print(f"{key}: {val}")
        
        return metrics
    
    def inspect_sample_frame(self, game_clock: float, event_id: int):
        """Inspect a sample frame in detail."""
        print(f"\n--- Sample Frame: event_id={int(event_id)}, game_clock={game_clock} ---")
        
        frame_id = self._frame_node_id(game_clock, event_id)
        frame_node = self.G.nodes[frame_id]
        
        print(f"Frame node: {frame_id}")
        print(f"  Attributes: {dict(frame_node)}")
        
        # Get all entities in this frame
        successors = list(self.G.successors(frame_id))
        print(f"  Entities in frame: {len(successors)}")
        
        for entity_node_id in successors[:10]:  # Show first 10
            entity_data = self.G.nodes[entity_node_id]
            entity_type = entity_data.get('entity_type', 'unknown')
            x = entity_data.get('x_loc', 'N/A')
            y = entity_data.get('y_loc', 'N/A')
            print(f"    {entity_node_id}: type={entity_type}, pos=({x}, {y})")
        
        # Inspect spatial edges
        spatial_edge_count = 0
        for u, v, data in self.G.edges(data=True):
            if data.get('edge_type') == 'spatial' and u in successors and v in successors:
                spatial_edge_count += 1
        
        print(f"  Spatial edges (entity-entity in frame): {spatial_edge_count}")
        
        # Sample an entity and show its temporal edges
        if successors:
            sample_entity = successors[0]
            temporal_successors = [v for v in self.G.successors(sample_entity) if 
                                   self.G[sample_entity][v].get('edge_type') == 'temporal']
            print(f"  Sample entity {sample_entity}:")
            print(f"    Temporal successors: {temporal_successors}")


def main():
    """Example usage: build and inspect a graph from tracking data."""
    csv_path = Path(__file__).parent.parent / "data" / "csv" / "0021500492.csv"
    
    # Load data
    builder = TrackingGraphBuilder()
    df = builder.load_data(str(csv_path))
    
    # Sample: use first 5 events only for quick testing
    unique_events = df['event_id'].unique()[:5]
    df_sample = df[df['event_id'].isin(unique_events)].copy()
    print(f"\nUsing sample of {len(df_sample)} records from {len(unique_events)} events for testing")
    print(f"(Full dataset has {len(df)} records from {df['event_id'].nunique()} events)")
    
    # Build graph
    G = builder.build(df_sample)
    
    # Validate
    metrics = builder.validate()
    
    # Inspect a sample frame
    sample_game_clock = df_sample['game_clock'].iloc[50]
    sample_event_id = df_sample[df_sample['game_clock'] == sample_game_clock]['event_id'].iloc[0]
    builder.inspect_sample_frame(sample_game_clock, sample_event_id)
    
    # Basic graph inspection
    print("\n--- Graph Structure ---")
    print(f"Graph type: {type(G).__name__}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is multigraph: {G.is_multigraph()}")
    
    # Sample nodes
    print("\n--- Sample Nodes ---")
    for node_id in list(G.nodes())[:5]:
        node_data = G.nodes[node_id]
        print(f"{node_id}: {node_data}")
    
    # Sample edges
    print("\n--- Sample Edges ---")
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"{u} -> {v}: {data}")
    
    # Visualize the frame
    print("\n--- Frame Visualization ---")
    output_dir = Path(__file__).parent.parent / "data" / "animations"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_frame.png"
    builder.visualize_frame(sample_game_clock, sample_event_id, str(output_path))
    
    return G, builder, df_sample


if __name__ == "__main__":
    G, builder, df = main()
