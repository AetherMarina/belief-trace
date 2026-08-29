import json
import networkx as nx
from pyvis.network import Network
import os


def load_jsonl(filepath: str) -> list:
    """Read JSON Lines file into a list of dictionaries."""
    data = []

    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def build_visualization(
        beliefs_path: str = "outputs/beliefs.jsonl",
        transitions_path: str = "outputs/transitions.jsonl",
        core_beliefs_path: str = "outputs/core_beliefs.jsonl",
        surface_to_core_mappings_path: str = "outputs/surface_to_core_mappings.jsonl",
        output_html: str = "outputs/graph.html",
):
    """
    Render the v0.2 Longitudinal Belief Graph.

    v0.2 semantics:
    - Surface Beliefs (Triplets) are boxes.
    - Core Beliefs (Taxonomy) are large ellipses.
    - ACTIVE surface beliefs are green; DEPRECATED are red.
    - SHATTERED/REFRAMED transitions are explicit event nodes.
    - Surface-to-Core mappings are directed edges holding the graph together.
    """

    print("Loading data from JSONL artifacts...")

    beliefs = load_jsonl(beliefs_path)
    transitions = load_jsonl(transitions_path)
    core_beliefs = load_jsonl(core_beliefs_path)
    mappings = load_jsonl(surface_to_core_mappings_path)

    if not beliefs and not core_beliefs:
        print("[Error] No beliefs found to visualize. Run the engine first.")
        return

    g = nx.DiGraph()

    # ================================================================
    # 1. SURFACE BELIEF NODES
    # ================================================================

    for belief in beliefs:
        belief_id = belief["belief_id"]
        status = belief.get("status", "active")

        label = (
            f"[{belief['subject']}]\n"
            f"{belief['relation']}\n"
            f"[{belief['object']}]"
        )

        hover_text = (
            f"SURFACE BELIEF\n"
            f"ID: {belief_id}\n"
            f"First seen: Step {belief['first_seen_step']}\n"
            f"Last seen: Step {belief['last_seen_step']}\n"
            f"Status: {status.upper()}"
        )

        if status == "active":
            color = "#00cc66"  # Green[cite: 5]
            border_width = 3
        elif status == "deprecated":
            color = "#ff4d4d"  # Red[cite: 5]
            border_width = 1
        else:
            color = "#ffb347"  # Orange[cite: 5]
            border_width = 2

        g.add_node(
            belief_id,
            label=label,
            title=hover_text,
            color={"background": color, "border": "white"},
            shape="box",  # Surface beliefs remain boxes[cite: 5]
            borderWidth=border_width,
            font={"color": "white"},
        )

    # ================================================================
    # 2. CORE BELIEF NODES
    # ================================================================

    for cb in core_beliefs:
        cb_id = cb["core_belief_id"]
        domain = cb.get("domain", "Unknown")
        label_val = cb.get("label", "Unknown")

        node_label = f"CORE: {domain}\n{label_val}"

        hover_text = (
            f"CORE SCHEMA\n"
            f"Entity: {cb.get('entity_id')}\n"
            f"Domain: {domain}\n"
            f"Label: {label_val}\n"
            f"First seen: Step {cb.get('first_seen_step')}\n"
            f"Last seen: Step {cb.get('last_seen_step')}"
        )

        g.add_node(
            cb_id,
            label=node_label,
            title=hover_text,
            color={"background": "#9b59b6", "border": "#8e44ad"},  # Deep purple
            shape="ellipse",  # Visually distinct from surface boxes
            borderWidth=3,
            size=35,  # Slightly larger to act as visual anchors
            font={"color": "white", "size": 16, "bold": True},
        )

    # ================================================================
    # 3. SURFACE -> CORE MAPPING EDGES
    # ================================================================

    for mapping in mappings:
        surface_id = mapping.get("surface_belief_id")
        core_id = mapping.get("core_belief_id")

        if surface_id in g.nodes and core_id in g.nodes:
            g.add_edge(
                surface_id,
                core_id,
                label="MAPS_TO",
                title=f"Confidence: {mapping.get('confidence_score', 1.0)}",
                color="#9b59b6",
                width=2,
                dashes=True  # Dashed line for abstraction mapping
            )

    # ================================================================
    # 4. TRANSITION EVENT NODES
    # ================================================================

    for transition in transitions:
        transition_id = transition["transition_id"]
        affected_belief_id = transition["affected_belief_id"]
        resulting_belief_id = transition.get("resulting_belief_id")

        transition_type = transition["transition_type"]
        step = transition["step"]
        reason = transition.get("reason", "")

        event_node_id = f"transition::{transition_id}"

        event_label = f"{transition_type.upper()}\nT{step}"

        hover_text = (
            f"Type: {transition_type.upper()}\n"
            f"Reason: {reason}"
        )

        # Transition explicitly visualized as an event
        g.add_node(
            event_node_id,
            label=event_label,
            title=hover_text,
            color={"background": "#8b1e1e", "border": "#ff8080"},
            shape="diamond",
            borderWidth=2,
            font={"color": "white"},
        )

        # Edge: Affected Belief -> Transition Event
        if affected_belief_id in g.nodes:
            g.add_edge(
                affected_belief_id,
                event_node_id,
                color="#ff4d4d",
                width=2,
                dashes=True,
            )

        # Edge: Transition Event -> Resulting Belief (Used for REFRAMED)
        if resulting_belief_id and resulting_belief_id in g.nodes:
            g.add_edge(
                event_node_id,
                resulting_belief_id,
                label="RESULTS_IN",
                color="#aaaaaa",
                width=2,
            )

    # ================================================================
    # 5. RENDER
    # ================================================================

    print(f"Rendering {len(g.nodes)} nodes and {len(g.edges)} edges...")

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )

    net.force_atlas_2based()

    # Loads the NetworkX graph into PyVis
    net.from_nx(g)
    net.write_html(output_html)
    print(f"\n[Success] Interactive graph generated: '{output_html}'.")


if __name__ == "__main__":
    print("==================================================")
    print("      LONGITUDINAL GRAPH VISUALIZER v0.2          ")
    print("==================================================\n")

    build_visualization()
