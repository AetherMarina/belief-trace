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
    output_html: str = "outputs/graph.html",
):
    """
    Render the Longitudinal Belief Graph.

    v0.1 semantics:
    - Beliefs are represented as nodes.
    - ACTIVE beliefs are green.
    - DEPRECATED beliefs are red.
    - SHATTERED transitions are represented as explicit event nodes.
    - A SHATTERED transition does NOT imply a direct replacement belief.
    """

    print("Loading data from JSONL artifacts...")

    beliefs = load_jsonl(beliefs_path)
    transitions = load_jsonl(transitions_path)

    if not beliefs:
        print("[Error] No beliefs found to visualize. Run the demo first.")
        return

    g = nx.DiGraph()

    # ================================================================
    # 1. BELIEF NODES
    # ================================================================

    for belief in beliefs:
        belief_id = belief["belief_id"]
        status = belief["status"]

        label = (
            f"[{belief['subject']}]\n"
            f"{belief['relation']}\n"
            f"[{belief['object']}]"
        )

        hover_text = (
            f"<b>ID:</b> {belief_id}<br>"
            f"<b>Entity:</b> {belief['entity_id']}<br>"
            f"<b>Source:</b> {belief['source_id']}<br>"
            f"<b>First seen:</b> Step {belief['first_seen_step']}<br>"
            f"<b>Last seen:</b> Step {belief['last_seen_step']}<br>"
            f"<b>Status:</b> {status.upper()}<br>"
            f"<b>Model:</b> {belief['provenance']['model']}<br>"
            f"<b>Prompt version:</b> "
            f"{belief['provenance']['prompt_version']}"
        )

        if belief.get("evidence_span"):
            hover_text += (
                f"<br><b>Evidence:</b> {belief['evidence_span']}"
            )

        if status == "active":
            color = "#00cc66"
            border_width = 3

        elif status == "deprecated":
            color = "#ff4d4d"
            border_width = 1

        else:
            # CHALLENGED — currently future-facing in v0.1
            color = "#ffb347"
            border_width = 2

        g.add_node(
            belief_id,
            label=label,
            title=hover_text,
            color={
                "background": color,
                "border": "white",
            },
            shape="box",
            borderWidth=border_width,
            font={"color": "white"},
        )

    # ================================================================
    # 2. TRANSITION EVENT NODES
    # ================================================================

    for transition in transitions:
        transition_id = transition["transition_id"]
        affected_belief_id = transition["affected_belief_id"]
        resulting_belief_id = transition.get("resulting_belief_id")

        transition_type = transition["transition_type"]
        step = transition["step"]
        reason = transition["reason"]

        # Separate graph-node ID so it can never collide with belief IDs.
        event_node_id = f"transition::{transition_id}"

        event_label = (
            f"{transition_type.upper()}\n"
            f"T{step}"
        )

        provenance = transition.get("provenance", {})

        hover_text = (
            f"<b>Transition:</b> {transition_id}<br>"
            f"<b>Type:</b> {transition_type.upper()}<br>"
            f"<b>Step:</b> {step}<br>"
            f"<b>Affected belief:</b> {affected_belief_id}<br>"
            f"<b>Reason:</b> {reason}<br>"
            f"<b>Model:</b> {provenance.get('model', 'unknown')}<br>"
            f"<b>Prompt version:</b> "
            f"{provenance.get('prompt_version', 'unknown')}"
        )

        # Transition itself is explicitly visualized as an event.
        g.add_node(
            event_node_id,
            label=event_label,
            title=hover_text,
            color={
                "background": "#8b1e1e",
                "border": "#ff8080",
            },
            shape="diamond",
            borderWidth=2,
            font={"color": "white"},
        )

        # Belief -> transition event
        if affected_belief_id in g.nodes:
            g.add_edge(
                affected_belief_id,
                event_node_id,
                title=(
                    f"{transition_type.upper()} at Step {step}<br>"
                    f"Reason: {reason}"
                ),
                color="#ff4d4d",
                width=2,
                dashes=True,
            )

        # Future-compatible:
        #
        # If a later version contains an explicit resulting_belief_id,
        # visualize exactly that relationship.
        #
        # IMPORTANT:
        # v0.1 normally has resulting_belief_id=None, so no replacement
        # belief is inferred from "same step" co-occurrence.
        if (
            resulting_belief_id
            and resulting_belief_id in g.nodes
        ):
            g.add_edge(
                event_node_id,
                resulting_belief_id,
                label="RESULTS_IN",
                title=(
                    f"Explicit resulting belief: "
                    f"{resulting_belief_id}"
                ),
                color="#aaaaaa",
                width=2,
            )

    # ================================================================
    # 3. RENDER
    # ================================================================

    print(
        f"Rendering {len(g.nodes)} nodes "
        f"and {len(g.edges)} edges..."
    )

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )

    net.force_atlas_2based()

    net.from_nx(g)

    net.write_html(output_html)

    print(
        f"\n[Success] Interactive graph generated: "
        f"'{output_html}'."
    )


if __name__ == "__main__":
    print("==================================================")
    print("      LONGITUDINAL GRAPH VISUALIZER v0.1          ")
    print("==================================================\n")

    build_visualization()