import json
from openai import OpenAI
import networkx as nx
from pyvis.network import Network


# ========================================================================
# 1. LLM MODULE (Extraction & Conflict Resolution)
# ========================================================================
def extract_core_beliefs(text, model="llama3"):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    system_prompt = """
    You are an expert cognitive psychologist analyzing a patient's internal monologue.
    Your task is to extract the subject's Core Beliefs (deep psychological schemas).

    Extract beliefs ONLY related to these three core ontological categories:
    1. [SELF] (The subject's identity, capabilities, worth)
    2. [WORLD] (How reality, rules, or the environment function)
    3. [OTHERS] (The inherent nature or intentions of other entities)

    Format the output EXACTLY as a JSON object containing a "triplets" key, 
    which holds a list of arrays representing (Subject, Relation, Object).
    Do not include any other text, markdown, or explanations. 

    Example input: "If I don't follow the rules exactly, something terrible will happen. I am too small to matter here."
    Example output:
    {
        "triplets": [
            ["World", "REQUIRES", "Strict Obedience"],
            ["World", "PUNISHES", "Mistakes"],
            ["Self", "IS", "Insignificant"]
        ]
    }
    """

    prompt = f"Analyze the following text and extract the core beliefs:\n\nTEXT:\n{text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        parsed_json = json.loads(response.choices[0].message.content)
        return parsed_json.get("triplets", [])
    except Exception as e:
        print(f"\n[LLM Error] Extraction failed! Details: {e}")
        return []


def detect_belief_shifts(current_triplets, new_text, model="llama3"):
    """
    Acts as an LLM Judge. Compares the subject's existing beliefs against
    a new narrative event and identifies which old beliefs are shattered.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    system_prompt = f"""
    You are a Cognitive Arbiter. You evaluate a subject's existing Core Beliefs against their new experiences.

    CURRENT BELIEFS:
    {json.dumps(current_triplets, indent=2)}

    Task: Read the provided NEW TEXT. Identify if the new experience shatters, contradicts, or fundamentally alters any of the CURRENT BELIEFS.

    Return EXACTLY a JSON object with a "deprecated_triplets" key containing ONLY the exact triplets from the CURRENT BELIEFS list that are no longer valid. If none are broken, return an empty list.
    Do not modify the strings of the triplets. They must match the input exactly.

    Example output:
    {{
        "deprecated_triplets": [
            ["Self", "IS", "Brave"]
        ]
    }}
    """

    prompt = f"NEW TEXT:\n{new_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        parsed_json = json.loads(response.choices[0].message.content)
        return parsed_json.get("deprecated_triplets", [])
    except Exception as e:
        print(f"[Arbiter Error] Conflict resolution failed: {e}")
        return []


# ========================================================================
# 2. GRAPH ENGINE MODULE
# ========================================================================
class LongitudinalGraphEngine:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_initial_t1_state(self, entity_id, triplets):
        """Builds the initial belief system state at T1."""
        for subject, relation, obj in triplets:
            self.graph.add_edge(
                subject,
                obj,
                relation=relation,
                status="active",
                temporal_step=1,
                entity=entity_id
            )
        print(f"[GraphEngine] T1 state initialized with {self.graph.number_of_edges()} active beliefs.")

    def transition_to_t2_state(self, deprecated_triplets, new_triplets, processing_time):
        """Updates graph state to T2, deprecating broken edges and adding new context."""
        # 1. Deprecate beliefs that were broken by the new narrative
        for u, _, v in deprecated_triplets:
            if self.graph.has_edge(u, v):
                self.graph[u][v]["status"] = "deprecated"
                self.graph[u][v]["deprecated_at_step"] = 2
                self.graph[u][v]["processing_overhead_minutes"] = processing_time
                print(f"[GraphEngine] Deprecated old belief: {u} -> {v}")
            else:
                print(f"[GraphEngine] Warning: Edge {u} -> {v} not found to deprecate.")

        # 2. Add new contextual constraints emerged at T2
        for subject, relation, obj in new_triplets:
            self.graph.add_edge(
                subject,
                obj,
                relation=relation,
                status="active",
                temporal_step=2
            )
            print(f"[GraphEngine] Added new T2 belief: {subject} --({relation})--> {obj}")


def visualize_graph_interactively(nx_graph, filename="data/alice_longitudinal_mind_map.html"):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)

    for u, v, data in nx_graph.edges(data=True):
        if data['status'] == 'deprecated':
            edge_color = "#ff4d4d"  # Red for old/deprecated beliefs
            edge_width = 1
            edge_label = f"(BROKEN)\n{data['relation']}"
        else:
            edge_color = "#00cc66"  # Green for new/active beliefs
            edge_width = 3
            edge_label = data['relation']

        net.add_node(u, label=u, color="#4da6ff", shape="box")
        net.add_node(v, label=v, color="#4da6ff", shape="box")
        net.add_edge(u, v, color=edge_color, width=edge_width, title=str(data), label=edge_label)

    net.write_html(filename)
    print(f"\n[Visualization] Success! Open '{filename}' in your web browser.")


# ========================================================================
# 3. PIPELINE EXECUTION (Longitudinal Fusion)
# ========================================================================
alice_monologue_t1 = """
'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! 
How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' 
Down, down, down. Would the fall never come to an end! 'I must be getting somewhere near the centre of the earth.'
"""

alice_monologue_t2 = """
'I wonder if I've been changed in the night? Let me think: was I the same when I got up this morning? 
I almost think I can remember feeling a little different. But if I'm not the same, the next question is, 
Who in the world am I? Ah, that's the great puzzle!'
"""


def main():
    # --- STEP 1: INITIAL STATE (T1) ---
    print("\n--- PHASE 1: Initial State (T1) ---")
    t1_triplets = extract_core_beliefs(alice_monologue_t1)

    engine = LongitudinalGraphEngine()
    engine.load_initial_t1_state(entity_id="Alice", triplets=t1_triplets)

    # --- STEP 2: COGNITIVE SHIFT (T2) ---
    print("\n--- PHASE 2: Cognitive Shift (T2) ---")
    print("Extracting new beliefs from Chapter 2...")
    t2_new_triplets = extract_core_beliefs(alice_monologue_t2)

    print("Arbiter is evaluating conflicts between old beliefs and new context...")
    deprecated_triplets = detect_belief_shifts(current_triplets=t1_triplets, new_text=alice_monologue_t2)

    engine.transition_to_t2_state(
        deprecated_triplets=deprecated_triplets,
        new_triplets=t2_new_triplets,
        processing_time=15.0
    )

    # --- STEP 3: RENDER ---
    print("\n--- PHASE 3: Rendering ---")
    visualize_graph_interactively(engine.graph)


if __name__ == "__main__":
    main()
