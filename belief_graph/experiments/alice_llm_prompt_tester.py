import json
from openai import OpenAI
import networkx as nx
from pyvis.network import Network


def extract_core_beliefs(text, model="llama3"):
    """
    Sends narrative text to a local Ollama instance using the OpenAI Python client
    and extracts a structured JSON object containing (Subject, Relation, Object) triplets.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    system_prompt = """
    You are an expert cognitive psychologist analyzing a patient's internal monologue.
    Your task is to extract the subject's Core Beliefs (deep psychological schemas) from the text.

    Extract beliefs ONLY related to these three core ontological categories:
    1. [SELF] (The subject's identity, capabilities, worth)
    2. [WORLD] (How reality, rules, or the environment function)
    3. [OTHERS] (The inherent nature or intentions of other entities)

    Format the output EXACTLY as a JSON object containing a "triplets" key, which holds a list of arrays representing (Subject, Relation, Object).
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

    prompt = f"Analyze the following text and extract the core beliefs in the requested JSON format:\n\nTEXT:\n{text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Kept at 0 for deterministic extraction
        )

        # Parse the JSON string from the response
        raw_content = response.choices[0].message.content
        parsed_json = json.loads(raw_content)

        # Safely extract the list from the "triplets" key
        return parsed_json.get("triplets", [])

    except Exception as e:
        print(f"Error during extraction: {e}")
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
        print(f"[GraphEngine] T1 state initialized with {self.graph.number_of_edges()} beliefs.")


def visualize_graph_interactively(nx_graph, filename="data/alice_t1_mind_map.html"):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)

    for u, v, data in nx_graph.edges(data=True):
        edge_color = "#00cc66" if data['status'] == 'active' else "#ff4d4d"
        edge_width = 3 if data['status'] == 'active' else 1

        net.add_node(u, label=u, color="#4da6ff", shape="box")
        net.add_node(v, label=v, color="#4da6ff", shape="box")
        net.add_edge(u, v, color=edge_color, width=edge_width, title=str(data), label=data['relation'])

    net.write_html(filename)
    print(f"\n[Visualization] Success! Open '{filename}' in your web browser.")

# ========================================================================
# ALICE IN WONDERLAND TESTING (Chapter 1 - Down the Rabbit-Hole)
# ========================================================================

alice_monologue_t1 = """
'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! 
How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' 
(Which was very likely true.) 
Down, down, down. Would the fall never come to an end! 'I wonder how many miles I've fallen by this time?' she said aloud. 
'I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think'
"""


def main():
    print("Sending Alice's monologue to llm...")
    extracted_triplets = extract_core_beliefs(alice_monologue_t1)

    print("\n--- Extracted Core Beliefs (SRO Triplets) ---")
    for triplet in extracted_triplets:
        print(triplet)

    print("\nStep 2: Injecting beliefs into Graph Engine...")
    engine = LongitudinalGraphEngine()
    engine.load_initial_t1_state(entity_id="Alice", triplets=extracted_triplets)

    print("\nStep 3: Rendering interactive visualization...")
    visualize_graph_interactively(engine.graph)


if __name__ == "__main__":
    main()
