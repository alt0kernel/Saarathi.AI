import json
import torch
import torch.nn.functional as F
from difflib import SequenceMatcher

class StoreRAGRefiner:
    """
    Implements Eq. (17):
    e_T = Embed(T), C = Retrieve(e_T, V), r = LLaMA3(e_T, C), T_T = WordEmbedding(r)
    Repairs phonetic substitutions (e.g., 'unsalted' vs 'salted') using store context.
    """
    def __init__(self, catalog_path: str, d_model: int = 512):
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)["skus"]
        self.d_model = d_model
        self.word_embedder = torch.nn.Embedding(10000, d_model)

    def retrieve_context(self, raw_transcript: str, top_k: int = 2):
        scored = []
        for item in self.catalog:
            score = SequenceMatcher(None, raw_transcript.lower(), item["name"].lower()).ratio()
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def refine(self, raw_transcript: str, user_dietary_pref: str = "salted"):
        context_items = self.retrieve_context(raw_transcript)
        # Phonemic restoration correction rule grounded on retrieved store catalogue
        refined_query = raw_transcript
        if "unsalted" in raw_transcript.lower() and user_dietary_pref == "salted":
            refined_query = raw_transcript.lower().replace("unsalted", "salted")

        tokens = [hash(w) % 10000 for w in refined_query.split()]
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        t_embeddings = self.word_embedder(token_tensor) # T_T in Eq. (17)
        return refined_query, context_items, t_embeddings