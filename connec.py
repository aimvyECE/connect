
import logging
import os
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupabaseAPI")

# Charger les variables d'environnement

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Vérification des credentials
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Les variables d'environnement SUPABASE_URL et SUPABASE_KEY doivent être définies.")

# Connexion à Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Chargement du modèle d'embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialisation du serveur FastAPI
app = FastAPI()

def get_embedding(text: str) -> List[float]:
    """Génère l'embedding d'un texte donné."""
    return embedding_model.encode(text).tolist()

@app.post("/search")
def search_in_vector_db(query_text: str, top_k: int = 5):
    """Recherche les documents les plus proches dans la base vectorielle Supabase."""
    try:
        query_embedding = get_embedding(query_text)
        response = supabase.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_count": top_k}
        ).execute()
        
        if response.data:
            return [
                {
                    "id": res["id"],
                    "url": res["url"],
                    "title": res["title"],
                    "summary": res["summary"],
                    "content": res["content"],
                    "metadata": res["metadata"],
                    "similarity": res["similarity"]
                }
                for res in response.data
            ]
        else:
            return {"message": "Aucun résultat trouvé."}
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
