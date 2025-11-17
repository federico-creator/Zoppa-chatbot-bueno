from __future__ import annotations

import os
import json
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# Carga de variables de entorno
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en el entorno")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------
app = FastAPI(title="Zoppa Chatbot API")

origins = [
    "http://localhost:5173",          # frontend local (Vite)
    "http://localhost:3000",          # por si usás Next local más adelante
    "https://zoppa-app.vercel.app",   # dominio de Vercel (sin /)
    "https://zoppashop.com",          # dominio real (sin /)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Cargar catálogo + embeddings
# ---------------------------------------------------------
CATALOG_PATH = os.getenv("CATALOG_PATH", "artifacts/catalog.parquet")
EMB_PATH = os.getenv("EMB_PATH", "artifacts/products_embeddings.parquet")

print("▶ Cargando catálogo y embeddings...")
catalog = pd.read_parquet(CATALOG_PATH)
emb_df = pd.read_parquet(EMB_PATH).set_index("id")

# Alinear embeddings con el orden del catálogo
emb_df = emb_df.loc[catalog["id"]]
embeddings = np.vstack(emb_df["embedding"].values)  # (N, D)

# Normalizar para similitud coseno
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
embeddings_norm = embeddings / norms

# Mapa id -> índice en embeddings_norm
ID_TO_POS = {int(pid): i for i, pid in enumerate(catalog["id"].tolist())}

print(f"✔ Catálogo: {catalog.shape}")
print(f"✔ Embeddings: {embeddings_norm.shape}")


# ---------------------------------------------------------
# Utilidades de filtrado y embeddings
# ---------------------------------------------------------
def filtrar_catalogo(
    gender: Optional[str] = None,
    max_price: Optional[float] = None,
    size: Optional[str] = None,
    category_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filtra el catálogo con campos estructurados:
    - gender: 'hombre', 'mujer', 'unisex'
    - max_price: precio máximo
    - size: talle ('S', 'M', '42', etc.)
    - category_name: parte del nombre de categoría
    """
    df = catalog.copy()

    if gender:
        df = df[df["gender"] == gender]

    if max_price is not None:
        df = df[df["effective_price"].notna()]
        df = df[df["effective_price"] <= max_price]

    if size:
        df = df[df["sizes"].apply(lambda xs: isinstance(xs, list) and size in xs)]

    if category_name:
        df = df[df["category_name"].str.contains(category_name, case=False, na=False)]

    return df


def embed_text(text: str) -> np.ndarray:
    """Embedding de una cadena de texto usando el mismo modelo del catálogo."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec


def get_recommendations(
    user_query: str,
    gender: Optional[str] = None,
    max_price: Optional[float] = None,
    size: Optional[str] = None,
    category_name: Optional[str] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Devuelve los top_k productos según filtros + similitud de embeddings.
    """
    # 1) Filtrar catálogo
    df_candidates = filtrar_catalogo(
        gender=gender,
        max_price=max_price,
        size=size,
        category_name=category_name,
    )

    if df_candidates.empty:
        return df_candidates

    # 2) Embedding de la consulta
    q_vec = embed_text(user_query)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    # 3) Similitud coseno contra candidatos
    cand_ids = df_candidates["id"].tolist()
    positions = [ID_TO_POS[int(pid)] for pid in cand_ids]
    cand_emb = embeddings_norm[positions]  # (Ncand, D)

    sims = cand_emb @ q_vec  # (Ncand,)

    df_candidates = df_candidates.copy()
    df_candidates["similarity"] = sims

    # 4) Ordenar y tomar top_k
    df_top = df_candidates.sort_values("similarity", ascending=False).head(top_k)
    return df_top


SYSTEM_PROMPT = """
Sos un asistente de ventas de ropa y calzado para la tienda Zoppa.

Reglas:
- Respondé SIEMPRE en español, con tono cercano pero profesional.
- Cuando recomiendes productos:
  - Incluí nombre del producto, marca, categoría, precio aproximado, talles disponibles y URL.
  - Presentá los productos en una lista clara (numerada o con viñetas).
- No inventes productos, talles ni precios que no estén en el catálogo.
- Si no hay coincidencias exactas, recomendá lo más parecido posible y explicá brevemente por qué.
- Si no encontrás nada, pedí más detalles (talle, color, rango de precio, estilo, etc.).
""".strip()


def _to_plain_sizes(val) -> List[str]:
    """Convierte cualquier forma rara de 'sizes' (ndarray, lista, NaN) a lista de strings JSON-safe."""
    if isinstance(val, np.ndarray):
        return [str(v) for v in val.tolist()]
    if isinstance(val, list):
        return [str(v) for v in val]
    if pd.isna(val):
        return []
    return [str(val)]


def answer_with_products(user_message: str, df_products: pd.DataFrame) -> str:
    """
    Llama al modelo de chat con:
    - mensaje del usuario
    - productos candidatos (como JSON)
    y devuelve un texto de respuesta listo para el frontend.
    """
    products_context: List[dict] = []
    for _, row in df_products.iterrows():
        sizes_list = _to_plain_sizes(row.get("sizes", []))

        products_context.append(
            {
                "id": int(row["id"]),
                "name": str(row["name"]),
                "brand": str(row.get("brand_name", "")),
                "category": str(row.get("category_name", "")),
                "color": str(row.get("color", "")),
                "price": float(row["effective_price"])
                if not pd.isna(row["effective_price"])
                else None,
                "sizes": sizes_list,
                "url": str(row.get("url", "")),
                "similarity": float(row["similarity"]),
            }
        )

    user_content = (
        "Mensaje del cliente:\n"
        + user_message
        + "\n\nProductos candidatos (JSON):\n"
        + json.dumps(products_context, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------
# Modelos Pydantic + endpoints
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    gender: Optional[str] = None
    max_price: Optional[float] = None
    size: Optional[str] = None
    category_name: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    products: List[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal al que va a pegar tu frontend (Next).

    Recibe:
    - message: texto libre del usuario
    - gender, max_price, size, category_name (opcionales)

    Devuelve:
    - answer: texto del bot
    - products: lista de productos recomendados (para renderizar tarjetas)
    """
    recs = get_recommendations(
        user_query=req.message,
        gender=req.gender,
        max_price=req.max_price,
        size=req.size,
        category_name=req.category_name,
    )

    if recs.empty:
        msg = (
            "No encontré productos que se ajusten exactamente a lo que pedís. "
            "Probá cambiar el talle, el rango de precio o el tipo de prenda, "
            "o contame un poco más qué buscás."
        )
        return ChatResponse(answer=msg, products=[])

    top_n = min(5, len(recs))
    recs_top = recs.head(top_n)
    answer = answer_with_products(req.message, recs_top)

    # Construimos productos con tipos simples para el JSON
    products_payload: List[dict] = []
    for _, row in recs_top.iterrows():
        products_payload.append(
            {
                "id": int(row["id"]),
                "name": str(row["name"]),
                "brand_name": str(row.get("brand_name", "")),
                "category_name": str(row.get("category_name", "")),
                "color": str(row.get("color", "")),
                "effective_price": float(row["effective_price"])
                if not pd.isna(row["effective_price"])
                else None,
                "sizes": _to_plain_sizes(row.get("sizes", [])),
                "url": str(row.get("url", "")),
                "similarity": float(row["similarity"]),
            }
        )

    return ChatResponse(answer=answer, products=products_payload)
