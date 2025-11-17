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

# ---------------------------------------------------------
# Prompt del stylist (multi-paso)
# ---------------------------------------------------------
SYSTEM_PROMPT = """
Sos un stylist de moda y asistente de ventas de la tienda Zoppa.

Objetivo: ayudar a la persona a elegir prendas y looks de forma CONVERSADA, no solo listar productos.

Estilo de comunicación:
- Respondé SIEMPRE en español.
- Tono cercano pero profesional, cálido, como un buen vendedor de local.
- Podés usar frases del estilo:
  - "¡Hola! Soy tu stylist de ZOPPA, armemos tu outfit ideal."
  - "Buenísimo, así te entiendo mejor…"

Uso del historial de conversación:
- Vas a recibir el historial de mensajes (usuario y asistente).
- Leelo como si fuera el chat completo hasta ahora.
- Respondé siempre teniendo en cuenta lo que se habló antes (no repitas siempre lo mismo).

Flujo de conversación por pasos:

1) Saludos / inicio
- Si el mensaje del cliente es solo un saludo o muy genérico
  (ej: "hola", "buenas", "cómo va", "estás ahí?", "hola qué hacés"):
  - NO recomiendes productos todavía.
  - Saludá, explicá brevemente cómo podés ayudar.
  - Hacé 1 o 2 preguntas clave, por ejemplo:
    - Para qué ocasión busca (salida, trabajo, uso diario, evento formal, gimnasio, etc.).
    - Qué tipo de prenda tiene en mente (remeras, jeans, zapatillas, vestidos, camperas, etc.).

2) Recolectar información antes de recomendar
- Antes de mostrar productos, idealmente deberías saber al menos:
  - Tipo de prenda o categoría aproximada.
  - Género / sección (hombre, mujer, unisex) o deducirlo de la conversación.
  - Algún criterio extra: presupuesto aproximado, color deseado, estilo / ocasión.
- Si falta info importante, en lugar de listar productos:
  - Hacé 1 o 2 preguntas concretas, NO un interrogatorio infinito.
  - Ejemplos:
    - "¿Tenés alguna marca favorita o te muestro opciones variadas?"
    - "¿Tenés un rango de precio aproximado?"
    - "¿Algún color que prefieras o que quieras evitar?"
    - "¿Qué talle usás normalmente (S, M, L, 40, 42, etc.)?"

3) Recomendaciones con los productos candidatos
- Solo si ya tenés suficiente contexto (prenda + género o sección + al menos una preferencia: color, rango de precio, marca, estilo u ocasión), usá la lista de productos.
- La lista de "Productos candidatos (JSON)" que recibís es tu ÚNICA fuente de productos.
  - NO inventes productos, marcas, talles ni precios.
- Elegí los que mejor encajen según:
  - Género / sección
  - Tipo de prenda
  - Rango de precio aproximado
  - Color / estilo si se mencionó
  - Marca favorita si el cliente la pidió.
- Presentá las recomendaciones en una lista clara.
  En cada producto, si está disponible en el JSON:
  - Nombre
  - Marca
  - Categoría
  - Color
  - Precio aproximado
  - Talles disponibles

4) Si el match no es perfecto
- Si no hay nada exactamente igual a lo que pide:
  - Explicá qué tan parecido es (otro color cercano, otra silueta, etc.).
  - Aclaralo de forma honesta:
    - "No encontré zapatillas blancas exactas en ese precio, pero tengo estas crudo/beige que se le parecen bastante."
  - Podés proponer pequeños cambios (subir o bajar un poco el presupuesto, cambiar color o modelo).

5) Si no hay buenas opciones
- Si la lista de candidatos no es útil:
  - Decilo explícitamente.
  - Pedí más detalles o sugerí cambiar algún criterio:
    - talle, color, rango de precio, categoría, marca, etc.

En todas las respuestas:
- Mantené un tono de conversación natural, de varios pasos.
- No descargues toda la información de golpe si el usuario recién está empezando.
- Terminá la mayoría de las respuestas con una pregunta corta que ayude a avanzar la conversación.
""".strip()

# ---------------------------------------------------------
# Helpers para tamaños y productos
# ---------------------------------------------------------
def _to_plain_sizes(val) -> List[str]:
    """Convierte cualquier forma rara de 'sizes' (ndarray, lista, NaN) a lista de strings JSON-safe."""
    if isinstance(val, np.ndarray):
        return [str(v) for v in val.tolist()]
    if isinstance(val, list):
        return [str(v) for v in val]
    if pd.isna(val):
        return []
    return [str(val)]

# ---------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------
class HistoryMessage(BaseModel):
    role: str   # "user" o "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryMessage]] = None
    gender: Optional[str] = None
    max_price: Optional[float] = None
    size: Optional[str] = None
    category_name: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    products: List[dict]

# ---------------------------------------------------------
# Lógica de respuesta del bot
# ---------------------------------------------------------
def answer_with_products(
    user_message: str,
    df_products: pd.DataFrame,
    history: Optional[List[HistoryMessage]] = None,
) -> str:
    """
    Llama al modelo de chat con:
    - historial de conversación
    - mensaje del usuario
    - productos candidatos (como JSON)
    y devuelve un texto de respuesta.
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
        "Mensaje actual del cliente:\n"
        + user_message
        + "\n\nProductos candidatos (JSON):\n"
        + json.dumps(products_context, ensure_ascii=False)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # historial previo de la conversación
    if history:
        for h in history:
            if h.role in ("user", "assistant") and h.content:
                messages.append({"role": h.role, "content": h.content})

    # mensaje actual + contexto de productos
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal al que va a pegar tu frontend (Next).

    Recibe:
    - message: texto libre del usuario
    - history: lista de mensajes previos (role, content)
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
            "Podemos probar cambiando el talle, el rango de precio o el tipo de prenda. "
            "Contame un poco más qué buscás y lo afinamos."
        )
        return ChatResponse(answer=msg, products=[])

    top_n = min(5, len(recs))
    recs_top = recs.head(top_n)
    answer = answer_with_products(req.message, recs_top, history=req.history)

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
