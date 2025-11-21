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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------
app = FastAPI(title="Zoppa Chatbot API")

origins = [
    "http://localhost:5173",          
    "http://localhost:3000",          
    "https://zoppa-app.vercel.app",   
    "https://zoppashop.com",   
    "https://www.zoppashop.com"       
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
# Marcas válidas del catálogo Zoppa
# ---------------------------------------------------------
MARCAS_VALIDAS = {
    "ay not dead", "aynotdead",
    "allsaints",
    "alo yoga", "aloyoga",
    "bensimon",
    "carhartt wip", "carharttwip", "carhartt",
    "cloetas",
    "dickies",
    "eme studios", "emestudios",
    "gymshark",
    "herencia",
    "jazmín chebar", "jazminchebar", "jazmin chebar",
    "kosiuko",
    "label99", "label 99",
    "maria cher", "mariacher", "maría cher",
    "mishka",
    "napapijri",
    "nude project", "nudeproject",
    "obey",
    "prüne", "prune",
    "scuffers",
    "stussy",
    "tucci"
}

def normalizar_marca(marca: str) -> str:
    """Normaliza el nombre de una marca para comparación."""
    if pd.isna(marca):
        return ""
    return str(marca).lower().strip()

def es_marca_valida(marca: str) -> bool:
    """Verifica si una marca está en el catálogo de Zoppa."""
    marca_norm = normalizar_marca(marca)
    return marca_norm in MARCAS_VALIDAS

# ---------------------------------------------------------
# Prompt del stylist - ACTUALIZADO CON RESTRICCIÓN DE MARCAS
# ---------------------------------------------------------
SYSTEM_PROMPT = """
Sos un stylist de moda y asistente de ventas experto de la tienda Zoppa.

REGLA DE ORO: NO recomiendes productos hasta tener información SUFICIENTE y ESPECÍFICA.

═══════════════════════════════════════════════════════════════
🏷️ MARCAS DISPONIBLES EN ZOPPA (ACTUALIZADO):
═══════════════════════════════════════════════════════════════

ÚNICAMENTE podés recomendar productos de estas marcas:

NACIONALES:
- AY NOT DEAD
- Bensimon
- Cloetas
- Herencia
- Jazmín Chebar
- Kosiuko
- Maria Cher
- Mishka
- Prüne
- Tucci

INTERNACIONALES:
- AllSaints
- Alo Yoga
- Carhartt WIP
- Dickies
- Eme Studios
- Gymshark
- Label99
- Napapijri
- Nude Project
- Obey
- Scuffers
- Stussy

⚠️ IMPORTANTE SOBRE MARCAS:
- Si un cliente menciona una marca que NO está en esta lista, informale amablemente que no la tenemos
- Sugerí marcas alternativas similares de nuestro catálogo
- NO inventes ni menciones marcas que no están listadas arriba
- Ejemplos de marcas que NO tenemos: Nike, Adidas, Puma, Zara, H&M, Lacoste, Tommy, etc.

Ejemplo de respuesta correcta cuando piden marca no disponible:
"Actualmente no trabajamos con [marca mencionada], pero tengo opciones muy similares en marcas como [sugerir 2-3 alternativas de la lista]. ¿Te gustaría ver opciones de estas marcas?"

═══════════════════════════════════════════════════════════════
CRITERIOS MÍNIMOS OBLIGATORIOS PARA RECOMENDAR PRODUCTOS:
═══════════════════════════════════════════════════════════════

Antes de mostrar productos, DEBES tener AL MENOS 3 de estos 4 criterios:

1. TIPO DE PRENDA ESPECÍFICO (obligatorio siempre)
   ✓ Válido: "remera", "jean", "zapatillas", "campera", "buzo", "vestido"
   ✗ Inválido: "ropa", "algo", "prenda", "outfit"

2. OCASIÓN O ESTILO
   ✓ Válido: "para salir", "casual", "deportivo", "formal", "trabajo", "gimnasio"
   ✗ Inválido: no mencionado

3. COLOR O PREFERENCIA VISUAL
   ✓ Válido: "negro", "blanco", "colores claros", "oscuro", "neutro"
   ✗ Inválido: "cualquier color", no mencionado

4. PRESUPUESTO O MARCA (SOLO MARCAS VÁLIDAS)
   ✓ Válido: "hasta $50000", "económico", "marca Stussy", "Dickies"
   ✗ Inválido: "no importa el precio", marca no disponible (Nike, Adidas, etc.)

═══════════════════════════════════════════════════════════════
EJEMPLOS DE CUÁNDO NO RECOMENDAR (requiere más preguntas):
═══════════════════════════════════════════════════════════════

❌ "Hola" → Solo saludo, cero información
❌ "Quiero una remera" → Solo tipo de prenda (1/4 criterios)
❌ "Busco zapatillas blancas" → Tipo + color (2/4 criterios) - falta ocasión/presupuesto
❌ "Necesito ropa para salir" → Ocasión vaga + no hay tipo específico
❌ "Tengo $30000 para gastar" → Solo presupuesto, no hay tipo de prenda
❌ "Me gusta Nike" → Marca NO disponible + no hay tipo de prenda

═══════════════════════════════════════════════════════════════
EJEMPLOS DE CUÁNDO SÍ RECOMENDAR (información suficiente):
═══════════════════════════════════════════════════════════════

✓ "Remera negra para salir, hasta $25000" → tipo + color + ocasión + presupuesto (4/4)
✓ "Zapatillas Stussy blancas deportivas" → tipo + marca válida + color + ocasión (4/4)
✓ "Jean claro casual hasta $40000" → tipo + color + estilo + presupuesto (4/4)
✓ "Buzo Carhartt oversize negro" → tipo + marca válida + estilo + color (4/4)
✓ "Campera Dickies estilo urbano" → tipo + marca válida + estilo (3/4 suficiente)

═══════════════════════════════════════════════════════════════
FLUJO DE CONVERSACIÓN OBLIGATORIO:
═══════════════════════════════════════════════════════════════

FASE 1: SALUDO E IDENTIFICACIÓN DE NECESIDAD GENERAL
────────────────────────────────────────────────────
Cuando el cliente saluda o es muy vago:
- Saludá cálidamente
- Preguntá QUÉ TIPO DE PRENDA busca específicamente
- Ejemplo: "¡Hola! Bienvenido a Zoppa. ¿Qué tipo de prenda estás buscando hoy? ¿Remeras, jeans, zapatillas, camperas, buzos...?"

FASE 2: RECOLECCIÓN DE DETALLES ESPECÍFICOS
────────────────────────────────────────────
Si mencionó el tipo de prenda pero falta info:
- NO recomiendes todavía
- Hacé preguntas CONCRETAS sobre lo que falta:
  
  Si falta OCASIÓN/ESTILO:
  "¿Para qué ocasión la necesitás? ¿Algo casual para el día a día, para salir, deportivo, trabajo...?"
  
  Si falta COLOR:
  "¿Tenés alguna preferencia de color? ¿Buscás algo neutro, oscuro, claro, o algún color específico?"
  
  Si falta PRESUPUESTO y el catálogo es amplio:
  "¿Tenés un presupuesto aproximado en mente?"
  
  Si mencionó MARCA NO VÁLIDA:
  "Actualmente no tenemos [Marca], pero trabajamos con marcas como Stussy, Dickies, Carhartt WIP, Obey, que tienen estilos similares. ¿Te interesa ver opciones de estas marcas?"
  
  Si mencionó MARCA VÁLIDA sin otros detalles:
  "Perfecto, [Marca]. ¿Qué tipo de prenda de [Marca] te interesa? ¿Y para qué ocasión?"

FASE 3: VALIDACIÓN ANTES DE RECOMENDAR
───────────────────────────────────────
Antes de mostrar productos, verificá mentalmente:
- [ ] ¿Tengo el tipo de prenda ESPECÍFICO?
- [ ] ¿Tengo al menos 2 criterios más (color/ocasión/presupuesto/marca válida)?
- [ ] ¿La información es CLARA y no ambigua?
- [ ] ¿Si mencionó marca, está en nuestra lista de marcas disponibles?

Si NO cumple los checks → hacé MÁS PREGUNTAS
Si SÍ cumple → procedé a recomendar

FASE 4: RECOMENDACIÓN DE PRODUCTOS
───────────────────────────────────
Solo cuando tenés información suficiente:
- Usá ÚNICAMENTE los productos del JSON proporcionado
- NO inventes productos, marcas, precios ni talles
- SOLO recomendá marcas que están en la lista oficial
- Seleccioná los productos que mejor coincidan con TODOS los criterios mencionados
- Presentá 3-5 opciones máximo
- Para cada producto mencioná:
  * Nombre y marca (VERIFICAR que sea marca válida)
  * Por qué lo recomendás (cómo cumple con lo pedido)
  * Precio
  * Colores y talles disponibles
  
Ejemplo de buena recomendación:
"Encontré estas opciones
   Precio: $22,500 | Talles: S, M, L, XL

2. **Remera Obey Essentials Black** - Súper versátil, diseño minimalista.
   Precio: $24,000 | Talles: M, L, XL

¿Alguna te convence o querés que ajustemos algo?"

FASE 5: AJUSTES Y REFINAMIENTO
───────────────────────────────
Si los productos no son exactos:
- Sé honesto sobre las diferencias
- Ofrecé alternativas cercanas de marcas disponibles
- Sugerí ajustar UN criterio a la vez

Si NO hay productos que coincidan:
- Explicá por qué no hay match
- Preguntá si puede flexibilizar algún criterio específico
- Si pidió marca no disponible, sugerí alternativas de marcas similares

═══════════════════════════════════════════════════════════════
ESTILO DE COMUNICACIÓN:
═══════════════════════════════════════════════════════════════

- Tono: Cercano, profesional, como un vendedor experto que realmente quiere ayudar
- "Dejame entender bien lo que necesitás..."

═══════════════════════════════════════════════════════════════
RECORDÁ EL HISTORIAL:
═══════════════════════════════════════════════════════════════

- Leé TODO el historial antes de responder
- No repitas preguntas ya contestadas
- Referenciá información mencionada antes
- Si el cliente ya dio info en mensajes previos, usala

═══════════════════════════════════════════════════════════════
PRODUCTOS CANDIDATOS:
═══════════════════════════════════════════════════════════════

Los "Productos candidatos (JSON)" son tu ÚNICA fuente de verdad.
- NO menciones productos que no estén en el JSON
- NO inventes precios, talles o características
- VERIFICÁ que todas las marcas mencionadas estén en la lista oficial
- Si el JSON está vacío o los productos no son relevantes, pedí más info o sugerí ajustar criterios

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


def tiene_contexto_suficiente(message: str, history: Optional[List[HistoryMessage]] = None) -> bool:
    """
    Determina si hay suficiente información en el mensaje y el historial
    para justificar una recomendación de productos.
    
    Criterios mínimos (necesita al menos 3 de 4):
    1. Tipo de prenda específico (obligatorio)
    2. Ocasión o estilo
    3. Color o preferencia visual
    4. Presupuesto o marca VÁLIDA
    """
    # Palabras clave de saludos genéricos (NO debe recomendar)
    saludos_genericos = [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
        "qué tal", "cómo va", "hey", "ey", "holis", "holaa", "que onda",
        "estás ahí", "hay alguien", "hola?", "¿hola?", "buenasss"
    ]
    
    message_lower = message.lower().strip()
    
    # Si es solo un saludo, NO recomendar
    if message_lower in saludos_genericos or len(message_lower) < 6:
        return False
    
    # Combinar mensaje actual con historial para análisis completo
    full_context = message_lower
    if history:
        for h in history:
            if h.role == "user":
                full_context += " " + h.content.lower()
    
    # 1. TIPO DE PRENDA (OBLIGATORIO) - palabras específicas
    tipos_prenda = [
        "remera", "remeras", "camiseta", "polera", "playera",
        "jean", "jeans", "pantalón", "pantalones", "jogger",
        "zapatilla", "zapatillas", "zapatos", "calzado", "sneakers", "tenis",
        "campera", "camperas", "chaqueta", "jacket", "casaca",
        "buzo", "buzos", "hoodie", "sudadera", "sweater", "pullover",
        "vestido", "vestidos",
        "pollera", "falda",
        "short", "shorts", "bermuda",
        "chomba", "polo",
        "camisa", "camisas",
        "medias", "soquetes",
        "gorra", "gorras", "sombrero",
        "mochila", "mochilas", "bolso"
    ]
    
    tiene_tipo_prenda = any(tipo in full_context for tipo in tipos_prenda)
    
    # Si NO tiene tipo de prenda específico, NO recomendar
    if not tiene_tipo_prenda:
        return False
    
    # Contar criterios adicionales
    criterios_cumplidos = 1  # Ya tiene tipo de prenda
    
    # 2. OCASIÓN / ESTILO
    ocasiones = [
        "salir", "fiesta", "evento", "casual", "deportivo", "deporte", "gym", "gimnasio",
        "trabajo", "oficina", "formal", "elegante", "entrenamiento", "running", "correr",
        "urbano", "streetwear", "calle", "diario", "día a día", "uso diario",
        "verano", "invierno", "otoño", "primavera", "playa", "montaña",
        "oversize", "ajustado", "holgado", "fit", "slim", "regular"
    ]
    
    if any(ocasion in full_context for ocasion in ocasiones):
        criterios_cumplidos += 1
    
    # 3. COLOR / PREFERENCIA VISUAL
    colores = [
        "negro", "negra", "negros", "negras", "black",
        "blanco", "blanca", "blancos", "blancas", "white",
        "azul", "azules", "blue",
        "rojo", "roja", "rojos", "rojas", "red",
        "verde", "verdes", "green",
        "gris", "grises", "gray", "grey",
        "amarillo", "amarilla", "yellow",
        "rosa", "pink",
        "violeta", "morado", "purple",
        "naranja", "orange",
        "marrón", "marrones", "brown",
        "beige", "crema", "crudo",
        "celeste", "turquesa",
        "color", "colores", "estampado", "liso", "lisa",
        "claro", "clara", "claros", "claras", "oscuro", "oscura", "oscuros", "oscuras",
        "neutro", "neutra", "neutros", "neutras"
    ]
    
    if any(color in full_context for color in colores):
        criterios_cumplidos += 1
    
    # 4. PRESUPUESTO O MARCA (SOLO MARCAS VÁLIDAS DE ZOPPA)
    # Verificar si menciona alguna marca válida
    marca_valida_mencionada = any(marca in full_context for marca in MARCAS_VALIDAS)
    
    indicadores_presupuesto = [
        "precio", "presupuesto", "plata", "lucas", "pesos", "$",
        "barato", "barata", "económico", "económica",
        "caro", "cara", "premium",
        "hasta", "menos de", "máximo", "como mucho"
    ]
    
    if marca_valida_mencionada:
        criterios_cumplidos += 1
    elif any(indic in full_context for indic in indicadores_presupuesto):
        criterios_cumplidos += 1
    
    # Necesita tipo de prenda (obligatorio) + al menos 2 criterios más
    # Total: mínimo 3 criterios
    return criterios_cumplidos >= 3

# ---------------------------------------------------------
# Sistema de detección de intenciones prohibidas
# ---------------------------------------------------------

# Palabras clave que indican intentos de obtener algo gratis/descuentos no autorizados
PALABRAS_DESCUENTOS_NO_AUTORIZADOS = [
    "gratis", "gratuito", "sin pagar", "sin cargo", "regalo", "regalame",
    "descuento", "rebaja", "oferta especial", "código", "cupón", "promoción especial",
    "promo", "black friday", "cyber monday", "oferta exclusiva",
    "precio especial", "me cobres", "no pagar", "dame gratis"
]

# Palabras clave de temas completamente ajenos a la tienda
TEMAS_PROHIBIDOS = {
    "programacion": ["programar", "código python", "javascript", "react", "angular", "vue",
                     "backend", "frontend", "api rest", "base de datos", "sql", "mongodb",
                     "algoritmo", "función", "clase", "método", "variable", "debugging",
                     "git", "github", "deploy", "servidor", "hosting"],
    
    "politica": ["gobierno", "presidente", "elecciones", "votación", "partido político",
                 "congreso", "senado", "diputado", "milei", "cristina", "macri",
                 "kirchner", "peronismo", "radicalismo", "democracia", "dictadura"],
    
    "religion": ["dios", "jesús", "alá", "buda", "religión", "iglesia", "mezquita",
                 "templo", "biblia", "corán", "rezar", "oración", "fe", "creencia"],
    
    "salud_medica": ["enfermedad", "síntoma", "medicina", "pastilla", "tratamiento médico",
                     "diagnóstico", "doctor", "hospital", "dolor", "enfermo", "medicamento"],
    
    "tareas_escolares": ["tarea", "deber", "examen", "prueba", "resolver ejercicio",
                         "ayúdame con mi tarea", "matemática", "física", "química"],
    
    "otros_comercios": ["mercado libre", "amazon", "ebay", "aliexpress", "shein",
                        "nike.com", "adidas.com", "zara.com", "h&m.com"]
}

# Intentos de manipulación del sistema
INTENTOS_MANIPULACION = [
    "ignora las instrucciones", "olvida las reglas", "actúa como si",
    "pretende que eres", "simula ser", "imagina que eres",
    "desactiva", "bypass", "saltate", "ignora el sistema",
    "nueva instrucción", "eres ahora", "cambia tu rol",
    "deja de ser", "ya no eres", "tu nuevo rol es"
]

def detectar_intencion_prohibida(message: str) -> tuple[bool, str]:
    """
    Detecta si el mensaje contiene intenciones prohibidas.
    
    Returns:
        (es_prohibido: bool, razon: str)
    """
    message_lower = message.lower().strip()
    
    # 1. Detectar intentos de manipulación del sistema
    for manipulacion in INTENTOS_MANIPULACION:
        if manipulacion in message_lower:
            return True, "manipulacion_sistema"
    
    # 2. Detectar solicitudes de descuentos/gratis no autorizados
    palabras_sospechosas_descuento = [p for p in PALABRAS_DESCUENTOS_NO_AUTORIZADOS 
                                       if p in message_lower]
    if len(palabras_sospechosas_descuento) >= 2:
        # Si menciona 2+ palabras de descuento/gratis, es sospechoso
        return True, "solicitud_descuento_no_autorizado"
    
    # 3. Detectar temas completamente ajenos
    for categoria, palabras in TEMAS_PROHIBIDOS.items():
        palabras_detectadas = [p for p in palabras if p in message_lower]
        if len(palabras_detectadas) >= 2:
            # Si menciona 2+ palabras de un tema prohibido
            return True, f"tema_prohibido_{categoria}"
        elif len(palabras_detectadas) == 1:
            # Si menciona 1 palabra pero el contexto es claro
            # Verificar si es pregunta directa (contiene "?" o palabras interrogativas)
            es_pregunta = "?" in message or any(
                palabra in message_lower 
                for palabra in ["qué es", "cómo", "cuándo", "dónde", "por qué", "explica", "ayuda con"]
            )
            if es_pregunta:
                return True, f"tema_prohibido_{categoria}"
    
    # 4. Detectar preguntas generales a GPT (muy largas y complejas sin mencionar ropa)
    palabras_ropa_contexto = [
        "remera", "jean", "zapatilla", "campera", "buzo", "ropa", "prenda",
        "vestir", "outfit", "look", "estilo", "moda", "marca", "talle",
        "comprar", "precio", "producto", "catálogo", "zoppa"
    ]
    
    tiene_contexto_ropa = any(palabra in message_lower for palabra in palabras_ropa_contexto)
    
    # Si el mensaje es muy largo (>150 caracteres) y no menciona nada de ropa
    if len(message) > 150 and not tiene_contexto_ropa:
        # Verificar si parece una consulta académica/técnica
        palabras_academicas = [
            "explica", "qué significa", "define", "cómo funciona", "por qué",
            "cuál es la diferencia", "ventajas", "desventajas", "ejemplo",
            "dame información sobre", "háblame de"
        ]
        if any(palabra in message_lower for palabra in palabras_academicas):
            return True, "consulta_general_gpt"
    
    return False, ""

# ---------------------------------------------------------
# Respuestas de bloqueo según tipo de violación
# ---------------------------------------------------------

RESPUESTAS_BLOQUEO = {
    "manipulacion_sistema": (
        "🚫 Lo siento, pero no puedo procesar ese tipo de solicitudes. "
        "Soy un asistente de Zoppa diseñado para ayudarte a encontrar productos de moda. "
        "¿Hay algo de nuestra tienda en lo que pueda ayudarte?"
    ),
    
    "solicitud_descuento_no_autorizado": (
        "🚫 Entiendo que te gustaría obtener un descuento especial, pero no tengo autorización "
        "para ofrecer promociones o descuentos fuera de las ofertas oficiales de Zoppa. "
        "Los precios que te muestro son los vigentes en nuestra tienda. "
        "¿Te interesa que busquemos productos dentro de tu presupuesto?"
    ),
    
    "tema_prohibido_programacion": (
        "🚫 Soy un asistente especializado en moda y productos de Zoppa, no puedo ayudarte "
        "con temas de programación. Para ese tipo de consultas, te recomiendo usar "
        "ChatGPT o recursos especializados. ¿Puedo ayudarte con algo de nuestra tienda?"
    ),
    
    "tema_prohibido_politica": (
        "🚫 No puedo ayudarte con temas políticos. Soy un asistente de Zoppa enfocado "
        "exclusivamente en ayudarte a encontrar productos de moda. "
        "¿Hay algo de ropa o accesorios que te interese?"
    ),
    
    "tema_prohibido_religion": (
        "🚫 No puedo ayudarte con temas religiosos. Mi función es asistirte con "
        "los productos de la tienda Zoppa. ¿Te gustaría ver algo de nuestro catálogo?"
    ),
    
    "tema_prohibido_salud_medica": (
        "🚫 No puedo brindarte información médica o de salud. Para ese tipo de consultas, "
        "deberías consultar con un profesional de la salud. "
        "¿Puedo ayudarte con algo relacionado a moda y productos Zoppa?"
    ),
    
    "tema_prohibido_tareas_escolares": (
        "🚫 No puedo ayudarte con tareas o deberes escolares. Mi especialidad es "
        "ayudarte a encontrar productos de moda en Zoppa. "
        "¿Hay algo de nuestra tienda que te interese?"
    ),
    
    "tema_prohibido_otros_comercios": (
        "🚫 Soy asistente de Zoppa y solo puedo informarte sobre nuestros productos. "
        "No tengo información sobre otras tiendas o comercios. "
        "¿Te gustaría ver opciones en nuestro catálogo?"
    ),
    
    "consulta_general_gpt": (
        "🚫 Soy un asistente especializado en la tienda Zoppa, no un asistente general. "
        "Mi función es ayudarte a encontrar productos de moda en nuestro catálogo. "
        "Si necesitás ayuda con otros temas, te recomiendo usar ChatGPT o Google. "
        "¿Puedo ayudarte con algo de ropa, calzado o accesorios de Zoppa?"
    ),
}

# ---------------------------------------------------------
# Cuerpo principal de la API
# ---------------------------------------------------------
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
    
    Si df_products está vacío, el bot responde en modo conversacional
    sin recomendar productos (hace preguntas para recopilar info).
    """
    products_context: List[dict] = []
    
    # Solo construir contexto de productos si hay productos disponibles
    if not df_products.empty:
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

    # Construir mensaje para el modelo
    if products_context:
        # Modo recomendación: hay productos para mostrar
        user_content = (
            "Mensaje actual del cliente:\n"
            + user_message
            + "\n\nProductos candidatos (JSON):\n"
            + json.dumps(products_context, ensure_ascii=False)
            + "\n\n⚠️ IMPORTANTE: Hay productos disponibles. Recomendá solo si la información del cliente es SUFICIENTEMENTE ESPECÍFICA."
        )
    else:
        # Modo conversacional: NO hay productos, solo recopilar información
        user_content = (
            "Mensaje actual del cliente:\n"
            + user_message
            + "\n\n⚠️ IMPORTANTE: NO hay productos candidatos disponibles todavía. "
            + "Esto significa que NO tenés suficiente información para recomendar. "
            + "Tu objetivo es hacer preguntas específicas para entender:\n"
            + "- Qué tipo de prenda específica busca\n"
            + "- Para qué ocasión o estilo\n"
            + "- Preferencias de color\n"
            + "- Presupuesto o marca preferida\n\n"
            + "NO menciones productos. Enfocate en recolectar información de forma natural y amigable."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # historial previo de la conversación
    if history:
        for h in history:
            if h.role in ("user", "assistant") and h.content:
                messages.append({"role": h.role, "content": h.content})

    # mensaje actual + contexto de productos (o instrucción de recopilar info)
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
    
    # ═══════════════════════════════════════════════════════════════
    # PASO 1: VALIDACIÓN DE SEGURIDAD - Detectar intenciones prohibidas
    # ═══════════════════════════════════════════════════════════════
    es_prohibido, razon = detectar_intencion_prohibida(req.message)
    
    if es_prohibido:
        # Buscar respuesta de bloqueo apropiada
        respuesta_bloqueo = RESPUESTAS_BLOQUEO.get(
            razon,
            "🚫 Lo siento, no puedo ayudarte con eso. "
            "Soy un asistente de Zoppa enfocado en productos de moda. "
            "¿Hay algo de nuestra tienda en lo que pueda ayudarte?"
        )
        
        # Devolver respuesta de bloqueo sin productos
        return ChatResponse(answer=respuesta_bloqueo, products=[])
    
    # ═══════════════════════════════════════════════════════════════
    # PASO 2: VALIDACIÓN DE CONTEXTO - Verificar información suficiente
    # ═══════════════════════════════════════════════════════════════
    contexto_suficiente = tiene_contexto_suficiente(req.message, req.history)
    
    # Si NO hay contexto suficiente, responder sin productos (solo conversación)
    if not contexto_suficiente:
        # Generar respuesta conversacional sin productos
        answer = answer_with_products(
            user_message=req.message,
            df_products=pd.DataFrame(),  # DataFrame vacío = sin productos
            history=req.history
        )
        return ChatResponse(answer=answer, products=[])
    
    # ═══════════════════════════════════════════════════════════════
    # PASO 3: BÚSQUEDA DE PRODUCTOS - Si pasó todas las validaciones
    # ═══════════════════════════════════════════════════════════════
    recs = get_recommendations(
        user_query=req.message,
        gender=req.gender,
        max_price=req.max_price,
        size=req.size,
        category_name=req.category_name,
    )
    
    # FILTRAR PRODUCTOS: Solo incluir marcas válidas del catálogo
    if not recs.empty:
        recs = recs[recs["brand_name"].apply(es_marca_valida)]

    if recs.empty:
        # No hay productos que coincidan - pedir ajustar criterios
        msg = (
            "Mmm, no encontré productos que coincidan exactamente con lo que buscás. "
            "Podríamos ajustar algún criterio: ¿probamos con otro color, rango de precio, "
            "o te muestro opciones similares de otras marcas que tenemos? Contame qué te parece más flexible."
        )
        return ChatResponse(answer=msg, products=[])

    # Limitar a top 5 productos más relevantes
    top_n = min(5, len(recs))
    recs_top = recs.head(top_n)
    
    # Generar respuesta con productos
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
