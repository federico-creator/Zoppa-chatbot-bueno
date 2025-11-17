#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesa el catálogo de Zoppa a partir de CSVs (products_rows, brands_rows, categories_rows),
genera un catálogo unificado + texto optimizado para embeddings y, opcionalmente, embeddings y PCA.

Mapeo de género (en TODAS las tablas):
    1 -> hombre
    2 -> mujer
    3 -> unisex

Entradas:
    --products "./data/products_rows.csv"
    --brands "./data/brands_rows.csv"
    --categories "./data/categories_rows.csv"

Salidas:
    outdir/catalog.parquet              -> metadatos + campos normalizados
    outdir/products_embeddings.parquet  -> id + vector (lista float)
    outdir/products_pca.parquet         -> (opcional) catálogo + emb_0..emb_k
    outdir/pca.pkl                      -> modelo PCA

Requisitos:
    pip install pandas numpy tqdm python-dotenv scikit-learn openai joblib
    export OPENAI_API_KEY="tu_clave"
"""
import os
import re
import json
import math
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
import joblib


# ---------------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------------

def parse_json_list(value: Any) -> List[str]:
    """Intenta interpretar una cadena como lista JSON; si falla, devuelve [value] o []."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    # si parece lista JSON
    if s.startswith("[") and s.endswith("]"):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(v).strip() for v in data if str(v).strip()]
        except Exception:
            pass
    # fallback: separadores comunes
    parts = re.split(r"[|;,]+", s)
    return [p.strip() for p in parts if p.strip()]


def parse_relacionados(value: Any) -> List[int]:
    """
    Convierte algo como "{6, 12}" o "6,12" en [6, 12].
    Si no se puede parsear, devuelve [].
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    s = s.strip("{}")
    ids: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


# ---------------------------------------------------------------------------
# Manejo de género (1=hombre, 2=mujer, 3=unisex)
# ---------------------------------------------------------------------------

def _map_gender_code(code: int) -> Optional[str]:
    if code == 1:
        return "hombre"
    if code == 2:
        return "mujer"
    if code == 3:
        return "unisex"
    return None


def _normalize_gender_field(value: Any) -> Optional[str]:
    """
    Normaliza cualquier campo de género:
      - 1/2/3 o "1"/"2"/"3"
      - "hombre", "mujer", "unisex"
      - variantes tipo "men", "women", "male", "female"
    Devuelve "hombre" / "mujer" / "unisex" o None si no se puede.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    # primero intentamos como entero 1/2/3
    try:
        code = int(value)
        label = _map_gender_code(code)
        if label:
            return label
    except Exception:
        pass

    # luego como string
    s = str(value).strip().lower()
    if not s:
        return None

    if s in {"1", "2", "3"}:
        try:
            code = int(s)
            return _map_gender_code(code)
        except Exception:
            pass

    if s in {"h", "hombre", "men", "male"}:
        return "hombre"
    if s in {"m", "mujer", "women", "female"}:
        return "mujer"
    if s in {"u", "unisex"}:
        return "unisex"

    return None


def infer_gender_label(
    product_genre: Any = None,
    brand_gender: Any = None,
    category_genre: Any = None,
) -> str:
    """
    Regla de prioridad para género del producto:
      1) genero del producto (products.genre)
      2) genero de la marca (brands.gender)
      3) genero de la categoría (categories.genre)
    Cada uno se interpreta con la convención 1=hombre, 2=mujer, 3=unisex.
    Si todo falla, devuelve "unisex".
    """
    for candidate in (product_genre, brand_gender, category_genre):
        label = _normalize_gender_field(candidate)
        if label is not None:
            return label
    return "unisex"


# ---------------------------------------------------------------------------
# Manejo de talles (incluyendo zapatos)
# ---------------------------------------------------------------------------

# Mapear nombres de columnas a etiquetas legibles
TALLE_LABEL_MAP: Dict[str, str] = {
    "size_unico": "Único",
    "size_xxxs": "XXXS",
    "size_xxs": "XXS",
    "size_xs": "XS",
    "size_s": "S",
    "size_m": "M",
    "size_l": "L",
    "size_xl": "XL",
    "size_xxl": "XXL",
    "size_xxxl": "XXXL",
    "size_xxxxl": "XXXXL",
    "size_xxxxxl": "XXXXXL",
    # zapatos que ya existen en tu CSV; el código es genérico para futuros size_42, etc.
    "size_20": "20",
    "size_22": "22",
    "size_24": "24",
    "size_26": "26",
    "size_28": "28",
    "size_30": "30",
    "size_32": "32",
    "size_34": "34",
    "size_36": "36",
    "size_38": "38",
    "size_40": "40",
}


def extract_sizes(row: pd.Series) -> List[str]:
    """
    Devuelve la lista de talles disponibles para un producto, usando TODAS
    las columnas que empiezan con 'size_' (ropa y zapatos).

    Criterio de disponibilidad: valor > 0 (stock positivo).
    """
    sizes: List[str] = []

    for col in row.index:
        if not col.startswith("size_"):
            continue

        val = row[col]
        available = False

        if isinstance(val, (int, float)) and not pd.isna(val) and val > 0:
            available = True
        elif isinstance(val, str):
            if val.strip():
                try:
                    num = float(val.replace(",", "."))
                    available = num > 0
                except Exception:
                    # texto no numérico pero no vacío -> contamos como disponible
                    available = True

        if not available:
            continue

        label = TALLE_LABEL_MAP.get(col)
        if label is None:
            # generalización: size_42 -> "42"
            raw = col.replace("size_", "")
            label = raw.upper()

        sizes.append(label)

    # eliminar duplicados conservando orden
    seen = set()
    out: List[str] = []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Catálogo unificado
# ---------------------------------------------------------------------------

def build_catalog(
    products_csv: str,
    brands_csv: str,
    categories_csv: str,
) -> pd.DataFrame:
    """
    Carga products_rows, brands_rows y categories_rows y genera un DataFrame de catálogo
    con columnas amigables para el chatbot y un campo embed_text listo para embeddings.
    """
    df_p = pd.read_csv(products_csv)
    df_b = pd.read_csv(brands_csv)
    df_c = pd.read_csv(categories_csv)

    # Normaliza nombres básicos de columnas
    df_p = df_p.rename(columns=str.strip)
    df_b = df_b.rename(columns=str.strip)
    df_c = df_c.rename(columns=str.strip)

    # Índices para join rápido
    brands_by_id = df_b.set_index("id")
    cats_by_id = df_c.set_index("id")

    records: List[Dict[str, Any]] = []

    for _, row in df_p.iterrows():
        pid = int(row["id"])
        brand_id = row.get("brand_id")
        cat_id = row.get("category_id")

        brand_name = None
        brand_origin = None
        brand_gender_raw = None

        if not pd.isna(brand_id) and brand_id in brands_by_id.index:
            b = brands_by_id.loc[brand_id]
            brand_name = str(b.get("name", "")).strip() or None
            brand_origin = str(b.get("origin", "")).strip() or None
            brand_gender_raw = b.get("gender", None)

        category_name = None
        category_logo = None
        category_genre_raw = None

        if not pd.isna(cat_id) and cat_id in cats_by_id.index:
            c = cats_by_id.loc[cat_id]
            category_name = str(c.get("name", "")).strip() or None
            category_logo = str(c.get("logo", "")).strip() or None
            category_genre_raw = c.get("genre", None)

        product_genre_raw = row.get("genre", None)

        # Género final del producto (prioridad: producto > marca > categoría)
        gender_label = infer_gender_label(
            product_genre=product_genre_raw,
            brand_gender=brand_gender_raw,
            category_genre=category_genre_raw,
        )

        # imágenes
        image_list = parse_json_list(row.get("images", None))
        primary_image = image_list[0] if image_list else None

        # talles
        sizes = extract_sizes(row)

        # relacionados
        related_ids = parse_relacionados(row.get("relacionados", None))

        # campos básicos
        name = str(row.get("name", "")).strip()
        description = str(row.get("description", "") or "").strip()
        color = str(row.get("color", "") or "").strip()
        url = str(row.get("url", "") or "").strip()

        price = row.get("price", None)
        try:
            price = float(price) if not pd.isna(price) else None
        except Exception:
            price = None

        discount = row.get("discount", None)
        try:
            discount = float(discount) if not pd.isna(discount) else None
        except Exception:
            discount = None

        # precio efectivo (por ahora igual a price; se puede ajustar si discount es %)
        effective_price = price

        rec: Dict[str, Any] = {
            "id": pid,
            "name": name,
            "brand_id": int(brand_id) if not pd.isna(brand_id) else None,
            "brand_name": brand_name,
            "brand_origin": brand_origin,
            "gender": gender_label,
            "category_id": int(cat_id) if not pd.isna(cat_id) else None,
            "category_name": category_name,
            "category_logo": category_logo,
            "color": color,
            "price": price,
            "discount": discount,
            "effective_price": effective_price,
            "url": url,
            "images": image_list,
            "primary_image": primary_image,
            "related_ids": related_ids,
            "raw_product_genre": product_genre_raw,
            "raw_brand_gender": brand_gender_raw,
            "raw_category_genre": category_genre_raw,
            "created_at": row.get("created_at", None),
        }

        # talles disponibles
        rec["sizes"] = sizes

        # descripción rica para embeddings
        desc_pieces: List[str] = []

        if name:
            desc_pieces.append(f"Producto: {name}")
        if brand_name:
            desc_pieces.append(f"Marca: {brand_name}")
        if gender_label:
            desc_pieces.append(f"Género: {gender_label}")
        if category_name:
            desc_pieces.append(f"Categoría: {category_name}")
        if color:
            desc_pieces.append(f"Color: {color}")
        if sizes:
            desc_pieces.append("Talles disponibles: " + ", ".join(sizes))
        else:
            desc_pieces.append("Talles disponibles: consultar")

        if price is not None:
            desc_pieces.append(f"Precio aproximado: {price:.0f} ARS")
        if brand_origin:
            desc_pieces.append(f"Origen de la marca: {brand_origin}")
        if description:
            desc_pieces.append("Descripción: " + description)

        rec["embed_text"] = " | ".join(desc_pieces)

        records.append(rec)

    catalog = pd.DataFrame.from_records(records)

    # Deduplicar productos evidentes por (name, brand_name, category_name, color)
    catalog = catalog.drop_duplicates(
        subset=["name", "brand_name", "category_name", "color"],
        keep="first",
    ).reset_index(drop=True)

    return catalog


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def batched(iterable, n: int = 128):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_embeddings(
    catalog: pd.DataFrame,
    client: OpenAI,
    model: str = "text-embedding-3-small",
    text_col: str = "embed_text",
    batch_size: int = 128,
) -> np.ndarray:
    """Genera embeddings para cada fila de catalog[text_col]. Devuelve np.ndarray [N, D]."""
    texts = catalog[text_col].fillna("").astype(str).tolist()
    vectors: List[List[float]] = []

    total = math.ceil(len(texts) / batch_size)
    for chunk in tqdm(batched(texts, n=batch_size), total=total, desc="Embeddings"):
        resp = client.embeddings.create(model=model, input=chunk)
        for d in resp.data:
            vectors.append(d.embedding)

    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Preprocesa catálogo Zoppa y genera embeddings.")
    parser.add_argument("--products", required=True, help="Ruta al CSV de products_rows")
    parser.add_argument("--brands", required=True, help="Ruta al CSV de brands_rows")
    parser.add_argument("--categories", required=True, help="Ruta al CSV de categories_rows")
    parser.add_argument("--outdir", default="./artifacts", help="Carpeta de salida")
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Modelo de embeddings de OpenAI",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Si se pasa, no calcula PCA ni products_pca.parquet",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=int(os.environ.get("PCA_DIM", "128")),
        help="Dimensión objetivo de PCA (si se usa)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1) Construir catálogo unificado ---
    print("▶ Construyendo catálogo unificado...")
    catalog = build_catalog(args.products, args.brands, args.categories)

    cat_path = os.path.join(args.outdir, "catalog.parquet")
    catalog.to_parquet(cat_path, index=False)
    print(f"✔ Catálogo guardado en: {cat_path} ({catalog.shape[0]} productos)")

    # --- 2) Embeddings con OpenAI ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY en el entorno o .env")

    client = OpenAI(api_key=api_key)

    print(f"▶ Generando embeddings con modelo: {args.embedding_model} ...")
    vecs = generate_embeddings(
        catalog=catalog,
        client=client,
        model=args.embedding_model,
    )

    if vecs.shape[0] != catalog.shape[0]:
        raise RuntimeError("Cantidad de embeddings no coincide con tamaño del catálogo")

    dim = vecs.shape[1]
    emb_df = pd.DataFrame(
        {
            "id": catalog["id"].tolist(),
            "embedding": vecs.tolist(),
        }
    )
    emb_path = os.path.join(args.outdir, "products_embeddings.parquet")
    emb_df.to_parquet(emb_path, index=False)
    print(f"✔ Embeddings crudos guardados en: {emb_path} (N={vecs.shape[0]}, dim={dim})")

    # --- 3) PCA opcional ---
    if not args.no_pca:
        target_dim = min(args.pca_dim, dim)
        print(f"▶ Ajustando PCA a dimensión {target_dim} ...")
        pca = PCA(n_components=target_dim, random_state=42)
        reduced = pca.fit_transform(vecs)

        joblib.dump(pca, os.path.join(args.outdir, "pca.pkl"))
        print("✔ Modelo PCA guardado en pca.pkl")

        red_cols = [f"emb_{i}" for i in range(reduced.shape[1])]
        reduced_df = pd.DataFrame(reduced, columns=red_cols)
        merged = pd.concat([catalog, reduced_df], axis=1)

        pca_path = os.path.join(args.outdir, "products_pca.parquet")
        merged.to_parquet(pca_path, index=False)
        print(f"✔ Catálogo + embeddings reducidos guardado en: {pca_path} ({merged.shape})")
    else:
        print("ⓘ PCA deshabilitado (--no-pca).")

    print("✅ Preproceso completo.")


if __name__ == "__main__":
    main()
