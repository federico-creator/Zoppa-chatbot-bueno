import pandas as pd

cat = pd.read_parquet("artifacts/catalog.parquet")
emb = pd.read_parquet("artifacts/products_embeddings.parquet")
pca_df = pd.read_parquet("artifacts/products_pca.parquet")

print("Data shapes:")

print("catalog:", cat.shape)
print("embeddings:", emb.shape)
print("pca_df:", pca_df.shape)

print(cat[['id','name','brand_name','gender','category_name','color','sizes']].head(10))
print(cat[['embed_text']].head(3))
print(emb.head(3))
