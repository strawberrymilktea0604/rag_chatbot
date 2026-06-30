import pandas as pd

df = pd.read_csv("backend/chunks_all.csv")

# Sample a few chunks per document
sampled = (
    df.groupby("document_id")
    .apply(lambda x: x.sample(n=20, random_state=42))
    .reset_index(drop=True)
)

# Save these as starting material
sampled[["document_id", "chunk_id", "page", "text"]].to_csv("qa_candidates.csv", index=False)
