
import torch
import chromadb
import transformers

from importlib.metadata import version
from FlagEmbedding import BGEM3FlagModel

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("FlagEmbedding:", version("FlagEmbedding"))
print("ChromaDB:", chromadb.__version__)

assert torch.cuda.is_available(), "CUDA is not available!"

print("\nLoading BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

print("Model loaded!")

texts = [
    "Ποια είναι η διαδικασία για την έγκριση μιας δαπάνης;",
    "Ποια είναι η προθεσμία υποβολής αιτήσεων;"
]

result = model.encode(
    texts,
    batch_size=2,
    max_length=512,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)

embeddings = result["dense_vecs"]

print("Embedding shape:", embeddings.shape)
print("\nEverything works!")
