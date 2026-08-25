from .embeddings import Embeddings

BATCH_SIZE = 15000
EMBEDDING_MODEL = Embeddings(normalize=True)          
PCA_ARGS = {'n_components': 50, 'svd_solver': 'full'}