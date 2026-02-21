import faiss
import numpy as np
import time


def faiss_search(search_space, topK=2, query_set=None):

    search_space = np.array(search_space).astype('float32')
    query_set = np.array(query_set).astype('float32')

    dim = search_space.shape[1]

    print("--------------------------------start build faiss index----------------------------------")
    start_time = time.time()

    index = faiss.IndexFlatL2(dim)   # exact L2 search
    index.add(search_space)

    end_time = time.time()
    print(f"build time: {(end_time-start_time)/60:.2f} min")
    print("--------------------------------build finish-------------------------------------------")

    print("--------------------------------start search--------------------------------------------")
    start_time = time.time()

    dist, ind = index.search(query_set, topK)

    end_time = time.time()
    print(f"search time: {(end_time-start_time)/60:.2f} min")
    print("--------------------------------search finish-------------------------------------------")

    return dist, ind