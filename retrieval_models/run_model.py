from retrieval_models.vsm.cosinsim import cosinsim
from retrieval_models.BM25.bm25 import bm25
from retrieval_models.BIR.bir import bir
from retrieval_models.BIR.bir_rel import bir_rel
from retrieval_models.BIR.ex_bir import ex_bir
from retrieval_models.BIR.ex_bir_rel import ex_bir_rel
from retrieval_models.LSI.lsi import lsi
from retrieval_models.lm.mle import mle
from retrieval_models.lm.laplace import laplace
from retrieval_models.lm.jm import jm
from retrieval_models.lm.dirichlet import dirichlet

from config import BM25_K1, BM25_B, LM_SMOOTHING_LAMBDA, LM_DIRICHLET_MU, LSI_K


def run_model(name, idx, query_vec, rel_docs: list[int]):
    """Dispatch query to the right model function, return ranked list."""
    if name == "vsm":
        return cosinsim(idx, query_vec)
    elif name == "bm25":
        return bm25(idx, query_vec, k=BM25_K1, b=BM25_B)
    elif name == "bir":
        return bir(idx, query_vec)
    elif name == "bir_rel":
        return bir_rel(idx, query_vec, relevant_docs=rel_docs)
    elif name == "ext_bir":
        return ex_bir(idx, query_vec)
    elif name == "ext_bir_rel":
        return ex_bir_rel(idx, query_vec, relevant_docs=rel_docs)
    elif name == "lsi":
        return lsi(idx, query_vec, k=LSI_K)
    elif name == "lm_mle":
        return mle(idx, query_vec)
    elif name == "lm_laplace":
        return laplace(idx, query_vec)
    elif name == "lm_jm":
        return jm(idx, query_vec, _lambda=LM_SMOOTHING_LAMBDA)
    elif name == "lm_dirichlet":
        return dirichlet(idx, query_vec, mu=LM_DIRICHLET_MU)
    else:
        raise ValueError(f"Unknown model: {name}")