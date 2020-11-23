import numpy as np
import scipy as sp
import scipy.sparse as sp_sp
from numpy.linalg import norm
from scipy.sparse.linalg import norm as sp_norm
from collections import defaultdict
from typing import List, Tuple, Callable, Union
from ..embeddings import FastTextWrapper


class Util:
    @classmethod
    def build_similarity_matrix(cls, sentence_vectors: List[Union[np.ndarray, sp_sp.spmatrix]],
                                sim_func: Callable[[List[str], List[str]], float] = None,
                                normalized: bool = False) -> List[List[float]]:
        """
        Params:
            - sentence_vectors: sentence vectors
            - sim_func: similarity function
            - normalized: normalize similarity_matrix's column sum to 1 or not
        returns: similarity matrix (sentence index, sentence index2, score)
        """
        n = len(sentence_vectors)
        sim_matrix = np.zeros([n, n])
        if not sim_func:
            sim_func = cls.compute_cosine_similarity
        for i, sentence_vector in enumerate(sentence_vectors):
            sim_matrix[i][i] = 1
            for j in range(i + 1, n):
                sentence_vector2 = sentence_vectors[j]
                sim_matrix[i][j] = sim_matrix[j][i] = sim_func(sentence_vector, sentence_vector2)
        if normalized:
            sim_matrix = sim_matrix / sim_matrix.sum(axis=1)
        return sim_matrix

    @staticmethod
    def build_tokenized_sentence_vectors(tokenized_sentences: List[List[str]]) -> List[sp_sp.spmatrix]:
        n = len(tokenized_sentences)
        token_set = set()
        token2id = dict()
        counters = []
        token_id = 0
        for tokens in tokenized_sentences:
            counter = defaultdict(int)
            for token in tokens:
                if token not in token2id:
                    token2id[token] = token_id
                    token_id += 1
                    token_set.add(token)
                counter[token] += 1
            counters.append(counter)

        sentence_vectors = []
        for i, tokens in enumerate(tokenized_sentences):
            tokens = set(tokens)
            i_arr = [0] * len(tokens)
            j_arr = [token2id[token] for token in tokens]
            data = [counters[i][token] for token in tokens]
            sentence_vectors.append(sp_sp.coo_matrix((data, (i_arr, j_arr)),
                                                     (n, token_id)))
        return sentence_vectors

    @staticmethod
    def build_fasttext_sentence_vectors(sentences: List[str], model: FastTextWrapper) -> List[np.ndarray]:
        if model is None:
            raise Exception('No model provided')
        return [model.inference(sentence) for sentence in sentences]

    @staticmethod
    def compute_cosine_similarity(
            sentence_vector: Union[np.ndarray, sp_sp.spmatrix],
            sentence_vector2: Union[np.ndarray, sp_sp.spmatrix]) -> float:
        """
        Params:
            - sentence_vector: sentence vector
            - sentence_vector2: sentence vector2
        returns: similarity score of the two sentences
        """
        score = sentence_vector @ (sentence_vector2.transpose())
        if isinstance(sentence_vector, sp_sp.spmatrix) and \
                isinstance(sentence_vector2, sp_sp.spmatrix):
            score = score.todense()[0, 0]
            nrm, nrm2 = sp_norm(sentence_vector), sp_norm(sentence_vector2)
            score = score / ( nrm * nrm2 ) if nrm * nrm2 > 0 else 0
        else:
            nrm, nrm2 = norm(sentence_vector), norm(sentence_vector2)
            score = score / ( nrm * nrm2 ) if nrm * nrm2 > 0 else 0
        return score

    @staticmethod
    def compute_centroid(sentence_vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(np.array(sentence_vectors), 0)

    @staticmethod
    def pagerank(M, num_iterations: int = 100, d: float = 0.85):
        """PageRank: The trillion dollar algorithm.

        Parameters
        ----------
        M : numpy array
            adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
            sum(i, M_i,j) = 1
        num_iterations : int, optional
            number of iterations, by default 100
        d : float, optional
            damping factor, by default 0.85

        Returns
        -------
        numpy array
            a vector of ranks such that v_i is the i-th rank from [0, 1],
            v sums to 1

        """
        N = M.shape[1]
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
        M_hat = (d * M + (1 - d) / N)
        for i in range(num_iterations):
            v = M_hat @ v
        return v
