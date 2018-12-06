import pymc3 as pm
import theano.tensor as tt

class CloneTreeGenotypes(pm.Categorical):
    """
    Clone tree genotypes (Markov chain)
    Parameters
    --------
    P : tensor
        transition probabilities
        shape = (N_states, N_states)
    PA : tensor
        stationary probabilities
        shape = (N_states)
    edges : tensor
        edge matrix
        shape = (N_edges, 2)
        
        
    NOTE (for me): Everything needs to be 0-indexed (unlike what the edge matrix currently is)
    """

    def __init__(self, PA=None, P=None, edges = None, k = None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.P = P
        self.PA = PA
        self.edges = edges
        self.k = k
        self.mean = 0.
        self.mode = tt.cast(0, dtype='int64')

    def logp(self, x):
        P = self.P
        PA = self.PA
        edges = self.edges
        k = self.k

        #x = tt.clip(x, 0, k-1)

        ## set root node to first row's target
        root_node = edges[(edges[:, 0] == -1), 1]
        ## remove the dummy -1 to root row
        #edges = edges[1:,]
        edges = edges[~(edges[:,0] == -1)]

        sources = edges[:,0]
        targets = edges[:,1]

        ## Get relevant transition terms
        PS = P[x[sources]]
        x_i = x[targets]
        ou_like = pm.Categorical.dist(PS).logp(x_i) ## Might not need .dist anymore
        return pm.Categorical.dist(PA).logp(x[root_node]) + tt.sum(ou_like)
