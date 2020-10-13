import numpy as np
import scipy.sparse as sp
import warnings
import itertools

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize
import torch



class DataProcessor:

    def __init__(self, A, X, L, K=3, p_val=0.1, p_test=0.05, p_nodes=0.0, batch_size=100,
                  scale=False, seed=334,with_attri=True):

        np.random.seed(seed)

        if p_nodes > 0:
            A = self.__setup_inductive(A, X, p_nodes)
        else:
            self.X = X # sparse matrix
            self.feed_dict = None
        if with_attri:
            self.input_dim = self.X.shape[-1]
        else:
            self.input_dim = self.A.shape[-1]
        self.N, self.D = X.shape

        self.batch_size= batch_size
        self.scale = scale

        if p_val + p_test > 0:
            train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
                A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
                undirected=False)#(A == A.T).nnz == 0)
            A_train = edges_to_sparse(train_ones, self.N)
            hops = get_hops(A_train, K)
        else:
            hops = get_hops(A, K)
        self._hops = hops
        self._scale_terms  = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}



        if p_val > 0:
            val_edges = np.row_stack((val_ones, val_zeros)) # N x 2
            self.left_val = self.X[val_edges[:,0],:]
            self.right_val = self.X[val_edges[:,1],:]
            self.val_ground_truth = A[val_edges[:,0], val_edges[:,1]].A1

        if p_test > 0:
            self.test_edges = test_edges = np.row_stack((test_ones, test_zeros)) # N x 2
            self.left_test = self.X[test_edges[:,0],:]
            self.right_test = self.X[test_edges[:,1],:]
            self.test_ground_truth = A[test_edges[:,0], test_edges[:,1]].A1

    def __build(self):

        w_init = tf.contrib.layers.xavier_initializer
        sizes = [self.D] + self.n_hidden
        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float64,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float64, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.batch_input_tf, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.tanh(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float64, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float64, initializer=w_init())
        mu_ = tf.matmul(encoded, W_mu) + b_mu
        self.mu =  tf.nn.tanh(mu_)

        self.codebooks = tf.cast(tf.get_variable("codebook", [self.M * self.K, self.L]),tf.float64)
        logits = self.mu
        W_mu2 = tf.get_variable(name='W_mu2', shape=[self.L, self.L], dtype=tf.float64, initializer=w_init())
        b_mu2 = tf.get_variable(name='b_mu2', shape=[self.L], dtype=tf.float64, initializer=w_init())
        logits_a = tf.nn.tanh(tf.matmul(logits,W_mu2)+b_mu2)
        logits_a = tf.reshape(logits_a,[-1,self.M,self.K])
        logits_a = tf.nn.softmax(logits_a,dim=-1)
        self.atten_index = tf.cast(tf.argmax(logits_a,axis=-1),tf.int32)
        logits_a = tf.reshape(logits_a, [-1, self.M*self.K])

        logits = logits*logits_a
        logits = tf.reshape(logits, [-1, self.M, self.K], name="logits")
        # D = self._gumbel_softmax(logits, self._TAU, sampling=True)
        D = tf.nn.softmax(logits,-1)
        gumbel_output = tf.reshape(D, [-1, self.M * self.K])  # ~ (B, M * K)
        # self.maxp = tf.reduce_mean(tf.reduce_max(D, axis=2))
        y_hat = self._decode(gumbel_output, self.codebooks)
        loss = 0.5 * tf.reduce_sum((y_hat - self.mu) ** 2, axis=1)
        self.loss_quatization = tf.reduce_mean(loss, name="loss")

        # recontruct rules
        self.max_index = max_index = tf.cast(tf.argmax(logits, axis=2), tf.int32)
        self.offset = offset = tf.range(self.M, dtype="int32") * self.K
        self.codes_with_offset=codes_with_offset = max_index + offset[None, :]
        selected_vectors = tf.gather(self.codebooks, codes_with_offset)  # ~ (B, M, H)
        self.reconstructed_embed = tf.reduce_sum(selected_vectors, axis=1)  # ~ (B, H)

    def fetch_batch(self):

        while True:
            data,_,neig_ty =  to_triplets(sample_all_hops(self._hops), self._scale_terms)
            num = data.shape[0]
            if num >= self.batch_size:
                its = int(num/self.batch_size)
            else:
                its = 1
                self.batch_size = num
            arr = np.arange(data.shape[0])
            np.random.shuffle(arr)
            np.random.shuffle(arr)
            for i in range(its):
                range_index = arr[(i*self.batch_size):(i+1)*self.batch_size]
                triplet_batch =data[range_index]
                # scale_batch = scal[range_index]
                neig_batch = neig_ty[range_index]
                triplet_batch_ = triplet_batch.transpose().reshape(-1)
                triplet_batch1 = np.unique(triplet_batch_)
                c = np.array([np.where(triplet_batch1 == i)[0][0] for i in  triplet_batch_])
                c = c.reshape(3,self.batch_size).transpose()

                yield spy_sparse2torch_sparse(self.X[triplet_batch1,:]),c,neig_batch.transpose().astype(np.float64)



def edges_to_sparse(edges, N, values=None):
    
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
   
    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert sp.csgraph.connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = sp.csgraph.minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def sparse_feeder(M):
    M = sp.coo_matrix(M)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def cartesian_product(x, y):
    
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def score_link_prediction(labels, scores):
  

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=False):
   
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        trace.append((f1_micro, f1_macro))

    return np.array(trace).mean(0)


def get_hops(A, K):
    
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K+1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop

    # hops[-1][hops[-1]>0] = 1
    # hops[-1] = 1 -  hops[-1].toarray()
    # hops[-1] = sp.lil_matrix(hops[-1])
    return hops

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
def sample_last_hop(A, nodes):
   
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]

    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]
    return sampled


def sample_all_hops(hops, nodes=None):
   
    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T


def to_triplets(sampled_hops, scale_terms):
  
    triplets = []
    triplet_scale_terms = []
    neigh_index = []
    h =- 333
    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]

        tmp = np.array([[x, i, j] for x in range(sampled_hops.shape[0])])

        tmp[tmp[:,2] == (sampled_hops.shape[1]-1),2] = 0
        index_1 = (triplet[:, 1] != -1) & (triplet[:, 2] != -1)
        triplet = triplet[index_1]
        tmp = tmp[index_1]

        index_2=(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])
        triplet = triplet[index_2]
        tmp = tmp[index_2]

        triplets.append(triplet)
        neigh_index.append(tmp)

        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

    return np.row_stack(triplets), np.concatenate(triplet_scale_terms),\
           np.row_stack(neigh_index).astype(np.float32)



def load_dataset(file_name):
   
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def edge_cover(A):
   

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges


def batch_pairs_sample(A, nodes_hide):
    
    A = A.copy()
    undiricted = (A != A.T).nnz == 0

    if undiricted:
        A = sp.triu(A, 1).tocsr()

    edges = np.column_stack(A.nonzero())
    edges = edges[np.in1d(edges[:, 0], nodes_hide) | np.in1d(edges[:, 1], nodes_hide)]

    # include the missing direction
    if undiricted:
        edges = np.row_stack((edges, np.column_stack((edges[:, 1], edges[:, 0]))))

    # sample the non-edges for each node separately
    arng = np.arange(A.shape[0])
    not_edges = []
    for nh in nodes_hide:
        nn = np.concatenate((A[nh].nonzero()[1], A[:, nh].nonzero()[0]))
        not_nn = np.setdiff1d(arng, nn)

        not_nn = np.random.permutation(not_nn)[:len(nn)]
        not_edges.append(np.column_stack((np.repeat(nh, len(nn)), not_nn)))

    not_edges = np.row_stack(not_edges)

    # include the missing direction
    if undiricted:
        not_edges = np.row_stack((not_edges, np.column_stack((not_edges[:, 1], not_edges[:, 0]))))

    pairs = np.row_stack((edges, not_edges))

    return pairs


def spy_sparse2torch_sparse(data):
    # samples=data.shape[0]
    # features=data.shape[1]
    # values=data.data
    # coo_data=data.tocoo()
    # indices=torch.LongTensor([coo_data.row,coo_data.col])
    # t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    t=torch.from_numpy(data.toarray()).float().cuda()
    return t



if __name__ == '__main__':
    g = load_dataset('./data/cora_ml.npz')
    A, X, z = g['A'], g['X'], g['z']
    hq = DataProcessor(A,X,L=128)
    a= hq.fetch_batch()
    b = a.__next__()
    print(hq.input_dim)
    print(hq.test_ground_truth.shape)
    print(b[0].shape,b[1].shape,b[2].shape)
