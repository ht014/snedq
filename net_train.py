#--coding=utf-8
import os


import lib
import numpy as np

from lib.utils_n import *
import torch.nn.functional as F
from lib.nn_utils import *
import math
from lib.predict import predict_cv
import torch
import torch.nn as nn
from lib.model import Model
from qhoptim.pyt import QHAdam
from lib.quantizer import compute_penalties
from lib.nn_utils import OneCycleSchedule, DISTANCES, training_mode, gumbel_softmax
from tensorboardX import SummaryWriter

class HierarchQuantizer(nn.Module):

    def __init__(self,data_processor,bottleneck_dim=128,num_codebooks=16,hidden_dim=512,decoder_layers=2,encoder_layers=2,**kwargs):
        super().__init__()
        self.data_processor = data_processor
        self.encoder1 = nn.Sequential(
            Feedforward(self.data_processor.input_dim, hidden_dim, num_layers=encoder_layers, **kwargs),
            nn.Linear(hidden_dim,  bottleneck_dim)
        )

        self.quntizer = Model( input_dim = bottleneck_dim, hidden_dim = 1024, bottleneck_dim = 256,
        encoder_layers = 2, decoder_layers = 2, Activation = nn.ReLU,
        num_codebooks = 8, codebook_size = 256, initial_entropy = 3.0,
        share_codewords = True).cuda()
        self.distance =  DISTANCES['euclidian_squared']
        self.triplet_delta = 5
        all_parameters =  list(self.encoder1.parameters())+list(self.quntizer.parameters())
        self.optimizer = OneCycleSchedule(
              QHAdam(all_parameters, nus=(0.8, 0.7), betas=(0.95, 0.998)),
              learning_rate_base=1e-3, warmup_steps=10000, decay_rate=0.2)
        self.experiment_path = 'logs'

        self.writer = SummaryWriter(self.experiment_path, comment='Cora')


    def forward(self,x_batch,anc_index,pos_index,neg_index,test=False):
        embeddings =  self.encoder1(x_batch)
        if test:
            return embeddings
        pos_emb = embeddings[pos_index]
        neg_emb = embeddings[neg_index]
        anc_emb = embeddings[anc_index]
        pos_dis = self.distance(anc_emb,pos_emb)
        neg_dis = self.distance(anc_emb,neg_emb)
        triplet_loss = F.relu(self.triplet_delta + pos_dis - neg_dis)
        # triplet_loss = -torch.log(torch.sigmoid(anc_emb*pos_emb))- torch.log(torch.sigmoid(-anc_emb*neg_emb))

        x_reconstructed, activations = self.quntizer.forward(embeddings, return_intermediate_values=True)

        reconstruction_loss = self.distance(embeddings, x_reconstructed).mean()

        penalties = compute_penalties(activations['logits'])

        metrics = dict(reconstruction_loss=reconstruction_loss, **penalties)
        metrics['quantization_loss'] = (reconstruction_loss + penalties['reg']).mean()
        metrics['triplet_loss'] = triplet_loss.mean()
        return embeddings, metrics

def train_on_batch(hq,x,step,ac,pos,neg,prefix='train/'):
    hq.optimizer.zero_grad()
    embedings, metrics = hq(x,ac,pos,neg)
    # ( metrics['triplet_loss']).backward()
    (metrics['quantization_loss'] + metrics['triplet_loss']).backward()
    hq.optimizer.step()
    for metric in metrics:
        hq.writer.add_scalar(prefix + metric, metrics[metric].mean().item(),step)
    return  metrics

def obtain_embeddins(model,X):
    fetch_batch = 100
    l = X.shape[0]
    X = X.toarray()
    model.train(False)
    embeddings = []
    for i in range(l//fetch_batch+1):
        x_batch = torch.from_numpy(X[i*fetch_batch:(i+1)*fetch_batch,:]).float().cuda()
        ems = model(x_batch,None,None,None,test=True)
        ems_np = ems.data.cpu().numpy()
        embeddings.append(ems_np)
    embeddings = np.vstack(embeddings)
    return embeddings


def train_epoch(dataset='cora_ml',max_iters=10000):
    g = load_dataset('./data/'+dataset+'.npz')
    A, X, z = g['A'], g['X'], g['z']
    batch_size =100
    dp = DataProcessor(A, X, L=128,batch_size=batch_size)
    iterator = dp.fetch_batch()
    num_iters = A.shape[0] // batch_size
    hq = HierarchQuantizer(dp,encoder_layers=3).cuda()
    step = 0
    hq.train(True)
    while step < max_iters:
        x_batch, index, neigbor_type = iterator.__next__()

        metrics = train_on_batch(hq,x_batch.cuda(),step,index[:,0],index[:,1],index[:,2])
        step += 1
        if step % 250 == 0:
            print("epoch: %.2f, recon_loss: %.4f, triplet_loss: %.4f\n"
                  %(step/num_iters,metrics['quantization_loss'],metrics['triplet_loss']))
        if step % 1000 ==0:
            eval(hq,X,z,dp)

def eval(model,x,z,dp):
    embeddings=obtain_embeddins(model,x)
    mi_f1, ma_f1 = score_node_classification(embeddings, z, n_repeat=1, norm=True)
    print("Node classification --> ","micra_f1: ",mi_f1, "macra_f1: ",ma_f1,'\n')
    z = np.eye(np.max(z) + 1)[z]
    predict_cv(embeddings, z, train_ratio=0.1, n_splits=10, C=1, random_state=2)
   
if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_epoch(dataset='cora_ml',max_iters=1000000)
