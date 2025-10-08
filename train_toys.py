import pytorch_lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
import torch

from models import JetGPT2Model
from datamodule_jetclass import JetSequenceDataset

##########################################################################
parser = ArgumentParser()
parser.add_argument("--num_nodes", "-N", type=int, default=1)
parser.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
parser.add_argument("--project_name", "-proj", type=str, default='synthetic-jets')
parser.add_argument("--comet_workspace", type=str, default='dfaroughy')
parser.add_argument("--comet_api_key", type=str, default='8ONjCXJ1ogsqG1UxQzKxYn7tz')
parser.add_argument("--data_path", type=str, default='/pscratch/sd/d/dfarough/JetClass')
parser.add_argument("--experiment_id", "-id", type=str, default=None)
parser.add_argument("--checkpoint", "-ckpt", type=str, default='last')
parser.add_argument("--tags", type=str, nargs='*')

parser.add_argument("--jet_type", "-type", type=str, default='Toy')
parser.add_argument("--max_seq_length", "-len", type=int, default=40)
parser.add_argument("--num_bins", "-bins", type=int, nargs=3, default=[31, 31, 31])
parser.add_argument("--log_pt_range", "-pt", type=float, nargs=2, default=[0, 100])
parser.add_argument("--eta_range", "-eta", type=float, nargs=2, default=[-5.1, 5.1])
parser.add_argument("--phi_range", "-phi", type=float, nargs=2, default=[-1.2, 1.2])
parser.add_argument("--batch_size", "-bs", type=int, default=256)

parser.add_argument("--n_emb", type=int, default=256)
parser.add_argument("--n_inner", type=int, default=1024)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--pos_encoding", "-pos", type=bool, default=True)
parser.add_argument("--activation", "-a", type=str, default='gelu_new')
parser.add_argument("--dropout_attention", "-do_att", type=float, default=0.1)
parser.add_argument("--dropout_embedding", "-do_emb",type=float, default=0.1)
parser.add_argument("--dropout_residual", "-do_res", type=float, default=0.1)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_final", type=float, default=0.0001)
parser.add_argument("--max_epochs", "-epochs", type=int, default=50)

config = parser.parse_args()
##########################################################################

logger = CometLogger(
            api_key=config.comet_api_key,
            project_name=config.project_name,
            workspace=config.comet_workspace,
            save_dir=config.dir,
            experiment_key=config.experiment_id if config.experiment_id else None
        )

logger.experiment.add_tags(config.tags)



##############

def make_ref_data(nevents,scale,mean,nconst='fixed'):
    if nconst=='fixed':
        nconst_use=config.max_seq_length
    else:
        nconst_use=np.random.poisson(config.max_seq_length,size=nevents)
        nconst_use[nconst_use>config.max_seq_length]=config.max_seq_length
        
    nconst_mask=np.transpose((np.arange(config.max_seq_length).reshape((-1,1)))<nconst_use)  # shape (config.max_seq_length, nevents)

    if nconst=='fixed':
        print(np.unique(np.sum(~nconst_mask)))
    
    # first: the pTs
    # this gets a uniform sampling of the surface of a 9d hypersphere
    # see https://mathworld.wolfram.com/HyperspherePointPicking.html
    pT=np.random.normal(size=(nevents,config.max_seq_length))*nconst_mask
    pT=pT/np.sqrt(np.sum(pT**2,axis=-1).reshape((-1,1))) 
    # take the positive quadrant
    pT=np.abs(pT)
    # rescale by m sampled from (truncated) gaussian
    m=np.empty((0))
    while(len(m)<len(pT)):
        temptemp=scale*np.random.normal(size=(10000))+mean
        m=np.concatenate((m,temptemp[(temptemp>0)]))
    m=m[:len(pT)]
    pT=pT*(m.reshape((-1,1)))    

    # then the etas
    eta=0.5*np.random.normal(size=(nevents,config.max_seq_length))*(1+pT*0.1)*nconst_mask
    phi=(2*(np.random.uniform(size=(nevents,config.max_seq_length))-0.5))*nconst_mask

    refdata=np.dstack((pT,eta,phi))

    return refdata

sigscale=10
sigmean=100

sigdata_raw=make_ref_data(30000,sigscale,sigmean,nconst='variable')
sigdata_raw=np.dstack((sigdata_raw[:,:,0],
                       sigdata_raw[:,:,1],
                       sigdata_raw[:,:,2]))

idx = np.argsort(sigdata_raw[..., 0])
sigdata_sorted = np.array([sigdata_raw[j][i][::-1] for j, i in enumerate(idx)])

pt_bins=np.linspace(0, 100, config.num_bins[0])
eta_bins=np.linspace(-5.1, 5.1, config.num_bins[1])
phi_bins=np.linspace(-1.2, 1.2, config.num_bins[2])

def discretize_data(
    data: np.array,
    output_file: str,
    nJets=10000,
):

    def Get_features(data):
    
        const_pt=data[...,0].copy()
        d_eta=data[...,1].copy()
        d_phi=data[...,2].copy()
    
        return const_pt,d_eta,d_phi
        
    def discretize():
        const_pt_disc = np.digitize(const_pt, pt_bins).astype(np.int16)
        d_eta_disc = np.digitize(d_eta, eta_bins).astype(np.int16)
        d_phi_disc = np.digitize(d_phi, phi_bins).astype(np.int16)
        const_pt_disc[const_pt == 0] = -1
        d_eta_disc[const_pt == 0] = -1
        d_phi_disc[const_pt == 0] = -1
        return const_pt_disc, d_eta_disc, d_phi_disc

    data = data[:nJets]

    print(f"INFO: Data shape: {data.shape}\n")

    const_pt, d_eta, d_phi = Get_features(data)
    const_pt_disc, d_eta_disc, d_phi_disc = discretize()

    print(f"INFO: pT bin range: {const_pt_disc[const_pt!=0].min()} {const_pt_disc.max()}")
    print(f"INFO: eta bin range: {d_eta_disc[const_pt!=0].min()} {d_eta_disc.max()}")
    print(f"INFO: phi bin range: {d_phi_disc[const_pt!=0].min()} {d_phi_disc.max()}\n")

    return np.dstack((const_pt_disc,d_eta_disc,d_phi_disc))


refdata_binned=torch.tensor(discretize_data(
            data=sigdata_sorted,
            output_file=None,
            nJets= 30000,
        ), dtype=torch.int32)

train_dataset=JetSequenceDataset(input_ids=refdata_binned[:int(0.75*len(refdata_binned))], 
                                 num_bins=[config.num_bins[0],config.num_bins[1],config.num_bins[2]],
                                 max_seq_length=config.max_seq_length)

val_dataset=JetSequenceDataset(input_ids=refdata_binned[int(0.75*len(refdata_binned)):], 
                               num_bins=[config.num_bins[0],config.num_bins[1],config.num_bins[2]],
                               max_seq_length=config.max_seq_length)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

checkpoint_every_epoch = L.callbacks.ModelCheckpoint(
    dirpath=None,
    filename="{epoch:03d}",       
    save_top_k=-1,                   # keep *all* checkpoints
    every_n_epochs=1,                # save after each epoch
    save_on_train_epoch_end=True,    # use training-epoch boundary
)

checkpoint_best_and_last = L.callbacks.ModelCheckpoint(
    dirpath=None,
    filename="best",                      # still keep the best model
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)

trainer = L.Trainer(
    max_epochs=config.max_epochs,
    accelerator='gpu',
    devices='auto',
    strategy='ddp',
    num_nodes=config.num_nodes,
    callbacks=[checkpoint_best_and_last],
    logger=logger,
    gradient_clip_val=1.0,
)


if config.experiment_id is None:

    model = JetGPT2Model(max_seq_length=config.max_seq_length,
                         num_bins=config.num_bins,
                         logpt_range=config.log_pt_range,
                         eta_range=config.eta_range,
                         phi_range=config.phi_range, 
                         n_embd=config.n_emb,
                         n_inner=config.n_inner,
                         n_layer=config.n_layer,
                         n_head=config.n_head,
                         activation=config.activation,
                         dropout_att=config.dropout_attention,
                         dropout_emb=config.dropout_embedding,
                         dropout_res=config.dropout_residual,
                         learning_rate=config.lr,
                         learning_rate_final=config.lr_final,
                         pos_encoding=config.pos_encoding,
                        )
    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
else:

    ckpt = f"{config.dir}/{config.project_name}/{config.experiment_id}/checkpoints/{config.checkpoint}.ckpt"
    model = JetGPT2Model.load_from_checkpoint(ckpt)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt
                )