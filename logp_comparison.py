import numpy as np
import torch
from datamodule_jetclass import JetSequence
from utils import binnify

import h5py

# jet='train_ZJetsToNuNu_2M'
# folder = 'train_100M'

# with h5py.File(f'/pscratch/sd/d/dfarough/JetClass/{folder}_binned_40_30_30/train_ZJetsToNuNu_10M_binned.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     # digits = arr[:2_000_000]
#     # digits = arr[2_000_000:4_000_000]
#     # digits = arr[4_000_000:6_000_000]
#     # digits = arr[6_000_000:8_000_000]
#     digits = arr[8_000_000:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# bins = binnify(digits, make_continuous=False)
# np.save(f'/pscratch/sd/d/dfarough/JetClass/{folder}_binned_40_30_30/{jet}_binned_only_5.npy', bins.numpy())

# bins = binnify(digits, make_continuous=True)
# np.save(f'/pscratch/sd/d/dfarough/JetClass/{folder}_binned_40_30_30/{jet}_binned_and_smeared_5.npy', bins.numpy())


# jets = JetSequence(max_seq_length=40)

# tokens = np.load('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_tokens.npy')[:, :42]

# digits = jets.seq_to_bins_decoding(tokens)
# bins = binnify(digits, make_continuous=True)
# np.save('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_epicfm/gen_TTBar_seq_epicfm_binned_smeared.npy', bins.numpy())
# print(bins[0])
# print(bins[10])
# print(bins[100])

#################

# load h5 file

import h5py

with h5py.File('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/TTBar_EPiC_FM_bins403030.h5', 'r') as f:
    arr = f['discretized/block0_values']
    digits = arr[:]
    digits = digits.reshape(digits.shape[0], -1, 3)[10_000_000: 10_010_000]

data_binned = []
for i in range(100):
    bins = binnify(digits, make_continuous=True)
    data_binned.append(bins)

bins = torch.stack(data_binned, axis=0)
np.save('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/TTBar_EPiC_FM_bins403030_10Kjets_smeared_100times_test.npy', bins.numpy())
print(bins.shape)
print(bins[0][0], bins[1][0])

#################/pscratch/sd/d/dfarough/tokenized-jets/4d9d92ca5dd3426b9444e91e8860fffc/gen_results_top5K

data_tokens = torch.tensor(np.load('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_1M/gen_TTBar_seq_1M_tokens.npy', 'r'))
data_tokens = data_tokens[:10000]

vocab_size = 41 * 31 * 31
Jets = JetSequence(start_token=vocab_size, 
                    end_token=vocab_size + 1, 
                    pad_token=vocab_size + 2, 
                    max_seq_length=40)

data_tokens = torch.where(data_tokens>=vocab_size, -1 * torch.ones_like(data_tokens), data_tokens)
data_binned = []

for i in range(100):
    tokens = data_tokens.clone()[:, 1:]
    bins = binnify(jets=Jets.seq_to_bins_decoding(tokens), make_continuous=True)
    data_binned.append(bins)

bins = torch.stack(data_binned, axis=0)
np.save('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_1M/gen_TTBar_seq_10K_bin_smeared_100times.npy', bins.numpy())
print(bins.shape)
print(bins[0][0], bins[1][0])


#################


# with h5py.File('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/ZJetsToNuNu_EPiC_FM_bins403030.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# bins = binnify(digits, make_continuous=True)
# np.save('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/ZJetsToNuNu_EPiC_FM_bins403030_smeared.npy', bins.numpy())
# print(bins.shape)
# print(bins[0])

# 
#################

# import h5py

# with h5py.File('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_bins403030.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# vocab_size = 41 * 31 * 31

# jets = JetSequence(data=digits,
#                    start_token=vocab_size, 
#                    end_token=vocab_size + 1, 
#                    pad_token=vocab_size + 2, 
#                    max_seq_length=40)

# tks = jets.map_to_sequence()
# np.save('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_tokens.npy', tks)
# print(tks[0])
# print(tks[10])
# print(tks[100])
# print(tks.shape)


#################


# logp = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/GPT2_logp/ZJetsToNuNu_logp_ZJetsToNuNu_epicfm_10k.npy')
# print(logp.shape)
# print(logp[0:10])

# # plot logp histogram

# import matplotlib.pyplot as plt
# plt.hist(logp, bins=30, density=True)
# plt.xlabel('Log Probability')
# plt.ylabel('Density')
# plt.title('Log Probability Histogram')
# plt.grid()
# plt.savefig('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/GPT2_logp/ZJetsToNuNu_logp_ZJetsToNuNu.png')


#################


# data = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned/ZJetsToNuNu_EPiC_FM_tokens.npy')[11_990_000:12_000_000]
# print(data.shape)
# print((data<39401).sum(axis=1)[:10])



# data_GPT = np.load('/pscratch/sd/d/dfarough/tokenized-jets/4e42e15262a34afb86a11033cc0e93ab/gen_results_1M/gen_TTBar_seq_1M_binned_smeared.npy')
# data_EPICFM = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/TTBar_EPiC_FM_bins403030_smeared.npy')

# print('GPT: ', data_GPT[...,1].min(), data_GPT[...,1].max())
# print('EPICFM: ', data_EPICFM[...,1].min(), data_EPICFM[...,1].max()) 

# data_GPT = np.load('/pscratch/sd/d/dfarough/tokenized-jets/6da52255f20d41b59f361b26a289e6ed/gen_results_1M/gen_ZJetsToNuNu_seq_1M_binned_smeared.npy')
# data_EPICFM = np.load('/pscratch/sd/d/dfarough/JetClass/train_12M_EPiC_FM_binned_40_30_30/ZJetsToNuNu_EPiC_FM_bins403030_smeared.npy')


# print('GPT: ', data_GPT[...,1].min(), data_GPT[...,1].max())
# print('EPICFM: ', data_EPICFM[...,1].min(), data_EPICFM[...,1].max()) 


#############

# import numpy as np
# import torch
# from datamodule_jetclass import JetSequence
# from utils import binnify

# import h5py

# with h5py.File('/pscratch/sd/d/dfarough/FundLims/ForZenodov2/ForZenodov2/qcd/discretized/2M_noseed/samples_samples_noseed_nsamples2000000_trunc_5000_1.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)
# print(digits.shape)

# bins = binnify(digits, make_continuous=False)
# np.save(f'/pscratch/sd/d/dfarough/JetClass/Aachen_GPT2_binned_40_30_30/ZJetsToNuNu_AachenGPT_2M_binned_only.npy', bins.numpy())

# bins = binnify(digits, make_continuous=True)
# np.save(f'/pscratch/sd/d/dfarough/JetClass//Aachen_GPT2_binned_40_30_30/ZJetsToNuNu_AachenGPT_2M_binned_and_smeared.npy', bins.numpy())


# with h5py.File('/pscratch/sd/d/dfarough/FundLims/ForZenodov2/ForZenodov2/top/discretized/2M_noseed/samples_samples_noseed_nsamples2000000_trunc_5000_0.h5', 'r') as f:
#     arr = f['discretized/block0_values']
#     digits = arr[:]
#     digits = digits.reshape(digits.shape[0], -1, 3)

# print(digits.shape)
# bins = binnify(digits, make_continuous=False)
# np.save(f'/pscratch/sd/d/dfarough/JetClass/Aachen_GPT2_binned_40_30_30/TTBar_AachenGPT_2M_binned_only.npy', bins.numpy())

# bins = binnify(digits, make_continuous=True)
# np.save(f'/pscratch/sd/d/dfarough/JetClass//Aachen_GPT2_binned_40_30_30/TTBar_AachenGPT_2M_binned_and_smeared.npy', bins.numpy())


