# Jet sequences generation with GPT2

This project uses the tokenized JetClass dataset, where jet constituent information is processed through discretization and tokenization. We order the constituents by descending $p_T$ and voxelize the 3D feature space as follows:
- Log transverse momentum: $\log p_T$ with $N_{p_T}=40$ bins.
- Pseudorapidity: $\Delta\eta\in (-0.8, 0.8)$ with $N_{\Delta\eta}=30$.
- Azimuthal angle: $\Delta\phi\in (-0.8, 0.8)$ with $N_{\Delta\phi}=40$.
  
Each jet is thus mapped to a sequence of discrete tokens $\mathbf{J}=({\tt[start]}, t_1,\dots,t_N,{\tt[stop]})$, where the token $t_i$ represents an individual particle from a vocabulary of size $V=N_{p_T}\cdot N_{\Delta\eta}\cdot N_{\Delta\phi}=36{,}000$. We include at the begining and end of each sequence special [start] and [stop] tokens. 

## Model Architecture 🤗 
The autoregressive model implementation in `models.py` is based on the GPT2 architecture from *Transformers* library froom *Hugging Face* 🤗 . GPT2 is a transformer-based language model that has been adapted here for processing jet physics data.

We train the generative model via next-token-prediction. Including the stop token allows for the model to learn the particle multiplicity distributions. During inference we generate new jet sequences from the start token. 
