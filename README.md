# GPT-2 for Jet sequences

This repository contains code for tokenized jet generation.

## Data

This project uses the JetClass dataset, where jet constituent information is processed through binning and tokenization. We order the constituents by descending $p_T$ and discretize their features as follows:
- Log transverse momentum: $\log p_T$ with $N_{p_T}=40$ bins.
- Pseudorapidity: $\Delta\eta\in (-0.8, 0.8)$ with $N_{\Delta\eta}=30$.
- Azimuthal angle: $\Delta\phi\in (-0.8, 0.8)$ with $N_{\Delta\phi}=40$.
  
Each jet is thus mapped to a sequence of tokens $\mathbf{J}=(t_1,\dots,t_N,{\tt[stop]})$, where the token $t_i$ represents an individual particle from a vocabulary of size $V=N_{p_T}\cdot N_{\Delta\eta}\cdot N_{\Delta\phi}=36{,}000$. 

## Model Architecture

The autoregressive model implementation in `models.py` is based on the GPT2 architecture from Hugging Face's Transformers library. GPT2 is a large transformer-based language model that has been adapted here for processing jet physics data.
