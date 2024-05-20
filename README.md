# GM_FANTASTX

### Nanocrystal structure search combining genetic algorithm and generative VAE with latent space Bayesian optimization

Traditional genetic algorithms for atomistic structure search are accelerated with generative VAE models. New candidate structures are generated using latent space Bayesian optimization. 

This package is intended to be used with our multi-objective genetic algorithm package, FANTASTX (Fully Automated Nanoscale To Atomistic Structure from Theory and eXperiments).

A nanocrystal VAE which uses graph embeddings for structures creates an organized latent space with an initial dataset curated from FANTASTX calculations. 

Local Latent Bayesian optimization (LOLBO) is performed on the organized latent space from VAE to generate the next nanocrystal structures. 

[Nanocrystal_utils](https://github.com/kvs11/GM_Fantastx/tree/main/lolbo/utils/nanocrystal_utils) includes all the ML models and dataset related features related to the solids or crystalline materials. Whereas dataset curation from VASP outputs can be done simply using the scripts in [accessories](https://github.com/kvs11/GM_Fantastx/tree/main/accessories). 

Lolbo was originally tested on molecular geometries using Selfies representation. Original LOLBO repository [here](https://github.com/nataliemaus/lolbo)

Status: under development

TODO:

1. Integrate FANTASTX -  for a continuous flow of GA and FANTASTX+GenVAE+LOLBO
2. Documentation
3. Add examples with CdTe and Na-P-S systems
