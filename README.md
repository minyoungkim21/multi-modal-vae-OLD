# Multi-Modal VAE

## A very simple experiment with 3D-Face dataset


### 1) Setup (brief)

- Two-modal paired image data by: <br />
  1) private-1 = elevation, <br />
  2) private-2 = illumination azimuth, <br />
  3) shared = (pose azimuth, id) <br />
  
- Let: xI = modality 1, xT = modality 2 <br />

### 2) Models (Competing)

#### 2-a) MuMo-VAE model

- partition of latent variables = (zI, zT, zS) <br />
- dim(zI) = 2, dim(zT) = 2, dim(zS) = 5 <br />
- 2 decoders: 1) pI(xI | zI, zS),  2) pT(xT | zT, zS) <br />
- 3 encoders (no parameter sharing): 1) qI(zI, zS | xI),  2) qT(zT, zS | xT),  3) q(zI, zT, zS | xI, xT) <br />
- 3 VAE losses, one for each of {xI}, {xT}, and {(xI,xT)} <br />

*Latent traversal from (xI,xT) (at iter 80K)<br />
![fixed3](https://user-images.githubusercontent.com/44901665/55332825-0d7e9700-548e-11e9-88a2-7ab8f150345b.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55332885-2129fd80-548e-11e9-9af1-def6d2931b03.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55332858-17a09580-548e-11e9-9864-61014125a9d1.gif)<br />


#### 2-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation

- no partitioning of latent variables <br />
- there is only one encoder model q(z | xI, xT) <br />
- there is only one decoder model p(xI, xT | z) <br />
- dim(z) = 10

*Latent traversal from (xI,xT) (at iter 80K)<br />
![fixed3](https://user-images.githubusercontent.com/44901665/55333299-e83e5880-548e-11e9-9159-3aa8afd23cca.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55333312-eeccd000-548e-11e9-9300-5dc52994797b.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55333373-1a4fba80-548f-11e9-9817-8ad7850ec5dd.gif)<br />


#### 2-c) MMPOE-VAE v1: induce q(zI, zT, zS | xI, xT) from Product-of-Experts

- partition of latent variables = (zI, zT, zS) <br />
- dim(zI) = 2, dim(zT) = 2, dim(zS) = 5 <br />
- 2 decoders: 1) pI(xI | zI, zS),  2) pT(xT | zT, zS) <br />
- 2 encoders (no parameter sharing): 1) qI(zI, zS | xI),  2) qT(zT, zS | xT) <br />
- q(zI, zT, zS | xI, xT) = qI(zI|xI) * qT(zT|xT) * q(zS | xI, xT), 
    where q(zS | xI, xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT) <br />
- (why v1?) 1 VAE loss, for {(xI,xT)} <br />

#### 2-d) MMPOE-VAE v2: induce q(zI, zT, zS | xI, xT) from Product-of-Experts

- The same setup as MMPOE-VAE-v1, but ...
- (why v2?) 3 VAE losses, one for each of {xI}, {xT}, and {(xI,xT)} <br />




### 3) Results 






