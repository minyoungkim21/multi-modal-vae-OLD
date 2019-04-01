# Multi-Modal VAE

## A very simple experiment with 3D-Face dataset


### 1) Setup (brief)
- Two-modal paired image data by: 
  private-1 = elevation, 
  private-2 = illumination azimuth,
  shared = (pose azimuth, id)


### 2) MM-VAE model

xI = modality 1, xT = modality 2
latent vector = (zI, zT, zS)

3 encoders: 1) qI(zI,zS | xI),  2) qT(zT,zS | xT),  3) q(zI, zT, zS | xI, xT)
2 decoders: 1) pI(xI | zI,zS),  2) pT(xT | zT, zS)



### 3) Vanilla VAE regarding (xI,xT) as (concatenated) observation

![fixed3](https://user-images.githubusercontent.com/44901665/55332825-0d7e9700-548e-11e9-88a2-7ab8f150345b.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55332885-2129fd80-548e-11e9-9af1-def6d2931b03.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55332858-17a09580-548e-11e9-9864-61014125a9d1.gif)<br />
