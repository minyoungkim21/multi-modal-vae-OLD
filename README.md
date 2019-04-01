# Multi-Modal VAE

## A very simple experiment with 3D-Face dataset


### 1) Setup (brief)
- Two-modal paired image data by: <br />
  1) private-1 = elevation, <br />
  2) private-2 = illumination azimuth, <br />
  3) shared = (pose azimuth, id) <br />


### 2) MM-VAE model

xI = modality 1, xT = modality 2 <br />
latent vector = (zI, zT, zS) <br />

3 encoders: 1) qI(zI,zS | xI),  2) qT(zT,zS | xT),  3) q(zI, zT, zS | xI, xT) <br />
2 decoders: 1) pI(xI | zI,zS),  2) pT(xT | zT, zS) <br />

(At iter 80K)<br />
![fixed3](https://user-images.githubusercontent.com/44901665/55332825-0d7e9700-548e-11e9-88a2-7ab8f150345b.gif)<br />
----<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55332885-2129fd80-548e-11e9-9af1-def6d2931b03.gif)<br />
----<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55332858-17a09580-548e-11e9-9864-61014125a9d1.gif)<br />


### 3) Vanilla VAE regarding (xI,xT) as (concatenated) observation

(At iter 80K)<br />

![fixed3](https://user-images.githubusercontent.com/44901665/55333299-e83e5880-548e-11e9-9159-3aa8afd23cca.gif)<br />
----<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55333312-eeccd000-548e-11e9-9300-5dc52994797b.gif)<br />
----<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55333373-1a4fba80-548f-11e9-9817-8ad7850ec5dd.gif)<br />

