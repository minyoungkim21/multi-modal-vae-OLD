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



![fixed_ellipse](https://user-images.githubusercontent.com/44901665/48269786-6a59b200-e406-11e8-9d45-33e3d725e2dd.gif) <br />
![fixed_heart](https://user-images.githubusercontent.com/44901665/48269792-6cbc0c00-e406-11e8-824b-74c07c7eda7b.gif) <br />
![fixed_square](https://user-images.githubusercontent.com/44901665/48269795-6f1e6600-e406-11e8-9ff6-e6db5b9eb256.gif) <br />
![random_img](https://user-images.githubusercontent.com/44901665/48269797-70e82980-e406-11e8-8477-920e8caf136e.gif) <br />
