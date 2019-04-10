# Multi-Modal VAE

## A very simple experiment with 3D-Face dataset


### 1) Setup (brief)

- Two-modal paired image data by: <br />
  1) private-1 = elevation (illumination = neutral value fixed) <br />
  2) private-2 = illumination (elevation = neutral value fixed)  <br />
  3) shared = (azimuth, id) <br />
  
- Let: xI = modality 1, xT = modality 2 <br />

### 2) Models (Competing)

#### 2-a) MuMo-VAE model

- partition of latent variables = (zI, zT, zS) <br />
- dim(zI) = 2, dim(zT) = 2, dim(zS) = 5 <br />
- 2 decoders: 1) pI(xI | zI, zS),  2) pT(xT | zT, zS) <br />
- 3 encoders (no parameter sharing): 1) qI(zI, zS | xI),  2) qT(zT, zS | xT),  3) q(zI, zT, zS | xI, xT) <br />
- 3 VAE losses, one for each of {xI}, {xT}, and {(xI,xT)} <br />

<!--
*Latent traversal from (xI,xT) (at iter 80K)<br />
![fixed3](https://user-images.githubusercontent.com/44901665/55332825-0d7e9700-548e-11e9-88a2-7ab8f150345b.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55332885-2129fd80-548e-11e9-9af1-def6d2931b03.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55332858-17a09580-548e-11e9-9864-61014125a9d1.gif)<br />
-->

#### 2-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation

- no partitioning of latent variables <br />
- there is only one encoder model q(z | xI, xT) <br />
- there is only one decoder model p(xI, xT | z) <br />
- dim(z) = 10

<!--
*Latent traversal from (xI,xT) (at iter 80K)<br />
![fixed3](https://user-images.githubusercontent.com/44901665/55333299-e83e5880-548e-11e9-9159-3aa8afd23cca.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55333312-eeccd000-548e-11e9-9300-5dc52994797b.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55333373-1a4fba80-548f-11e9-9817-8ad7850ec5dd.gif)<br />
-->

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

#### 2-e) WG-VAE v1: no private variables; induce q(z | xI, xT) from Product-of-Experts

- Wu-Goodman's multi-modal VAE
- no partition of latent variables, just shared z <br />
- dim(z) = 10 <br />
- 2 decoders: 1) pI(xI | z),  2) pT(xT | z) <br />
- 2 encoders (no parameter sharing): 1) qI(z | xI),  2) qT(z | xT) <br />
- q(z | xI, xT) \propto p(z) * qI(z|xI) * qT(z|xT) <br />
- (why v1?) 1 VAE loss, for {(xI,xT)} <br />

#### 2-f) WG-VAE v2: no private variables; induce q(z | xI, xT) from Product-of-Experts

- The same setup as WG-VAE-v1, but ...
- (why v2?) 3 VAE losses, one for each of {xI}, {xT}, and {(xI,xT)} <br />
- So, the setup is pretty much the same as the original Wu-Goodman's multi-modal VAE

<!--
### R1) Reconstruction: (xI,xT) -> z or (zI,zS,zT) -> (xI',xT')
#### R1-a) MuMo-VAE model
#### R1-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation
#### R1-c) MMPOE-VAE v1
#### R1-d) MMPOE-VAE v2
### R2) Pure synthesis: z or (zI,zS,zT) ~ N(0,I) -> (xI,xT)
#### R2-a) MuMo-VAE model
#### R2-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation
#### R2-c) MMPOE-VAE v1
#### R2-d) MMPOE-VAE v2
### R3) Cross-modal synthesis: Given xI, infer zS, zT ~ N(0,I) -> xT (changing the role of I and T)
#### R3-a) MuMo-VAE model
#### R3-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation
Of course, N/A
#### R3-c) MMPOE-VAE v1
#### R3-d) MMPOE-VAE v2
-->

---

### + Latent traversal: (xI,xT) -> z or (zI,zS,zT), from which traverse along each axis -> (xI',xT')

(at iter 300K) <br />

#### Trv-a) MuMo-VAE model

3 instances, each: <br />
True xI | xI w/ zI(1) change |  xI w/ zI(2) | xI w/ zS(1) | xI w/ zS(2) | ... | xI w/ zT(1) | xI w/ zT(2) <br />
True xT | xT w/ zI(1) change |  xT w/ zI(2) | xT w/ zS(1) | xT w/ zS(2) | ... | xT w/ zT(1) | xT w/ zT(2) <br />

![fixed3](https://user-images.githubusercontent.com/44901665/55629573-6b232400-57ab-11e9-8cef-b84f3a651b9a.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55629559-6494ac80-57ab-11e9-9ab3-4947889314c6.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55629533-53e43680-57ab-11e9-87fd-82af64fe49a6.gif)<br />

(note: quite accurately identify private and shared factors, but computational issue of having dyadic inf net) <br />

#### Trv-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation

3 instances, each: <br />
True xI | xI w/ z(1) change |  xI w/ z(2) | ... | xI w/ z(10) <br />
True xT | xT w/ z(1) change |  xT w/ z(2) | ... | xT w/ z(10) <br />

![fixed3](https://user-images.githubusercontent.com/44901665/55629683-b3424680-57ab-11e9-9aa2-38293cd12790.gif)<br />
![fixed2](https://user-images.githubusercontent.com/44901665/55629680-afaebf80-57ab-11e9-911d-b6ffda29fae3.gif)<br />
![fixed1](https://user-images.githubusercontent.com/44901665/55629640-97d73b80-57ab-11e9-8f76-36f2cc3561c4.gif)<br />

(note: variation of z(4) or z(7), none of them shared factors, results in changes in both xI and xT)<br />

#### Trv-c) MMPOE-VAE v1

3 instances, each: <br />
True xI | xI w/ zI(1) change |  xI w/ zI(2) | xI w/ zS(1) | xI w/ zS(2) | ... | xI w/ zT(1) | xI w/ zT(2) <br />
True xT | xT w/ zI(1) change |  xT w/ zI(2) | xT w/ zS(1) | xT w/ zS(2) | ... | xT w/ zT(1) | xT w/ zT(2) <br />

![fixed3](https://user-images.githubusercontent.com/44901665/55707692-1a464200-59dc-11e9-9dd7-74e4b2689830.gif)
![fixed2](https://user-images.githubusercontent.com/44901665/55707696-1c100580-59dc-11e9-9f5c-acc72ba4bc16.gif)
![fixed1](https://user-images.githubusercontent.com/44901665/55707689-187c7e80-59dc-11e9-8d32-fd81339967f0.gif)

(note: problematic! eg, zS(1) learns elevation factor, but it should be a private factor in zI)<br />

#### Trv-d) MMPOE-VAE v2

3 instances, each: <br />
True xI | xI w/ zI(1) change |  xI w/ zI(2) | xI w/ zS(1) | xI w/ zS(2) | ... | xI w/ zT(1) | xI w/ zT(2) <br />
True xT | xT w/ zI(1) change |  xT w/ zI(2) | xT w/ zS(1) | xT w/ zS(2) | ... | xT w/ zT(1) | xT w/ zT(2) <br />

![fixed3](https://user-images.githubusercontent.com/44901665/55708062-eddef580-59dc-11e9-81bb-5d276bf26f6f.gif)
![fixed2](https://user-images.githubusercontent.com/44901665/55708073-f33c4000-59dc-11e9-932b-37b55004d768.gif)
![fixed1](https://user-images.githubusercontent.com/44901665/55708077-f5060380-59dc-11e9-8f58-bcfe19f5ddbf.gif)

(note: better identify/discern the private and shared factors, which implies that the loss terms for marginal data, ie, {xI} and {xT}, are necessary?)<br />

#### Trv-e) WG-VAE v1

3 instances, each: <br />
True xI | xI w/ z(1) change |  xI w/ z(2) | ... | xI w/ z(10) <br />
True xT | xT w/ z(1) change |  xT w/ z(2) | ... | xT w/ z(10) <br />


![fixed3](https://user-images.githubusercontent.com/44901665/55861783-054de800-5b6f-11e9-81f9-4dde8d71347f.gif)
![fixed2](https://user-images.githubusercontent.com/44901665/55861768-febf7080-5b6e-11e9-9560-1107a7c60450.gif)
![fixed1](https://user-images.githubusercontent.com/44901665/55861753-f830f900-5b6e-11e9-8912-2fba57c4aa5f.gif)

#### Trv-f) WG-VAE v2

3 instances, each: <br />
True xI | xI w/ z(1) change |  xI w/ z(2) | ... | xI w/ z(10) <br />
True xT | xT w/ z(1) change |  xT w/ z(2) | ... | xT w/ z(10) <br />

![fixed3](https://user-images.githubusercontent.com/44901665/55861845-28789780-5b6f-11e9-9154-e5490b6555c6.gif)
![fixed2](https://user-images.githubusercontent.com/44901665/55861850-2b738800-5b6f-11e9-8f85-21d4b8e21e47.gif)
![fixed1](https://user-images.githubusercontent.com/44901665/55861825-21ea2000-5b6f-11e9-90b2-ac322220357c.gif)

---

### + Pure synthesis: z or (zI,zS,zT) ~ N(0,I) -> (xI,xT)

(at iter 300K) <br />

#### PureSynth-a) MuMo-VAE model

[xI, xT] <br />
![synth_pure_300000](https://user-images.githubusercontent.com/44901665/55635682-38802800-57b9-11e9-9719-7b7a650c90c5.jpg)<br />

#### PureSynth-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation

[xI, xT] <br />
![synth_300000](https://user-images.githubusercontent.com/44901665/55635718-5057ac00-57b9-11e9-9019-00d83e47a70c.jpg)<br />

#### PureSynth-c) MMPOE-VAE v1

[xI, xT] <br />
![synth_pure_300000](https://user-images.githubusercontent.com/44901665/55709210-ba519a80-59df-11e9-9bec-fa42ceafc536.jpg)<br />

(note: the quality of generated images is not satisfactory.. especially when compared to the v2 model below)

#### PureSynth-d) MMPOE-VAE v2

[xI, xT] <br />
![synth_pure_300000](https://user-images.githubusercontent.com/44901665/55709220-c2a9d580-59df-11e9-91f6-d2f17400f7fb.jpg)<br />


#### PureSynth-e) WG-VAE v1

[xI, xT] <br />
![synth_pure_300000](https://user-images.githubusercontent.com/44901665/55863589-c15ce200-5b72-11e9-8077-2fa00d09e9cf.jpg)<br />


#### PureSynth-f) WG-VAE v2

[xI, xT] <br />
![synth_pure_300000](https://user-images.githubusercontent.com/44901665/55863609-ccb00d80-5b72-11e9-9283-829964adc48f.jpg)<br />


---

### + Cross-modal synthesis: Given xI, infer zS, zT ~ N(0,I) -> xT (changing the role of I and T)

(at iter 300K) <br />

#### CMSynth--a) MuMo-VAE model

XI -> XT 
[XI | three randomly synthesized XT images]
![synth_cross_modal_I2T_300000](https://user-images.githubusercontent.com/44901665/55636199-45514b80-57ba-11e9-95e4-8c67c57e6491.jpg)

(note: in the synthesized XT images, illumination (private-T) can vary, but elevation (private-I) should be neutral, and (azimuth, id) should be identical to those of XI) <br />

XT -> XI 
[XT | three randomly synthesized XI images]
![synth_cross_modal_T2I_300000](https://user-images.githubusercontent.com/44901665/55636216-4edab380-57ba-11e9-9049-3d945e2d2777.jpg) <br />

(note: in the synthesized XI images, elevation (private-I) can vary, but illumination (private-T) should be neutral, and (azimuth, id) should be identical to those of XT) <br />


#### CMSynth-b) Vanilla VAE regarding (xI,xT) as (concatenated) observation

Of course, N/A <br />

#### CMSynth-c) MMPOE-VAE v1

XI -> XT 
[XI | three randomly synthesized XT images]
![synth_cross_modal_I2T_300000](https://user-images.githubusercontent.com/44901665/55710107-c0e11180-59e1-11e9-8278-bf118dff33c6.jpg) <br />

XT -> XI 
[XT | three randomly synthesized XI images]
![synth_cross_modal_T2I_300000](https://user-images.githubusercontent.com/44901665/55710123-cf2f2d80-59e1-11e9-9bc2-b14b4e2ae684.jpg) <br />

(note: again, v1 suffers from poor quality of synthesized images. It seems to be necessary to take into account the marginal data {xI} and {xT} in the training..)

#### CMSynth-d) MMPOE-VAE v2

XI -> XT 
[XI | three randomly synthesized XT images]
![synth_cross_modal_I2T_300000](https://user-images.githubusercontent.com/44901665/55710148-e0783a00-59e1-11e9-8988-66fe765bb521.jpg) <br />

XT -> XI 
[XT | three randomly synthesized XI images]
![synth_cross_modal_T2I_300000](https://user-images.githubusercontent.com/44901665/55710163-ea01a200-59e1-11e9-9abd-e5b63a06d688.jpg) <br />

#### CMSynth-e) WG-VAE v1

XI -> XT 
[XI | a synthesized XT image]


XT -> XI 
[XT | a synthesized XI image]


#### CMSynth-f) WG-VAE v2

XI -> XT 
[XI | a synthesized XT image]


XT -> XI 
[XT | a synthesized XI image]



