Also see my [website page](https://www.cmwonderland.com/blog/2018/10/10/102-deepshape-project/)

### Predicting icSHAPE: secondary structure probing data


### methods:
We use window to convert RNA sequence into uniform slices for 1D deep learning model. We also try something **new and fun**: we convert one sequence to a 2D map and use **specialized designed 2D U-net** to predict it. (*Later we find that google use the similar idea to predict SNP as images*).

- 1D
    - CNN
    - RNN
    - ResNet
    - Seq2Seq
    - Attention
- 2D
    - U-net
   
### Discover MOTIF: predict existence and location of motif


### methods:
We revise and improve MEME's EM algorithm to Mixture-PWM to make the model more robust to noises.

We also replace the PWM matrix with a Variational Auto-Encoder (**VAE**).

We then use Graph Convolutional Neural Network (**GCN**) to explore the possibility to predict **Structural related motif**. During our long time exploration, we find GCN may be the best method (in deep learning) to truly understand the structural information in RNA sequence.


