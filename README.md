# jax_metric_CalabiYau
 
Some python code written in JAX for finding Ricci-flat metrics on Calabi-Yau manifolds.

This is easiest to install using pip

```console
pip install git+https://github.com/harveyThomas4692/jax_metric_CalabiYau.git
```

An example can be found in example.ipynb, which can be used to find the metric on both the tetraquadric and quintic with some pre-generated points. Functions for generating points are also included in the package. The setup is relatively modular, and can also be applied to any codimension-1 CICY, but new neural network architectures will needed to be written. The method is equivalent to the phi-model introduced in cymetric but with projective neural networks that automatically satisfy the transition loss criteria.

For the quintic, the loss saturates within a couple of minutes of training and it can be observed that the Ricci-Scalar is a gaussian around zero (which is to be expected for the correct Ricci-flat metric, up to numerical error).

<div style="display: flex; justify-content: space-around;">
  <img src="Images/fs_scal.png" alt="|Ricci Scalar| for Fubini-Study Metric" width="300">
  <img src="Images/init_scal.png" alt="|Ricci Scalar| for Randomly Initialised Metric" width="300">
  <img src="Images/flat_scal.png" alt="|Ricci Scalar| for Ricci-flat Metric" width="300">
  <img src="Images/loss.png" alt="Monge-Ampere loss vs batch" width="300">
</div>

## Using this for your own research
This was written mostly for my own amusement, as I wanted something, unlike the alternatives, in a functional programming paradigm. My hope being that this would maximise clarity, over speed, for someone who is more familiar with the mathematics than the computational methods. As such, I've used this when introducing mathematicians and physicists to numerical geometry. For research purposes, I suggest using either https://github.com/pythoncymetric/cymetric or https://github.com/Justin-Tan/cymyc which are both more generic.
