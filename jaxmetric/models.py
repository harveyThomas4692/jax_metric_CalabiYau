from flax import linen as nn
import jax.numpy as jnp

# Define the model
class FuncQuintic(nn.Module):
  def setup(self):
    self.dense1 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense2 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense3 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense4 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense5 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense6 = nn.Dense(features=64,param_dtype=jnp.float64)
    self.dense_end = nn.Dense(features=1,use_bias=False,param_dtype=jnp.float64)

  def __call__(self, x):

    xR = x[0]
    xI = x[1]

    kap = jnp.sum(jnp.square(xR) + jnp.square(xI),axis=-1)

    y = jnp.array( 
      [
        xR[0]*xR[1] + xI[0]*xI[1], xR[0]*xI[1] - xI[0]*xR[1],
        xR[0]*xR[2] + xI[0]*xI[2], xR[0]*xI[2] - xI[0]*xR[2],
        xR[0]*xR[3] + xI[0]*xI[3], xR[0]*xI[3] - xI[0]*xR[3],
        xR[0]*xR[4] + xI[0]*xI[4], xR[0]*xI[4] - xI[0]*xR[4],
        xR[1]*xR[2] + xI[1]*xI[2], xR[1]*xI[2] - xI[1]*xR[2],
        xR[1]*xR[3] + xI[1]*xI[3], xR[1]*xI[3] - xI[1]*xR[3],
        xR[1]*xR[4] + xI[1]*xI[4], xR[1]*xI[4] - xI[1]*xR[4],
        xR[2]*xR[3] + xI[2]*xI[3], xR[2]*xI[3] - xI[2]*xR[3],
        xR[2]*xR[4] + xI[2]*xI[4], xR[2]*xI[4] - xI[2]*xR[4],
        xR[3]*xR[4] + xI[3]*xI[4], xR[3]*xI[4] - xI[3]*xR[4],
        xR[0]*xR[0] + xI[0]*xI[0], xR[1]*xR[1] + xI[1]*xI[1],
        xR[2]*xR[2] + xI[2]*xI[2], xR[3]*xR[3] + xI[3]*xI[3],
        xR[4]*xR[4] + xI[4]*xI[4]
      ])/kap
    
    y = self.dense1(y)
    y = nn.gelu(y)
    y = self.dense2(y)
    y = nn.gelu(y)
    y = self.dense3(y)
    y = nn.gelu(y)
    y = self.dense4(y)
    y = nn.gelu(y)
    y = self.dense5(y)
    y = nn.gelu(y)
    y = self.dense6(y)
    y = nn.gelu(y)
    y = self.dense_end(y)
    return y.squeeze()
  
  
  # Define the model
class ConstFunc(nn.Module):
  def setup(self):
    self.dense1 = nn.Dense(features=1,param_dtype=jnp.float64)

  def __call__(self, x):
    y = jnp.array([1.])
    y = self.dense1(y)
    return y.squeeze()
  
  
  # Define the model
class FuncTQ(nn.Module):
  def setup(self):
    self.dense1 = nn.Dense(features=32,param_dtype=jnp.float64)
    self.dense2 = nn.Dense(features=32,param_dtype=jnp.float64)
    self.dense3 = nn.Dense(features=32,param_dtype=jnp.float64)
    self.dense_end = nn.Dense(features=1,use_bias=False,param_dtype=jnp.float64)

  def __call__(self, x):

    xR = x[0]
    xI = x[1]

    kap = jnp.sqrt(jnp.array([jnp.square(xR[0]) + jnp.square(xI[0])+jnp.square(xR[1]) + jnp.square(xI[1]), 
                              jnp.square(xR[0]) + jnp.square(xI[0])+jnp.square(xR[1]) + jnp.square(xI[1]),
                              jnp.square(xR[2]) + jnp.square(xI[2])+jnp.square(xR[3]) + jnp.square(xI[3]), 
                              jnp.square(xR[2]) + jnp.square(xI[2])+jnp.square(xR[3]) + jnp.square(xI[3]),
                              jnp.square(xR[4]) + jnp.square(xI[4])+jnp.square(xR[5]) + jnp.square(xI[5]), 
                              jnp.square(xR[4]) + jnp.square(xI[4])+jnp.square(xR[5]) + jnp.square(xI[5]), 
                              jnp.square(xR[6]) + jnp.square(xI[6])+jnp.square(xR[7]) + jnp.square(xI[7]), 
                              jnp.square(xR[6]) + jnp.square(xI[6])+jnp.square(xR[7]) + jnp.square(xI[7])]))
    xR = xR/kap
    xI = xI/kap

    y = jnp.array( 
      [
      xR[0]*xR[1] + xI[0]*xI[1], xR[0]*xI[1] - xI[0]*xR[1], xR[0]*xR[0]+xI[0]*xI[0], xR[1]*xR[1]+xI[1]*xI[1],
      xR[2]*xR[3] + xI[2]*xI[3], xR[2]*xI[3] - xI[2]*xR[3], xR[2]*xR[2]+xI[2]*xI[2], xR[3]*xR[3]+xI[3]*xI[3],
      xR[4]*xR[5] + xI[4]*xI[5], xR[4]*xI[5] - xI[4]*xR[5], xR[4]*xR[4]+xI[4]*xI[4], xR[5]*xR[5]+xI[5]*xI[5],
      xR[6]*xR[7] + xI[6]*xI[7], xR[6]*xI[7] - xI[6]*xR[7], xR[6]*xR[6]+xI[6]*xI[6], xR[7]*xR[7]+xI[7]*xI[7]
      ])
    
    y = self.dense1(y)
    y = nn.gelu(y)
    y = self.dense2(y)
    y = nn.gelu(y)
    y = self.dense3(y)
    y = nn.gelu(y)
    y = self.dense_end(y)
    return y.squeeze()