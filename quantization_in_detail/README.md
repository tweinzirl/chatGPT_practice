Three kinds of model compression:
- Quantization: Store model parameters (weights, activations) in a lower precision.
- Knowledge distillation: Train a smaller (Student) model using the original (Instructor) model.
- Pruning: Remove weights from the model.

First course overviewed linear quantization w/ HuggingFace package Quanto.

Linear quanitization diagram

```bash
r_min = -234.1                                  r_max = 251.51
  |_______________________________________________|

   \                                             /
    \                                           /
     \                                         /
      \                                       /
       \                                     /
        |-----------------------------------|
      s_min = -128                        s_max = 127
```

Advantages of Quantization:
 - Smaller model
 - Speed gains (memory bandwidth, faster matrix operations)

Math: r = s * (q – z) or q = int( round( r/s + z ) )

where:
 - r is the original tensor,
 - q is the quantized tensor,
 - s is the scale parameter,
 - z is the zero point parameter.
 
How to determine scale s and zeropoint z:
 - s = (r_max – r_min) / (q_max – q_min)
 - z = int( round( q_min – r_min/s))
 
because:
 - r_min = s * (q_min – z)
 - r_max = s * (q_max – z)
 
If zeropoint is out of range in new scale, set it to q_min or q_max, whichever is nearer

In symmetric quantization, zerpoint is 0: [-r_max, r_max] -> [-q_max, q_max] (simpler and saves memory by not storing a zeropoint)

In Asymmetric quantization: [r_min, r_max] -> [q_min, q_max]

Quantization granularity:
 - Per tensor
 - Per channel (along axis)
 - Per group (n elements together)
 
“Per channel” is typically used when converting to 8-bit

“Per group” can require a lot more memory
