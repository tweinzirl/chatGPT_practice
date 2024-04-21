### Handling Big Models
 - Model size has already exceeds the hosting limits of commodity hardware. Challenge to run the models without access to largescale resources.
 - One solution is to make the models smaller through compression:
    - Pruning: removing model layers that do no have much importance on the output (e.g., as determined by model weights or other metrics [not sure which])
    - Knowlege distillation: use a teacher model to train a smaller (student) model. Enough compute is needed to train the original model and get outputs to pass to the smaller model.
    - Quantization: Represent model weights with a lower precision
      Example: 3x3 matrix going from float32 to int8 reduces memory usage from 36 bytes (4 bytes/value * 9 values) to 9 bytes (1 byte per value)

### Data Types
 - 8-bit unsigned integer (torch.uint8) has range [0, 255] = [0, 2^(n-1)]
 - 8-bit signed integer with 2s-complement representation [-2^(n-1), 2^(n-1)-1]
 - Doing addition on sequences of bits is analogous to base-10 addition, carry the '1' to the left
 - PyTorch has the following datatypes:
   - torch.int8 (signed 8-bit int)
   - torch.uint8 (unsigned 8-bit int)
   - torch.int16 (signed 16-bit int)
   - torch.int32 (signed 32-bit int)
   - torch.int64 (signed 64-bit int)
 - check info of dtype:
   ```python
   import torch
   torch.iinfo(torch.uint8)  # use torch.finfo for floats
   ```
 - floating point numbers have three components:
   - sign (1 bit)
   - exponent: impacts range of number
   - fraction: impacts precision of number
 - FP32, BF16 (brain floating point), FP16, FP8 are floating points with specific allocations for exponent and fraction bits:
   - Example FP32 has 1 bit for sign, 8 for exponent (range), and 23 for fraction (precision)
   - FP32 has the best precision.
   - bfloat16 has bigger range but worse precision than float16

 - Downcasting: Transform variable from a higher to lower datatype. Reduces memory and increases computation speed, but at the price of reduced precision.
  - Use case: mixed-precision training: do computations in smaller precision but store and update weights in higher precision

### Quantization Theory
 - Linear quantization is the most popular approach
 - quanto is a huggingface package for applying quantization
 - linear quantization uses a scale and a zeropoint to e.g., convert float32 to int8
   - in this case, values are clipped to the min/max values of int 8 (-128/127)

### Quantization methods for LLMs
 - It works. Open source examples:
  - LLM.INT8: Dettmers et al. 2022
  - QLoRA (4-bit): Dettmers et al. 2023
  - AWQ: Lin et al. 2023
  - GPTQ: Frantar et al. 2022
  - SmoothQuant: Xiao et al. 2022
  - 2-bit quantization: QuIP# (Tseng et al. 2023), HQQ (Badri et al. 2023), AQLM (Egiazarian et al. 2024)
 - Ready to use methods:
  - Linear quantization (this course)
  - LLM.INT8
  - QLoRA
  - HQQ
 - Example of impact:
  - Llama 2 70B requires 280 GB storage in 32-bit precision, but only 40GB in 4-bit precision
  - Further reduce to 4GB if using 4-bit precision in GGUF format
 - Quantify performance differences in standard benchmarks using HuggingFace Open LLM Leaderboard
 - Finetuning works on quantized models. Requires Quantization Aware Training. Not compatible with Post Training Quantization methods such as linear quantization.
 - Parameters efficient fine tuning (PEFT): significantly reduce the number of trainable parameters while keeping the same performance as full fine tuning (PEFT + QLoRA). See also hugging face [article](https://pytorch.org/blog/finetune-llms)

