# Learning of physical-layer communication with quantized feedback. 
This code demonstrates how to perform learning of a communication system over a binary feedback channel. 

## Usage
The code has comprises 6 python notebooks:
* Fiber_Optical_learning_with_quantized_feedback.py: alternating training with quantized feedback
* Fiber_Optical_perfect_feedback.py: alternating training with perfect feedback
* Fiber_Optical_SER_no_quantization.py: compute SER when feedback are preprocessed, while not quantized
* Fiber_Optical_SER_one_bit_quantization.py: compute SER when feedback are preprocessed, and quantized with 1 bit
* Fiber_Optical_SER_vs_bits_flipping.py: compute SER when the quantization are flipped with probability p
* Fiber_SER_vs_quantization_bits.py: Compute SER when n bits are used for quantization

We recommend to start with the first notebook, which will determine a transmitter and a receiver for a optical nonlinear communication channel. The code has the following parameters:
```
M = 16                # number of points in the constellation
P_in_dBm = -5         # transmit power in dBm
gamma = 1.27          # fiber non-linearity parameter (set to zero for an AWGN channel)
L = 2000              # total link length in km
K = 20                # number of segments
P_noise_dBm = -21.3   # noise power per segment in dBm
sigma_pi = np.sqrt(0.0005)  # Variance for Gaussian policy (before scaling with the transmit power)
num_bits = 1          # number of bits used for quantization
```
## Authors
The code was developed by Jinxiang Song, Master Student at Chalmers University of Technology. 
This code is based on the paper 

Jinxiang Song, Bile Peng, Christian Häger, Henk Wymeersch, Anant Sahai, "Learning Physical-Layer Communication
with Quantized Feedback," in *Arxiv*. 

If you plan to use or modify this code, we kindly ask you to cite this paper. 

