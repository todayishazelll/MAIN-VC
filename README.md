# MAIN-VC

[**MAIN-VC** home page](https://pecholal.github.io/MAIN-VC-demo/).

## Abstract
One-shot voice conversion aims to change the timbre of any source speech to match that of the unseen target speaker with only one speech sample. Existing methods face difficulty in satisfactory speech representation disentanglement and suffer from sizable networks. We propose a method to effectively disentangle with a concise neural network. Our model learns clean speech representations via siamese encoders with the enhancement of the designed mutual information estimator. The siamese structure and the newly designed convolution module contribute to the lightweight of our model while ensuring the performance in diverse voice conversion tasks.

## Usage
### I. Process training data.
`make_dataset.py`->`sample_dataset.py`
Excute the bash `./data/preprocess/preprocess.sh` after modifying the configuration.
**Absolute path is preferred.**

### II. Train MAIN-VC.
The CMI module of MAIN-VC is packaged in `mi.py`. Then all the components are assmebled in `model.py`.
The configuration of the model is in `./config.yaml`. 
Excute the bash `./train.sh` after modifying the configuration. The configuration in the file is our recommended. You can also adjust the size of layers in the network for better performance or less training consuming.
**Absolute path is preferred.**

### III. Inference via trained MAIN-VC.
Any suitably sized (i.e. the bank size of Mel-spectrogram) **pre-trained** vocoder model (eg. WaveRNN, Hifi-GAN, Mel-GAN) can be leveraged as a vocoder for MAIN-VC for Mel-spectrogram to waveform conversion.

Set the path to the check-point file of pre-trained vocoder in `inference.sh` with the argument '-v'.

Set the path to source/target/converted(output) wave file in `inference.sh` then excute it. 


## Citation
If **MAIN-VC** helps your research, please cite it as, 
`coming soon`