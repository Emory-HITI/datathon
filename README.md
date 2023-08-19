# Emory Health AI Bias Datathon

Welcome to the Emory Health AI Bias Datathon!

**Overview:**<br/>
Please find your server assignments by team below. The schedule for the weekend is available [on our website](https://emory.healthdatathon.com). 

.yml files have been provided to set up GPU accelerated environments for either Tensorflow or Pytorch. You can test whether your environment is correctly configured with:

**Tensorflow:**
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**PyTorch:**
```
python -c "import torch; print(torch.cuda.is_available();"
```
<br/>**Server/Room Assignments:**
| **Team** | **Dataset** | **Server** | **Room** | **Project Statement** |
| - | - | - | - | - |
| 1 | MRKR #1 | jhub9.datathon.org | 169 | Validating the ALP score for Osteoarthritis assessment. |
| 2 | EMBED #1 | jhub.datathon.org | 251 | Detecting and mitigating bias in medical imaging algorithms. |
| 3 | EMBED #2 | jhub2.datathon.org | 253 | Detecting and mitigating bias in medical imaging algorithms. |
| 4 | Various Datasets #1 | jhub4.datathon.org | 257 | Are new forms of dataset good enough for AI model development ? What are their impacts on bias? |
| 5 | ICU #1 | jhub1.datathon.org | 303 | Evaluate biases in the Emory ICU dataset and disparities proxies. |
| 6 | CXR #1 | jhub5.datathon.org | 269 | Detecting and mitigating bias in medical imaging algorithms. |
| 7 | CXR #2 | jhub6.datathon.org | 268 | Detecting and mitigating bias in medical imaging algorithms. |
| 8 | ICU #2 | jhub7.datathon.org | 353 | Evaluate biases in the Emory ICU dataset and disparities proxies. |
| 9 | MRKR #2 | jhub8.datathon.org | 188 | Validating the ALP score for Osteoarthritis assessment. |
| 10 | EMBED #3 | jhub3.datathon.org | 255 | Detecting and mitigating bias in medical imaging algorithms. |
| 11 | Various Datasets #2 | jhub10.datathon.org | 325 | Are new forms of dataset good enough for AI model development ? What are their impacts on bias? |
| 12 | ICU #3 | jhub11.datathon.org | 355 | Evaluate biases in the Emory ICU dataset and disparities proxies. |
| 13 | ChatGPT | jhub12.datathon.org | 130 | Red Teaming to test the performance of large language models (ChatGPT for health). |
