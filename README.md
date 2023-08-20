# Emory Health AI Bias Datathon

Welcome to the Emory Health AI Bias Datathon!

**Overview:**<br/>
Please find your server assignments by team below. The schedule for the weekend is available [on our website](https://emory.healthdatathon.com). 

.yml files have been provided to set up GPU accelerated environments for either Tensorflow or Pytorch. You can test whether your environment is correctly configured with:

**To install the environment and add the kernel to Jupyterhub**
```
conda env create -f <env-config-file>.yml
conda activate <env-name>
python -m ipykernel install --name=<env-name> --user
```
After running the above commands refresh your browser window.  
*Note: Without the --user flag this command will fail*

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
| 1 + 9 | MRKR #1 + #2 | jhub9.datathon.org, jhub8.datathon.org | 188 | Validating the ALP score for Osteoarthritis assessment. |
| 2 | EMBED #1 | jhub.datathon.org | 251 | Detecting and mitigating bias in medical imaging algorithms. |
| 3 | EMBED #2 | jhub2.datathon.org | 253 | Detecting and mitigating bias in medical imaging algorithms. |
| 4 + 11 | Various Datasets #1 + #2 | jhub4.datathon.org, jhub10.datathon.org | 325 | Are new forms of dataset good enough for AI model development ? What are their impacts on bias? |
| 5 + 8 + 12 | ICU #1 + #2 + #3 | jhub1.datathon.org, jhub7.datathon.org, jhub11.datathon.org | 353 | Evaluate biases in the Emory ICU dataset and disparities proxies. |
| 6 | CXR #1 | jhub5.datathon.org | 269 | Detecting and mitigating bias in medical imaging algorithms. |
| 7 | CXR #2 | jhub6.datathon.org | 268 | Detecting and mitigating bias in medical imaging algorithms. |
| 10 | EMBED #3 | jhub3.datathon.org | 255 | Detecting and mitigating bias in medical imaging algorithms. |
| 13 | ChatGPT | jhub12.datathon.org | 130 | Red Teaming to test the performance of large language models (ChatGPT for health). |

<br/>**Room Assignment Maps:**<br/>
[Floor 1 Room Assignments Sunday Revised.pdf](https://github.com/Emory-HITI/datathon/files/12388258/Floor.1.Room.Assignments.Sunday.Revised.pdf)
[Second Floor Assignments Sunday Revised.pdf](https://github.com/Emory-HITI/datathon/files/12388259/Second.Floor.Assignments.Sunday.Revised.pdf)
[Third Floor Assignments Sunday Revised.pdf](https://github.com/Emory-HITI/datathon/files/12388260/Third.Floor.Assignments.Sunday.Revised.pdf)
