# Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge

Source codes for the baseline models of **[Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge](https://arxiv.org/abs/2112.08619)**, accepted at [AAAI-22](https://aaai.org/Conferences/AAAI-22/).



### Environment Setting
We trained the models under the setting of `python==3.7` and `torch==1.5.0`,  with one RTX8000 GPU. Also, our codes are built on the codes of [huggingface](https://github.com/huggingface/transfer-learning-conv-ai), and we utilized [pytorch-ignite](https://github.com/pytorch/ignite) from pytorch in [`ignite`](https://github.com/pkchat-focus/FoCus/tree/main/ignite) folder.

1. Make a virtual environment
```bash    
conda create -n 37_env python=3.7
```
```bash
conda activate 37_env
```

2. Install `pytorch==1.5.0`
```bash
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```

3. Install the required libraries.
```bash    
pip install -r requirements.txt
```    


### Dataset [**[FoCus dataset v2](https://drive.google.com/file/d/1YmEW12HqjAjlEfZ05g8VLRux8kyUjdcI/view?usp=sharing)**]
This data is the modified version of the original data (which is reported in the paper) after ethical inspection.

| FoCus v2 STATISTICS | Train | Valid |
| --- | --- | --- |
| `# dialogues` | 12,484 | 1,000 |
| `# avg rounds` | 5.63 | 5.64 |
| `# knowledge-only answers` | 37,488 | 3,007 |
| `# persona-knowledge answers` | 32,855 | 2,630 |
| `# landmarks` | 5,152 | 923 |
| `avg len of Human's utterances` | 40.70 | 40.21 |
| `avg len of Machine's utterances` | 138.16 | 138.60 |

You should create directories named **`infer_log_focus`, `train_log_focus`, `test_log_focus`, `models`, `data`** under FoCus folder.

We put train, valid, test files of the dataset in the **`data`** folder. (The test set will be available after March 2022.)

The project directory should follow this directory structure:


    📦FoCus
    ┣ 📂data
    ┃ ┗ 📜train.json
    ┃ ┗ 📜valid.json
    ┣ 📂ignite
    ┣ 📂infer_log_focus
    ┣ 📂models
    ┣ 📂python_tf_idf
    ┣ 📂test_log_focus
    ┣ 📂train_log_focus
    ┣ 📜classification_modules.py
    ┣ 📜data_utils.py
    ┣ 📜evaluate_test.py
    ┣ 📜evaluate_test_ppl.py
    ┣ 📜inference.sh
    ┣ 📜inference_test.py
    ┣ 📜LICENSE
    ┣ 📜README.md
    ┣ 📜requirements.txt
    ┣ 📜test.sh
    ┣ 📜train.sh
    ┣ 📜train_focus.py
    ┗ 📜utils_focus


### Training the models
Uncomment the command lines in the **`train.sh`** file, to start training the model. 

    $ sh train.sh 


### Evaluation
Uncomment the command lines in the **`test.sh`** file, to evaluate the model on the test set. 

    $ sh test.sh


### Inference
Uncomment the command lines in the **`inference.sh`** file, to generate utterances with the trained models.

    $ sh inference.sh


### Join Our Workshop @ [COLING 2022](https://coling2022.org/)
We are going to hold **[the 1st workshop on Customized Chat Grounding Persona and Knowledge](https://sites.google.com/view/persona-knowledge-workshop)** in Octorber 2022.
Stay tuned for our latest updates!

Written by [Yoonna Jang](https://github.com/YOONNAJANG).


(c) 2021 [NCSOFT Corporation](https://kr.ncsoft.com/en/index.do) & [Korea University](http://blp.korea.ac.kr/). All rights reserved.

---

### Настроить установку разных версий питона через apt 
- https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/

### установить питон 3.7 в виртуальное окружение
```bash
sudo apt-get install python3.7-venv
```

```bash
python3.7 -m venv 37_env
```

### Исправление ошибки несуществующей архитектуры под RTX 3060 на pytorch
Hi! I faced a similar problem, but could not use conda to solve it. I managed to set it up without conda. Here are steps:

1. Update your CUDA Toolkit. All steps to do that are listed here 13. You just need to select your architecture, distribution and its version then download and install. Before you do that it is usually a good idea to go to pytorch website and check what is the latest supported CUDA version. Sometimes you may overshoot and install a version that is too new so it is worth checking out.

2. Execute required post-install actions. In my case the only thing that I needed to do is add export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}} to my .bashrc file.

3. Confirm that your CUDA is installed correctly. Run nvcc --version. Do not really on CUDA version that is displayed by nvidia-smi. Sometimes those two commands show different values and nvcc --version is what we care about.

4. Uninstall torch and torchvision from your Python environment - pip uninstall torch torchvision.

5. Once again go pytorch website and select a configuration that works for you. You will get pip install command that is ready to use. In my case it was pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

6. Test is torch installed correctly. Run python3 in terminal, and than:
```bash
>>> import torch
>>>  torch.version.cuda
'11.6'
>>> torch.cuda.get_arch_list()
['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
```

### Скачать конду для установки питона 3.7
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
```
```bash
chmod +x ./Miniconda3-py37_4.12.0-Linux-x86_64.sh
``` 
- **важно** выбрать при установке 
```text
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
```
### активировать базовую среду конды
```bash
source ~/.bashrc
```

### отключить автоматическую активацию среды в conda
```bash
conda config --set auto_activate_base false
```

### активировать среду
```bash
source /home/dimweb/Desktop/deeppavlov/d_env/bin/activate
```