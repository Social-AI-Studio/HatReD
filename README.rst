HatReD: Decoding the Underlying Meaning of Multimodal Hateful Memes
====================================================================

A public repository containing **Hat**\ eful memes **Re**\ asoning **D**\ ataset (HatReD) and code implementation for the paper "Decoding the Underlying Meaning of Multimodal Hateful Memes" (IJCAI'23)

.. contents:: Table of Contents 
   :depth: 2

***************
Dataset
***************

The HatReD dataset contains the social target(s) and reasoning(s) annotations for hateful memes, which can be found in the :code:`datasets/hatred/annotations` folder

***********************
Baseline Experiments
***********************

Build HatReD's train and test dataset
-------------------------------------


To reproduce the models used in the experiments, you will need to create the training and testing dataset for HatReD using the original Facebook AI's Hateful Memes dataset. To do so, you can refer to the `README`_ file located in the :code:`datasets`` section.

.. _README: ./datasets/README.md


Encoder Decoder Models 
----------------------

Installation
~~~~~~~~~~~~

**Step 1**. Navigate to encoder decoder model subdirectory

.. code-block:: bash

  cd encoder-decoder-models

**Step 2**. You can find the necessary packages under :code:`requirements.txt`. You can install the packages using the following command:

.. code-block:: bash

  conda create -n ed-models python=3.8 -y
  conda activate ed-models
  pip install --upgrade pip  # enable PEP 660 support
  pip install -r requirements


Training
~~~~~~~~

You can use :code:`main.py` to execute the experiment using different encoder decoder models (i.e. T5, RoBERTa-RoBERTa). For your convenience, we have prepared the respective training scripts for each model settings is provided under :code:`encoder-decoder-models/scripts/train` folder. 

These scripts can also serve as reference point on how you can use the :code:`main.py` file. For instance, you can train the T5 model on HatReD dataset using the following command:

.. code-block:: bash

  bash scripts/train/t5.sh


Testing
~~~~~~~

Similarly, you can find the respective evaluation scripts under :code:`scripts/test` folder. You can evaluate the trained T5 model on HatReD dataset using the following command:

.. code-block:: bash

  bash scripts/test/t5.sh

VL-T5 
-----

Installation
~~~~~~~~~~~~

**Step 1**. Navigate to VL-T5 subdirectory

.. code-block:: bash

  cd VL-T5

**Step 2**. You can find the necessary packages under :code:`requirements.txt`. You can install the packages using the following command:

.. code-block:: bash

  conda create -n vl-t5 python=3.8 -y
  conda activate vl-t5
  pip install --upgrade pip  # enable PEP 660 support
  pip install -r requirements

  
**Step 3**. Download the pretrained model (provided by the VL-T5 authors)

.. code-block:: bash
  gdrive download 1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph --recursive

Training
~~~~~~~~

You can use :code:`src/fhm.py` to execute the experiment. For your convenience, we have prepared the training scripts under :code:`VL-T5/VL-T5/scripts/train` folder. 

The script will also serve as reference point on how you can use the :code:`src/fhm.py` file. For instance, you can train the VL-T5 on HatReD dataset using the following command:

.. code-block:: bash

  cd VL-T5
  bash scripts/train/FHM_VLT5.sh


Testing
~~~~~~~

Similarly, you can find the evaluation script under :code:`VL-T5/VL-T5/scripts/test` folder. You can evaluate the trained VL-T5 model on HatReD dataset using the following command:

.. code-block:: bash

  bash scripts/test/FHM_VLT5.sh


**************************
Citations
**************************

If you find HatReD useful for your your research and applications, please cite the following works using this BibTeX:

.. code-block:: latex

  @inproceedings{hee2023hatred,
    title={Decoding the Underlying Meaning of Multimodal Hateful Memes},
    author={Hee, Ming Shan and Chong, Wen-Haw and Lee, Ka-Wei Roy},
    booktitle={32nd International Joint Conference on Artificial Intelligence (IJCAI 2023)},
    year={2023},
    organization={International Joint Conferences on Artifical Intelligence (IJCAI)}
  }

Additionally, you should also cite the following datasets 

.. code-block:: latex
  
  @article{kiela2020hateful,
    title={The hateful memes challenge: Detecting hate speech in multimodal memes},
    author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
    journal={Advances in Neural Information Processing Systems},
    volume={33},
    pages={2611--2624},
    year={2020}
  }

  @inproceedings{fersini2022semeval,
    title={SemEval-2022 Task 5: Multimedia automatic misogyny identification},
    author={Fersini, Elisabetta and Gasparini, Francesca and Rizzi, Giulia and Saibene, Aurora and Chulvi, Berta and Rosso, Paolo and Lees, Alyssa and Sorensen, Jeffrey},
    booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
    pages={533--549},
    year={2022}
  }