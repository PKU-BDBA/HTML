# HTML
A Highly Trustworthy Multimodal Learning Method



## Getting Started

To get start with HTML, please follow the instructions below.

### Prerequisites

```
pip install -r requirements.txt
```

### Installation

```
git clone git@github.com:YuxingLu613/HTML.git
```


## Usage

To train the model from stratch with defult settings, you can run the code

```
python main.py
```

If you just want to test the saved model in checkpoints/, you can add argument

```
python main.py -test_only True
```

If you want to change the number of input modalities, you can change the "uni_modality", "dual_modality", "triple_modality" argument

```
python main.py -uni_modality True -dual_modality False triple_modality False
```

If you want to change the hyper-parameters in the model, you can edit the train_test.py.


## Citation

TBA


## Contact

If you have any questions, please feel free to get touch with me, my email is yxlu0613 AT gmail DOT com
