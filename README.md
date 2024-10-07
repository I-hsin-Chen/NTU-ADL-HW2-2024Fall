# ADL HW2

## Environment SetUp
```
conda create -n adl-hw2 python=3.8.10
conda activate adl-hw2
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.2 datasets==2.21.0 accelerate==0.34.2 sentencepiece==0.2.0 evaluate==0.4.3 rouge==1.0.1 spacy==3.7.6 nltk==3.9.1 ckiptagger==0.2.1 tqdm==4.66.5 pandas==2.0.3 jsonlines==4.0.0 protobuf==4.25.5
pip install rouge-score absl-py
```

## Training
```
bash train.sh /path/to/train.jsonl /path/to/public.jsonl
```
* You could adjust or add parameters `--num_beams`, `--top_k`, `--top_p` and `temperature` in `train.sh`
* Model would be save at `--output-dir`

## Inference
1. Download model
```
bash ./download.sh
```
The model would be located at `./model`

2. Run testing data
```
bash ./run.sh /path/to/testing.jsonl /path/to/output.jsonl
```
* The final prediction result would be saved at `./path/to/output.jsonl`.