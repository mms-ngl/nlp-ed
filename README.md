## Event Detection

Event Detection (ED) project for "Natural Language Processing" course.

### 📝 Project documentation

[**REPORT**](https://github.com/mms-ngl/nlp-ed/blob/main/report.pdf)

### Course Info: http://naviglinlp.blogspot.com/

### 🚀 Project setup

#### Project directory
[[Downloads]](https://drive.google.com/drive/folders/1qwW9aDUaXCJTvl_49fKNgqsXbuchYZsR?usp=sharing) Word embedding, label vocabulary, trained model and token vocabulary: glove-wiki-gigaword-50, label_vocab.pth, model_weights.pth, token_vocab.pth.
```
root
- data
- ed
- logs
- model
 - glove-wiki-gigaword-50
 - label_vocab.pth
 - model_weights.pth
 - token_vocab.pth
 - .placeholder
- Dockerfile
- README
- report
- requirements
- test
```

#### Requirements

* Ubuntu distribution
  * Either 20.04 or the current LTS (22.04).
* Conda 

#### Setup Environment

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

#### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

#### Setup Client

```bash
conda create -n nlp-ed python=3.9
conda activate nlp-ed
pip install -r requirements.txt
```

#### Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp-ed
bash test.sh data/test.jsonl
```
