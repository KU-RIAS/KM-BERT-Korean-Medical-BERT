# KM-BERT-Korean-Medical-BERT
KM-BERT: A Pretrained BERT for Korean Medical Natural Language Processing.<br><br>
KM-BERT has been trained on the Korean medical corpus collected from three types of text (medical textbook, health information news, medical research articles).<br><br>
Please visit the original repo of [BERT] (Devlin, et al.) for more information about the pre-trained BERT.<br>
And, please find the repos of [KR-BERT] (Lee, et al.) and [KoBERT] (SKTBrain) that are a pre-trained BERT for Korean language.<br>

## Pre-trained Model
This repo includes two types of models.

[KM-BERT.tar]: Korean Medical BERT tar file. <br>
[KM-BERT.zip]: Korean Medical BERT zip file. <br>
[KM-BERT-vocab.tar]: Korean Medical BERT with additional medical vocabulary tar file. <br>
[KM-BERT-vocab.zip]: Korean Medical BERT with additional medical vocabulary zip file. <br>

Each file is composed of **(config.json)**, **(vocab.txt)**, and **(model.bin)**.

## Environments

	python 3.6  
	pytorch 1.2.0  
	pytorch-pretrained-bert 0.6.2  

## Usage

Example:  

	python KMBERT_medsts.py --pretrained kmbert		#MedSTS with KM-BERT
	python KMBERT_medsts.py --pretrained kmbert_vocab	#MedSTS with KM-BERT-vocab

	python KMBERT_ner.py --pretrained kmbert		#NER with KM-BERT
	python KMBERT_ner.py --pretrained kmbert_vocab	#NER with KM-BERT-vocab

Arguments:  

	--pretrained		Model
  
	

## Citation

```
@article{KMBERT,
    title={KM-BERT: A Pre-trained BERT for Korean Medical Natural Language Processing},
    author={TBD},
    year={TBD},
    journal={TBD},
    volume={TBD}
  }
```
<br>

[BERT]: https://github.com/google-research/bert
[KR-BERT]: https://github.com/snunlp/KR-BERT
[KoBERT]: https://github.com/SKTBrain/KoBERT
[KM-BERT.tar]: http://www.kurias.co.kr/file/5847315836/kmbert/kmbert.tar
[KM-BERT.zip]: http://www.kurias.co.kr/file/5847315836/kmbert/kmbert.zip
[KM-BERT-vocab.tar]: http://www.kurias.co.kr/file/5847315836/kmbert_vocab/kmbert_vocab.tar
[KM-BERT-vocab.zip]: http://www.kurias.co.kr/file/5847315836/kmbert_vocab/kmbert_vocab.zip
