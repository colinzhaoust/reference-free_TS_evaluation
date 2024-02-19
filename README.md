# Towards Reference-free Text Simplification Evaluation with a BERT Siamese Network Architecture
This is the github repo for our ACL 2023 paper "Towards Reference-free Text Simplification Evaluation with a BERT Siamese Network Architecture" [(Link)](https://aclanthology.org/2023.findings-acl.838.pdf).

## Pre-trained Checkpoint
The original checkpoint is deprecated due to cluster shutdown. The author quickly re-trained a small model at this link: [model](https://drive.google.com/file/d/1XvQvTVakLPFMKWbCYGHjDd9vK69Lm9GY/view?usp=sharing)

This model can replicate similar performance on SemEval 2012 and Simplicity-DA reported in the original paper.

Do email me if you need a higher-performance model.

## Usage

The ranker is in **comparative_complexity.py**. The rank takes two tokenized words/phrases as the input and outputs a score from -1 to 1. As said in Table 1 of the paper, it denotes complicating->simplifying.

An example is as follows:

    ranker = torch.load("./bert-base-all.ckpt")
    
    pairs = [["currently", "now"], ["resolve", "solve"], ["phones", "telephones"]]

    for pair in pairs:
        uids = tokenizer.encode(pair[1], add_special_tokens=True, return_tensors='pt').to(device)
        cids = tokenizer.encode(pair[0], add_special_tokens=True, return_tensors='pt').to(device)        

        prediction = ranker(cids, uids)
        print(prediction.cpu().detach().tolist())

The whole pipeline to get BETS is in **metric.py**, which combines the P_simp and R_meaning scores.
  
## Citation

      @inproceedings{zhao-etal-2023-towards,
          title = "Towards Reference-free Text Simplification Evaluation with a {BERT} {S}iamese Network Architecture",
          author = "Zhao, Xinran  and
            Durmus, Esin  and
            Yeung, Dit-Yan",
          editor = "Rogers, Anna  and
            Boyd-Graber, Jordan  and
            Okazaki, Naoaki",
          booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
          month = jul,
          year = "2023",
          address = "Toronto, Canada",
          publisher = "Association for Computational Linguistics",
          url = "https://aclanthology.org/2023.findings-acl.838",
          doi = "10.18653/v1/2023.findings-acl.838",
          pages = "13250--13264",
      }

## Others
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](mailto:xinranz3@andrew.cmu.edu), I will respond to that as soon as possible.

Details about how to set up and run the code will be available soon.

