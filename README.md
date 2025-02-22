# A4: Do you Agree

## Student Information

- name: Kaung Sithu
- id: st124974

## Task 1

To train the BERT model in task 1. I used 100K samples from BookCorpus dataset. Originally introduced by:
Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., & Fidler, S. (2015).
"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books".
Proceedings of the IEEE International Conference on Computer Vision (ICCV).

After training I saved the weights for later use in Task 2.

## Task 2

In task 2, I continue training the model saved from Task 1 on another dataset. For this task, another dataset called **Multi-Genre Natural Language Inference (MNLI)** is used, which is a part of the **GLUE benchmark**.

Williams, Adina, Nangia, Nikita, & Bowman, Samuel R. (2018).  
"A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference."  
Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT).  
[https://www.nyu.edu/projects/bowman/multinli/](https://www.nyu.edu/projects/bowman/multinli/)

The dataset bascially consists of:

- mnli_matched: In-domain examples.
- mnli_mismatched: Out-of-domain examples.

The test set in both `mnli_matched` and `mnli_mismatched` does not have gold labels. In MNLI, the test set is unannotated, so label = -1 is used as a placeholder. Therefore, I cannot use the test set for evaluation since it is not labled. I used `validation_matched` for validation and `validation_mismachted` for testing. The code for Task 2 can be found in `code/S-BERT_scratch.ipynb`.

## Task 3

| Model Type | SNLI OR MNLI Performance                               |
|------------|--------------------------------------------------------|
| Our Model  | Accuracy: 0.2900                                        |
|            | Weighted Precision: 0.0841                             |
|            | Weighted Recall: 0.2900                                |
|            | Weighted F1-score: 0.1304                              |
|            | Macro Precision: 0.0967                                |
|            | Macro Recall: 0.3333                                   |
|            | Macro F1-score: 0.1499                                 |
|            | Average Cosine Similarity: 0.9982                      |

The code for evaluation and analysis can be found in `code/S-Bert_task3.ipynb`

limitations, challenges and and proposing potential
improvements or modifications:

- Initial training with only 1 epoch resulted in significant underfitting. The model failed to learn the complex relationships between Premise and Hypothesis. Total Epoches will increased to 4.
- During training, the model was heavily biased towards the Neutral class. This was primarily due to the imbalance in label distribution, where Neutral is the most frequent class. Apply class weights to CrossEntropyLoss to penalize the majority class (Neutral) more heavily to encounter this.
- The model frequently encountered CUDA error: device-side assert triggered, particularly when processing long sequences with high batch sizes. This was primarily due to the large Max Sequence Length (1000 tokens) and the complex custom BERT architecture. Reducing Max Sequence Length from 1000 to 512 or lower would fix this issue.

To transform the model from `BERT-update.ipynb` to `S-BERT_scratch.ipynb`, several key modifications were made to adapt the architecture for sentence-pair tasks in Natural Language Inference (NLI). The primary change involved converting the model to a Sentence-BERT (S-BERT) architecture, which specializes in learning rich semantic representations for sentence pairs. Specifically, a mean pooling layer was added to obtain fixed-size sentence embeddings for both the premise and hypothesis. These embeddings were then concatenated along with their element-wise difference (|u-v|), forming a joint representation that was fed into a 3-class classifier head for NLI (Entailment, Neutral, Contradiction). Additionally, the custom classifier head was re-initialized to better learn decision boundaries for the new concatenated representation, and the model was fine-tuned using cosine similarity to enhance semantic understanding. These changes were made to better capture the relationship between sentence pairs, aligning the architecture more closely with the S-BERT framework.

Hyperparameters:

- Batch Size: 32 (Reduced to 2 during debugging to avoid CUDA memory issues)
- Max Sequence Length: 1000 tokens
- Hidden Dimension (d_model): 768
- Number of Layers (n_layers): 12
- Number of Attention Heads (n_heads): 12
- Feed Forward Dimension (d_ff): 3072 (4 times the hidden dimension)
- Dropout: Not explicitly mentioned but recommended for future improvements
- Learning Rate:
  - BERT Model: `2e-5`
  - Classifier Head: `2e-5` (Later reduced to `1e-5` during fine-tuning)
- Optimizer: Adam with weight decay (from `torch.optim.Adam`)
- Learning Rate Scheduler: Linear Warmup with:
  - Warmup Steps: 10% of total steps
  - Scheduler: `get_linear_schedule_with_warmup`
- Number of Epochs:
  - Initial Training: `1` (Later increased to `5` due to underfitting)
  - Fine-tuning: `2` (After re-initializing the classifier head)

## Task 4

Navigate to the app directory and execute the following command:

```bash
python manage.py runserver
```

The application will be running at <http://localhost:8000/>. Click on Get Started to navigate to prediction page.

<img src='code/figures/webapp.PNG' alt='web application'>
