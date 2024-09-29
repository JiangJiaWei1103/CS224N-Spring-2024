# CS224N-Spring-2024

[[course](https://web.stanford.edu/class/cs224n/index.html)]

> Note that this repo is still a work in progress...

In this repository, I share my solutions and additional resources for CS224N in the spring of 2024. While there are already some excellent repos offering solutions for various semesters, I found that none fully cover the latest version of the course. Moreover, many of these solutions seem to be copied from past versions, which may not reflect the latest updates. Hence, I decide to document my own learning process and provide all the solutions. If you spot any mistakes, feel free to file an issue!

## Offline Schedule
> Just follow your own path.

### Wed Sep 4
* Watch [Lecture 1 - Intro & Word Vectors](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=1) and [Lecture 2 - Neural Classifiers](https://www.youtube.com/watch?v=gqaHkPEZAew&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=2).

### Thu Sep 5
* Watch [Lecture 3 - Backprop and Neural Networks](https://www.youtube.com/watch?v=X0Jw4kgaFlg&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=3).
* Add [assignment 1](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/blob/main/assignments/a1/exploring_word_vectors.ipynb) part1 draft.

### Fri Sep 6
* Finish [assignment 1](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/blob/main/assignments/a1/exploring_word_vectors.ipynb).
* Watch [Lecture 4 - Syntactic Structure and Dependency Parsing](https://www.youtube.com/watch?v=PSGIodTN3KE&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=5).

### Wed Sep 11
* Watch [Lecture 5 - Recurrent Neural networks (RNNs)](https://www.youtube.com/watch?v=PLryWeHPcBs&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=5) and [Lecture 6 - Simple and LSTM RNNs](https://www.youtube.com/watch?v=0LixFSa7yts&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=6).
* Read [cs224n-2019-notes04-dependencyparsing](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/blob/main/notes/cs224n-2019-notes04-dependencyparsing.pdf).
* Add [assignment 2](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a2) coding part and some written.
    * Train a neural dependency parser, which achieves UAS 88.47 on the dev set and 89.18 on the test set. 

### Thu Sep 12
* Watch [Lecture 7 - Translation, Seq2Seq, Attention](https://www.youtube.com/watch?v=wzfWHP6SXxY&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=7), [Lecture 8 - Self-Attention and Transformers](https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=8), and [Lecture 9 - Pretraining](https://www.youtube.com/watch?v=DGfCRXuNA2w&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=9).
* (Optional) Read [An Analysis on the Learning Rules of the Skip-Gram Model](https://arxiv.org/abs/2003.08489) to section III.
* Add [assignment 2](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a2) written to Q1 (c).

### Fri Sep 13
* Read [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf).
    * Only focus on the motivation and effect of dropout.
* Finish [assignment 2](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a2).

### Sun Sep 15
* Watch [Lecture 10 - Prompting, Reinforcement Learning from Human Feedback](https://www.youtube.com/watch?v=SXpJ9EmG3s4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=10).

### Mon Sep 16
* Do [assignment 3](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a3) coding part.
    * Temporarily skip code snippets related to beam search. 
    * Train a NMT model, which achieves BLEU 22.33 on the test set.

### Tue Sep 17
* Finish [assignment 3](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a3). 

### Wed Sep 18
* Read chapters 1 and 2 of [Natural Language Processing with Transformers](https://transformersbook.com/).

### Thu Sep 19
* Read chapters 3 to 5 of [Natural Language Processing with Transformers](https://transformersbook.com/).

### Fri Sep 20
> I switch back to the course schedule in the spring of 2023, because there's no lectures updated for 2024. However, all solutions still follow the 2024 version.
* Watch [Lecture11 - Natural Language Generation](https://www.youtube.com/watch?v=N9L32bFieEY&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=11).
    * I think it's more consistent to take this lecture before [Lecture 10 - Prompting, Reinforcement Learning from Human Feedback](https://www.youtube.com/watch?v=SXpJ9EmG3s4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=10).
* Read chapter 6 of [Natural Language Processing with Transformers](https://transformersbook.com/).

### Sat Sep 21
* Add [assignment 4](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a4) written to Q1 (c) i and Q2.

### Wed Sep 25
* Recently play around with a [RAG application](https://github.com/JiangJiaWei1103/Hack-RAG/tree/main).
   * As my custom final project?

### Fri Sep 28
* Read chapter 7 of [Natural Language Processing with Transformers](https://transformersbook.com/).
    * Haven't run the corresponding notebook due to the problem of launching Elasticsearch locally.
* Watch [Lecture12 - Question Answering](https://www.youtube.com/watch?v=NcqfHa0_YmU&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=13)
* Read [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
    * Have to reread it to grasp the main points.

### Sat Sep 29
* Add [assignment 4](https://github.com/JiangJiaWei1103/CS224N-Spring-2024/tree/main/assignments/a4) written and coding parts.
    * Pre-train and fine-tune a simple GPT which achieves accuracy of 29.00 of the dev set in Q3 (g).
    * I will finish Q1 (c) to (e) after reviewing probability theory.
