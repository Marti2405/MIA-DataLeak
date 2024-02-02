# MIA-DataLeak

## Abstract
This research explores the vulnerabilities and defenses of neural networks, focusing on the membership inference attack. Our investigation involves implementing a **grey-box membership inference attack** on a **Lenet5 model**, utilizing the **CIFAR-10 dataset**. 

The attack implementation involves strategically sampling a percentage of data from training and testing datasets, creating a demarcation between known and private parts. Employing **kernel density estimation**, we predicted membership by evaluating probabilities of known and private datasets. Moreover, we computed the **Wasserstein distance metric** to analyze what impact has the percentage of data leakage on the success of the attack. 

Furthermore, the research extends its focus to defense mechanisms. For example, the **L2 regularization** defense mechanism effectively increased the uncertainty in the model’s predictions, making it more difficult to confidently infer membership status.
