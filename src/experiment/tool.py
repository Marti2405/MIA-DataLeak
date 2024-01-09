import sys
import argparse
import keras
import numpy as np




# sys.path.append("..")
# from sampling.Sampler import Sampler

pic_class = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = pic_class.load_data()


# Random percentage sampler

def random_percentage_sample(x_data, y_data, sample_percentage=0.05):
    # Check if x_data and y_data have the same length
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    num_samples = int(len(x_data) * sample_percentage)

    # Generate random indices for sampling
    random_indices = np.random.choice(len(x_data), num_samples, replace=False)

    # Extract the sampled data and labels based on the random indices
    sampled_x_data = x_data[random_indices]
    sampled_y_data = y_data[random_indices]

    return sampled_x_data, sampled_y_data




def Evaluate(percentage):

    # # get known training dataset and private dataset.
    know_dts, _ =  random_percentage_sample(x_train, y_train, percentage)
    private_dts, _ = random_percentage_sample(x_test, y_test, percentage)


    # perform inference and compute the gaussians
    known_mean, known_std = 0,0
    private_mean, private_std = 0,0


    # Evaluation Dataset (Change the size as convenience)
    size_percentage = 0.05
    eval_known ,_ = random_percentage_sample(x_train, y_train, size_percentage)
    eval_private, _ = random_percentage_sample(x_test, y_test, size_percentage)

    # predict function returns 1 if known 0 if unkwnown
    def predict(data): # waitin for the function
        return 0
    

    # Compute True positive....
    tp, fp, tn, fn = 0,0,0,0
    
    for image in eval_known:
        p = predict(image)

        if p==0:
            fn+=1
        else:
            tp+=1
    
    for image in eval_private:
        p = predict(image)

        if p==0:
            tn+=1
        else:
            fp+=1

    # Compute True positive rate...
    fpr = fp / (fp + tn) * 100
    tpr = tp / (tp + fn) * 100
    fnr = fn / (fn + tp) * 100
    tnr = tn / (tn + fp) * 100
    
    return tpr, fpr, tnr, fnr





    



