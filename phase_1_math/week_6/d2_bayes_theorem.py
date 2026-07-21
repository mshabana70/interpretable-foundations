import math
import numpy as np
import pandas as pd

def generate_bayes_dataset(num_patients=10000, base_rate=0.01, sensitivity=0.80, specificity=0.90):
    np.random.seed(42)

    # going to assign disease status based on Base Rate
    has_disease = np.random.binomial(1, base_rate, num_patients)

    # gen test results
    test_result = np.zeros(num_patients, dtype=int)
    for i in range(num_patients):
        if has_disease[i] == 1:
            # if sick patient, positive test depends on sensitivity
            test_result[i] = np.random.binomial(1, sensitivity)
        else:
            # healthy patient, positive test depends on false positive rate (1 - specificity)
            test_result[i] = np.random.binomial(1, 1 - specificity)
    
    # creating the dataframe
    df = pd.DataFrame({
        'Patient_ID': range(1, num_patients + 1),
        'True_Disease_Status': has_disease,
        'Test_Result': test_result,
        'Dataset_Base_Rate': base_rate,
        'Dataset_Sensitivity': sensitivity,
        'Dataset_Specificity': specificity
    })
    return df

def bayes(data):
    
    # Let's compute p(x | y) => probability of true disease rate given test result
    x_n = len(data["True_Disease_Status"])
    y_n = len(data["Test_Result"])

    x_n_1 = (data["True_Disease_Status"] == 1).sum()
    x_n_0 = (data["True_Disease_Status"] == 0).sum()
    y_n_1 = (data["Test_Result"] == 1).sum()
    y_n_0 = (data["Test_Result"] == 0).sum()


    p_x_1 = (data["True_Disease_Status"] == 1).sum() / x_n
    p_x_0 = (data["True_Disease_Status"] == 0).sum() / x_n

    print(f"Prob of x = 1: {p_x_1}")
    print(f"Prob of x = 0: {p_x_0}")

    p_y_1 = (data["Test_Result"] == 1).sum() / y_n
    p_y_0 = (data["Test_Result"] == 0).sum() / y_n

    print(f"Prob of y = 1: {p_y_1}")
    print(f"Prob of y = 0: {p_y_0}")

    rows_disease_1 = data[data["True_Disease_Status"] == 1]
    rows_disease_1_test_1 = rows_disease_1[rows_disease_1["Test_Result"] == 1]
    p_y_given_x_1 =  len(rows_disease_1_test_1) / x_n_1
    print(f"P(T = 1 | D = 1) (likelihood) = {p_y_given_x_1}")
    print(f"P(D = 1 | T = 1) (posterior) = {(p_y_given_x_1 * p_x_1) / p_y_1}")

if __name__ == "__main__":
    raw_data = generate_bayes_dataset()

    bayes_data = raw_data[["True_Disease_Status", "Test_Result"]]
    bayes(bayes_data)


