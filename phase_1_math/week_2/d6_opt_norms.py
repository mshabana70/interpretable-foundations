# have some time today so will try this optional exercise for norms
# should be easy hopefully
import math
import numpy as np

def l1_norm(x):
    return sum([abs(elem) for elem in x])

def l2_norm(x):
    return math.sqrt(sum([elem ** 2 for elem in x]))

def l_inf_norm(x):
    return max([abs(elem) for elem in x])

def dot(vec_1, vec_2):
    vec_len = len(vec_1)
    result = 0
    for i in range(vec_len):
        result += vec_1[i] * vec_2[i]
    return result
def cos_sim(vec_1, vec_2):
    
    vec_1_mag = l2_norm(vec_1)
    vec_2_mag = l2_norm(vec_2)

    return dot(vec_1, vec_2) / (vec_1_mag * vec_2_mag)

def test():

    # some test values

    x1 = [6, 0, 10]
    x2 = [9, 18, 45, 10, 9, 32, 70]
    x3 = [0, 1, 9, 4, 7, 10, 3.00, 9]

    print(f"L1 norm of {x1}: {l1_norm(x1)} == Numpy: {np.linalg.norm(x1, ord=1)}")
    print(f"L1 norm of {x2}: {l1_norm(x2)} == Numpy: {np.linalg.norm(x2, ord=1)}")
    print(f"L1 norm of {x3}: {l1_norm(x3)} == Numpy: {np.linalg.norm(x3, ord=1)}")

    print(f"L2 norm of {x1}: {l2_norm(x1)} == Numpy: {np.linalg.norm(x1, ord=2)}")
    print(f"L2 norm of {x2}: {l2_norm(x2)} == Numpy: {np.linalg.norm(x2, ord=2)}")
    print(f"L2 norm of {x3}: {l2_norm(x3)} == Numpy: {np.linalg.norm(x3, ord=2)}")

    print(f"L_inf norm of {x1}: {l_inf_norm(x1)} == Numpy: {np.linalg.norm(x1, ord=np.inf)}")
    print(f"L_inf norm of {x2}: {l_inf_norm(x2)} == Numpy: {np.linalg.norm(x2, ord=np.inf)}")
    print(f"L_inf norm of {x3}: {l_inf_norm(x3)} == Numpy: {np.linalg.norm(x3, ord=np.inf)}")

    # test cosine sim
    vec_1 = [1, 2, 3, 4]
    vec_2 = [5, 8, 9, 0]

    sim = cos_sim(vec_1, vec_2)
    print(f"Similarity between {vec_1} and {vec_2}: {sim}")

if __name__ == "__main__":
    test()
