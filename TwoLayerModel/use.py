import pickle
import numpy as np
from learn import TwoLayerMachine

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

    case_woman = np.array([166, 54, 56, 3.995])
    case_man = np.array([174, 66, 57, 4.552])

    test_case = case_man
    print("Case    : {}".format(test_case))
    print("Predict : {}".format(model.predict(test_case)))