import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    pmf = PMF()
    total = 27753444
    batch_size = 10000
    pmf.set_params({"num_feat": 20, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 100, "num_batches": int(np.ceil(total / batch_size)),
                    "batch_size": batch_size})
    ratings = pd.read_csv("data/ml-latest/ratings.csv")
    ratings = ratings[['userId', 'movieId', 'rating']].values
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.fit(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("result.png")
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
