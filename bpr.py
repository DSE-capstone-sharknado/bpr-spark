import random
from tqdm import tqdm
import numpy as np


def _gradient_single_point(user_id, prod_id, prod_id_neg, user_mat, prod_mat, lambda_reg, alpha, item_bias):
    x_uij = item_bias[prod_id] - item_bias[prod_id_neg] + user_mat[user_id].dot(prod_mat[prod_id]) - user_mat[
        user_id].dot(prod_mat[prod_id_neg])

    step_size = np.exp(-x_uij) / (1 + np.exp(-x_uij))

    user_mat[user_id] += alpha * \
                         (step_size * (prod_mat[prod_id] - prod_mat[prod_id_neg]) - lambda_reg * user_mat[user_id])

    prod_mat[prod_id] += alpha * (step_size * user_mat[user_id] - lambda_reg * prod_mat[prod_id])

    prod_mat[prod_id_neg] += alpha * (-1 * step_size * user_mat[user_id] - lambda_reg * prod_mat[prod_id_neg])

    item_bias[prod_id] += alpha * (step_size - lambda_reg * item_bias[prod_id])
    item_bias[prod_id_neg] += alpha * (step_size - lambda_reg * item_bias[prod_id_neg])


def _sample_optimize_partition(ratings, user_mat, prod_mat, item_bias, num_prods, lambda_reg=0.01, alpha=0.1,
                               position=None):
    sampled_ratings = random.sample(list(ratings), 1000)

    for u, i, j in sampled_ratings:
        _gradient_single_point(u, i, j, user_mat, prod_mat, lambda_reg, alpha, item_bias)

    yield user_mat, prod_mat, item_bias


def optimizeMF(ratings, rank, num_iter=10, num_neg_samples=30):
    """ Provides a spark-facing non-ditributed version of BPR

    Args:
    -----
        ratings: an rdd of (user, item) pairs
        num_iter: number of iterations
        num_neg_samples: how many negative samples to take

    Returns:
    --------
        (user_mat, prod_mat)
    """

    ratings_partitioned = ratings.partitionBy(4).persist()

    num_users = ratings_partitioned.map(lambda x: x[0]).max()
    num_prods = ratings_partitioned.map(lambda x: x[1]).max()

    user_mat = np.random.uniform(size=(num_users + 1, rank))
    prod_mat = np.random.uniform(size=(num_prods + 1, rank))
    item_bias = np.random.uniform(size=num_prods + 1)

    for _ in xrange(num_iter):
        result = ratings_partitioned.flatMap(
            lambda x: [x] * num_neg_samples
        ).map(
            lambda x: x[:2] + (np.random.randint(num_prods) + 1,)
        ).mapPartitionsWithIndex(
            lambda ix, ratings: _sample_optimize_partition(
                ratings, user_mat, prod_mat, item_bias, num_prods, position=ix
            )
        ).persist()

        num = float(result.count())

        user_mat, prod_mat, item_bias = result.reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))

        user_mat /= num
        prod_mat /= num
        item_bias /= num

    return (user_mat, prod_mat, item_bias)