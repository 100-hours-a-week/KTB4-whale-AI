from types import SimpleNamespace
from train_perceptron import train_perceptron

def train_batch(inputs, outputs, weights, bias, learning_rate, activate_f):
    """
        Batch Size 만큼 학습하기
    """

    total_weights = weights
    total_bias = bias

    for index in range(len(inputs)):
        perceptron = train_perceptron(index, inputs, outputs, weights, bias, learning_rate, activate_f)

        total_weights += perceptron.new_weights
        total_bias += perceptron.new_bias

    return SimpleNamespace(
        weights=total_weights,
        bias=total_bias
    )