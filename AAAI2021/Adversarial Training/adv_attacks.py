from art.attacks.evasion import ElasticNet
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGDAttack
from art.estimators.classification import PyTorchClassifier


def EAD_L1(classifier, x_test):
    attack = ElasticNet(classifier=classifier,
                        learning_rate=1e-2,
                        binary_search_steps=9,
                        max_iter=10,
                        beta=1e-3,
                        initial_const=1e-3,
                        decision_rule="L1")
    x_test_adv = attack.generate(x=x_test)
    return x_test_adv


def PGD(classifier, x_test, norm, eps, a):
    attack = PGDAttack(estimator=classifier,
                       norm=norm,
                       eps=eps,
                       eps_step=a,
                       max_iter=100,
                       verbose=False)
    x_test_adv = attack.generate(x=x_test)
    return x_test_adv