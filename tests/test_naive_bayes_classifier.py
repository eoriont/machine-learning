import sys
sys.path.append("src")
try:
    from naive_bayes_classifier import NaiveBayesClassifier
    from dataframe import DataFrame
    from otest import do_assert
except ImportError as e:
    print(e)

df = DataFrame.from_array(
    [
        [False, False, False],
        [True, True, True],
        [True, True, True],
        [False, False, False],
        [False, True, False],
        [True, True, True],
        [True, False, False],
        [False, True, False],
        [True, False, True],
        [False, True, False]
    ],
    ['errors', 'links', 'scam']
)
naive_bayes = NaiveBayesClassifier(df, dependent_variable='scam')

do_assert("probability", naive_bayes.probability('scam', True),
          0.4)
do_assert("probability 2", naive_bayes.probability('scam', False),
          0.6)

do_assert("conditional probability", naive_bayes.conditional_probability(('errors', True), given=('scam', True)),
          1.0)
do_assert("conditional probability 2", naive_bayes.conditional_probability(('links', False), given=('scam', True)),
          0.25)

do_assert("conditional_probability 3", naive_bayes.conditional_probability(('errors', True), given=('scam', False)),
          0.16666666666666666)
do_assert("conditional_probability 4", naive_bayes.conditional_probability(('links', False), given=('scam', False)),
          0.5)

observed_features = {
    'errors': True,
    'links': False
}
do_assert("likelihood", naive_bayes.likelihood(('scam', True), observed_features),
          0.1)
do_assert("likelihood 2", round(naive_bayes.likelihood(('scam', False), observed_features), 5),
          0.05)

do_assert("classify", naive_bayes.classify(observed_features),
          ('scam', True))
