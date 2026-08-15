from semrag.reasoning.prompting.token_estimator import HeuristicTokenEstimator


def test_empty_string_is_zero_tokens():
    assert HeuristicTokenEstimator().estimate("") == 0


def test_longer_text_estimates_more_tokens():
    estimator = HeuristicTokenEstimator()
    short = estimator.estimate("hello")
    long = estimator.estimate("hello " * 100)
    assert long > short


def test_rounds_up_not_down():
    # 5 chars / 4 chars-per-token = 1.25 -> must round up to 2, not
    # truncate down to 1 (under-budgeting is the worse failure mode).
    estimator = HeuristicTokenEstimator(chars_per_token=4.0)
    assert estimator.estimate("abcde") == 2


def test_custom_chars_per_token_ratio():
    estimator = HeuristicTokenEstimator(chars_per_token=2.0)
    assert estimator.estimate("abcd") == 2


def test_minimum_one_token_for_nonempty_text():
    estimator = HeuristicTokenEstimator(chars_per_token=100.0)
    assert estimator.estimate("x") == 1
