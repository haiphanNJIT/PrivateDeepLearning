import numpy as np
import math
import scipy.stats

# UB and LB from:
#   https://gist.github.com/DavidWalz/8538435
#   http://www.statsmodels.org/dev/_modules/statsmodels/stats/proportion.html#proportion_confint
# using the Clopper–Pearson interval:
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
def clopper_pearson_upper_bound(count, nobs, alpha, bonferroni_hyp_n=1):
    return scipy.stats.beta.ppf(1 - alpha/bonferroni_hyp_n, count+1, nobs-count)
def clopper_pearson_lower_bound(count, nobs, alpha, bonferroni_hyp_n=1):
    return scipy.stats.beta.ppf(alpha/bonferroni_hyp_n, count, nobs-count+1)

# Hoeffding's inequality
# https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
def hoeffding_bound(nobs, alpha, bonferroni_hyp_n=1):
    return math.sqrt(math.log(bonferroni_hyp_n/alpha) / (2*nobs))
def hoeffding_upper_bound(tot_sum, nobs, alpha, bonferroni_hyp_n=1):
    bound = hoeffding_bound(nobs, alpha, bonferroni_hyp_n)
    return tot_sum / nobs + bound
def hoeffding_lower_bound(tot_sum, nobs, alpha, bonferroni_hyp_n=1):
    bound = hoeffding_bound(nobs, alpha, bonferroni_hyp_n)
    return tot_sum / nobs - bound

# Empirical Bernstein Bounds and Sample Variance Penalization
# https://arxiv.org/pdf/0907.3740.pdf
# For variance estimate formula equivalence:
# https://en.wikipedia.org/wiki/Variance
def empirical_bernstein_bound(tot_sum, sqr_sum, nobs, alpha, bonferroni_hyp_n=1):
    avg = tot_sum / nobs
    # The var is sometimes tiny and negative, probably due to float roundings.
    var = abs((nobs / (nobs-1)) * (sqr_sum/nobs - avg**2))

    log_proba = math.log(2*bonferroni_hyp_n/alpha)
    return math.sqrt((2 * var * log_proba) / nobs) + 7*log_proba/(3 * (nobs-1))
def empirical_bernstein_upper_bound(tot_sum, sqr_sum, nobs, alpha, bonferroni_hyp_n=1):
    bound = empirical_bernstein_bound(tot_sum, sqr_sum, nobs, alpha, bonferroni_hyp_n)
    return tot_sum / nobs + bound
def empirical_bernstein_lower_bound(tot_sum, sqr_sum, nobs, alpha, bonferroni_hyp_n=1):
    bound = empirical_bernstein_bound(tot_sum, sqr_sum, nobs, alpha, bonferroni_hyp_n)
    return tot_sum / nobs - bound

def _laplace_robustness_size(p_max_lb, p_sec_ub, attack_size, dp_epsilon):
    if p_max_lb <= p_sec_ub:
        # we're not even robust to the measurement error...
        return 0.0

    return attack_size * math.log(p_max_lb/p_sec_ub) / (2 * dp_epsilon)

def _guaussian_mech_mult(delta):
    return math.sqrt(2 * math.log(1.25 / delta))

def _gaussian_robustness_size(p_max_lb, p_sec_ub, attack_size, dp_epsilon, dp_delta):
    if p_max_lb <= p_sec_ub:
        # we're not even robust to the measurement error...
        return 0.0

    max_r = 0.0
    max_r_eps  = None
    max_r_delt = None
    delta_range = list(np.arange(0.001, 0.3, 0.001))
    #  epsilon_range = list(np.arange(0.1, 1.00000001, 0.001))  # we want 1 included
    for delta in delta_range:
        eps_min, eps_max, eps = (0.0, 1.0, 0.5)
        while eps_min < eps and eps_max >= eps:
        #  for eps in epsilon_range:
            l = attack_size *  \
                (eps / dp_epsilon) *  \
                (_guaussian_mech_mult(dp_delta) / _guaussian_mech_mult(delta))
            if p_max_lb >= math.e ** (2 * eps) * p_sec_ub + (1 + math.e ** eps) * delta:
                if l > max_r:
                    max_r = l
                    max_r_eps = eps
                    max_r_delt = delta
                # best eps for this delta may be bigger
                eps_min = eps
                eps = (eps_min + eps_max) / 2.0
            else:
                # eps is too big for delta
                eps_max = eps
                eps = (eps_min + eps_max) / 2.0

            if eps_max - eps_min < 0.001:
                break

    return max_r

def robustness_size_argmax(counts, eta,
                           dp_attack_size, dp_epsilon, dp_delta, dp_mechanism):
    """Robustness size for E(argmax(f(x))) predictions (Bernoulli variable).

    Args:
      counts: the counts of prediction for each label.
      eta: the bounds hold with probability (1 - eta).
      dp_attack_size: DP noise scaled for this attack size.
      dp_epsilon: DP epsilon param.
      dp_delta: DP delta param.
      dp_mechanism: DP mechanism used, in {laplace,gaussian}.

    Returns: The maximum size of attack this prediction is robust against, with
             proba at least (1-eta).
    """
    hyp_n = len(counts)
    count_tot = sum(counts)
    count2, count1 = sorted(counts)[-2:]
    p_max_lb = clopper_pearson_lower_bound(count1, count_tot, 0.05, bonferroni_hyp_n=hyp_n)
    p_sec_ub = clopper_pearson_upper_bound(count2, count_tot, 0.05, bonferroni_hyp_n=hyp_n)

    if   dp_mechanism == 'laplace':
        return _laplace_robustness_size(p_max_lb, p_sec_ub, dp_attack_size, dp_epsilon)
    elif dp_mechanism == 'gaussian':
        return _gaussian_robustness_size(p_max_lb, p_sec_ub, dp_attack_size, dp_epsilon, dp_delta)
    else:
        raise ValueError('Only supports the following DP mechanisms: laplace gaussian.')

def robustness_size_softmax(tot_sum, sqr_sum, counts, eta,
                            dp_attack_size, dp_epsilon, dp_delta, dp_mechanism):
    """Robustness size for E(softmax) predictions.

    Args:
      tot_sum: the sum of all predictions for this label (after softmax).
      sqr_sum: the sum of squares of all predictions for this label (after
               softmax).
      counts: the counts of prediction for each label.
      eta: the bounds hold with probability (1 - eta).
      dp_attack_size: DP noise scaled for this attack size.
      dp_epsilon: DP epsilon param.
      dp_delta: DP delta param.
      dp_mechanism: DP mechanism used, in {laplace,gaussian}.

    Returns: The maximum size of attack this prediction is robust against, with
             proba at least (1-eta).
    """
    n = sum(counts)
    if n == 1:
        return 0

    hyp_n = len(counts)
    i2, i1 = np.argsort(tot_sum)[-2:]

    if 7*math.log(2*hyp_n/eta)/(3*(n-1)) > 0.025:
        # Use Hoeffding bounds as the 1/n term is Bernstein is too big.
        p_max_lb = hoeffding_lower_bound(
                tot_sum[i1], n, eta, bonferroni_hyp_n=hyp_n)
        p_sec_ub = hoeffding_upper_bound(
                tot_sum[i2], n, eta, bonferroni_hyp_n=hyp_n)
    else:
        # Use empirical Bernstein bounds
        p_max_lb = empirical_bernstein_lower_bound(
                tot_sum[i1], sqr_sum[i1], n, eta, bonferroni_hyp_n=hyp_n)
        p_sec_ub = empirical_bernstein_upper_bound(
                tot_sum[i2], sqr_sum[i2], n, eta, bonferroni_hyp_n=hyp_n)

    if   dp_mechanism == 'laplace':
        return _laplace_robustness_size(p_max_lb, p_sec_ub, dp_attack_size, dp_epsilon)
    elif dp_mechanism == 'gaussian':
        return _gaussian_robustness_size(p_max_lb, p_sec_ub, dp_attack_size, dp_epsilon, dp_delta)
    else:
        raise ValueError('Only supports the following DP mechanisms: laplace gaussian.')

