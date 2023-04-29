from numpy import exp, log, sqrt, pi
from scipy import special


# brownian motion PDF

def gbmpdf(x: float, Price: float, Time: float, mu: float, sigma: float) -> float:
    assert Price >= 0, 'price must be non-negative'
    assert Time >= 0, 'time must be non-negative'
    assert sigma > 0, 'volatility must be positive'

    exponent = -(log(x / Price) - (mu - sigma ** 2 / 2)
                 * Time) ** 2 / (2 * Time * sigma ** 2)

    pdf = exp(exponent) / (sqrt(2 * pi * Time) * x * sigma)

    return pdf


# Brownian motion CDF

def gbmcdf(x: float, Price: float, Time: float, mu: float, sigma: float) -> float:
    assert Price >= 0, 'price must be non-negative'
    assert Time >= 0, 'time must be non-negative'
    assert sigma > 0, 'volatility must be positive'

    foo = (log(Price / x) + (mu - sigma ** 2 / 2)
           * Time) / (sqrt(2 * Time) * sigma)

    cdf = special.erfc(foo) / 2
    return cdf
