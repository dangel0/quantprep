import numpy as np
from scipy.stats import norm

def dydx(func, x, d = 1e-5):
    return (func(x+d) - func(x))/d

def newton_solve(
        func,
        guess = 0.0,
        tol = 1e-5,
        max_iter = 10000
):
    """
    Jump in the opposite direction of the derivative until very close to root.
    1) Start by evaluating tangent line func(guess) to get the line's root
    2) 0 = func(guess) + func'(guess)(new_guess - guess)   <=== solve 0 = first order Taylor Expansion
        *  new_guess =  guess - ( func(guess) / func'(guess) ) 
    3) repeat until y < tol
    """
    temp_guess = guess

    for i in range(0,max_iter):
        if abs(func(temp_guess)) < tol:
            print('stinky')
            return temp_guess
        change = dydx(func, temp_guess)
        new_guess = temp_guess - func(temp_guess) / change
        temp_guess = new_guess

    return temp_guess

def black_scholes_call(S_0, K, sigma, T, r=0.05):
    """
    C = e^-rt [F * N(d1) - K * N(d2)] 
    """
    disc = np.exp(-r * T)
    F = S_0 * np.exp(r * T)
    d1 = np.log(F/K) / (sigma * np.sqrt(T)) + (sigma * np.sqrt(T))/2
    d2 = d1 - sigma * np.sqrt(T)

    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
def black_scholes_put(S_0, K, sigma, T, r=0.05):
    """
    P = e^-rt [K * N(-d2) - F * N(-d1)] 
    """
    disc = np.exp(-r * T)
    F = S_0 * np.exp(r * T)
    d1 = np.log(F/K) / (sigma * np.sqrt(T)) + (sigma * np.sqrt(T))/2
    d2 = d1 - sigma * np.sqrt(T)

    return disc * (K * norm.cdf(- d2) - F * norm.cdf(-d1))

def implied_vol(market_price, S_0, K, T, r=0.05):
    """
    In theory market_price - BS_price = 0
    """
    func = lambda sigma : black_scholes_call(S_0=S_0, K=K, T=T, sigma = sigma, r=r) - market_price

    return newton_solve(func, guess=0.15)


if __name__ == "__main__":

    print(black_scholes_call(40,40,.22, 1, 0.04))

    market_price, S, K, T, r = 100, 215, 225, 5,0.05
    iv = implied_vol(market_price, S, K, T, r)
    print(iv)












