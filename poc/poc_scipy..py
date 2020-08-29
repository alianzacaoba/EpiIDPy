import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint


# The SIR model differential equations.
def deriv(state, t, n, beta, gamma):
    s, i, r = state
    # Change in S population over time
    dSdt = -beta * s * i / n
    # Change in I population over time
    dIdt = beta * s * i / n - gamma * i
    # Change in R population over time
    dRdt = gamma * i
    return dSdt, dIdt, dRdt


contact_rate = 0.2
recovery_rate = 1 / 14

# We'll compute this for fun
print("R0 is", contact_rate / recovery_rate)

# Everyone not infected or recovered is susceptible
total_pop = 50000000
recovered = 180258
infected = 16619
susceptible = total_pop - infected - recovered

# A list of days, 0-180
days = range(0, 180)

# Use differential equations magic with our population

ret = odeint(deriv, [susceptible, infected, recovered],
             days,
             args=(total_pop, contact_rate, recovery_rate))
s, i, r = ret.T

# Build a dataframe because why not
df = pd.DataFrame({'susceptible': s, 'infected': i, 'recovered': r, 'day': days})

plt.style.use('ggplot')
df.plot(x='day',
        y=['infected', 'susceptible', 'recovered'],
        color=['#bb6424', '#aac6ca', '#cc8ac0'],
        kind='area',
        stacked=True)
plt.show()
print(df.to_string())
