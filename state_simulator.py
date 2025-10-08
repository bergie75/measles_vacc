import numpy as np

# the RHS of our compartmental model for disease spread, used in simulate_day
# we exclude the R category via conservation laws to save compute
def compartment_rhs(x, t, params, vax_rate, npi_multiplier):
    S,V,E,I = x
    mu, beta, c, gamma, delta = params

    dS_dt = mu*(1-S)-vax_rate*S-beta*npi_multiplier*(c*E+I)*S
    dV_dt = vax_rate*S-mu*V
    dE_dt = beta*npi_multiplier*(c*E+I)*S-(mu+gamma)*E
    dI_dt = gamma*E-(mu+delta)*I

    return np.array([dS_dt, dV_dt, dE_dt, dI_dt])

# this method simulates one day of disease spread, given an initial state and
# an agent's choices for vaccination rate and npi_multiplier, as well as other
# parameters specific to the disease 
def simulate_day(initial_state, params, vax_rate, npi_multiplier):
    pass