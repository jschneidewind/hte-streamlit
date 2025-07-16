import numpy as np
import matplotlib.pyplot as plt


#print("hello I am running")

def kinetic_function(Ru_conc, Ox_conc, k1, k2):

    rate = (k1 * Ru_conc * Ox_conc) / (1 + k2 * Ru_conc**2)

    return rate



if __name__ == "__main__":

    # k1 = 0.1
    # k2 = 0.01

    # Ox_conc = 1000
    # Ru_conc = np.linspace(0, 150, 100)


    k1 = 0.00005
    k2 = 0.03

    Ox_conc = 6000
    Ru_conc = np.linspace(0, 100, 100)

    # Ru conc needs to be >1 to see effect ??? 

    rate = kinetic_function(Ru_conc, Ox_conc, k1, k2)

    plt.plot(Ru_conc, rate)

    plt.show()
