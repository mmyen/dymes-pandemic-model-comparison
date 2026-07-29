import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

class model:
    def __init__(self, c_0, d_0, G = 10):
        """
        Args:
            c_0: Transmission rate of the SIR model.
            d_0: Death rate of the SIR model.
            G: Number of groups in the SIR model.
        """
        self.c_0 = c_0
        self.d_0 = d_0
        self.G = G

        self.t = []
        self.sim_s = []
        self.sim_n = []
        self.sim_r = []

        self.sim_ds_dt = []
        self.sim_dn_dt = []
        self.sim_dr_dt = []
    
    def calculate_derivatives(self, t, states):
        """
        Given the numbers of suceptible, infected, and removed individuals in the population,
        this method calculates the changes in these different groups based on the equations
        in a classic SIR model with constant population.

        Args:
            t: Moment in time. Used below in simulate_sir.
            states: A list with 3 elements, each representing the initial number of
                    suceptible, infected, and removed individuals, in that order.
        """
        s, n, r = states
        ds_dt = -self.c_0 * s * n
        dn_dt = self.c_0 * s * n - self.d_0 * n
        dr_dt = self.d_0 * n
        return [ds_dt, dn_dt, dr_dt]
    
    def simulate_sir(self, s, n, r, time, dt):
        """
        Given the numbers of suceptible, infected, and removed individuals in the population,
        calculates the changes in these different groups based on the equations in a classic
        SIR model. Calculations are done for one group. We assume groups behave homogeneously.

        Also calculates the values of ds_dt, dn_dt, dr_dt at each timestep.

        Args:
            s: The starting number of suceptible individuals in a group.
            n: The starting number of infected individuals in a group.
            r: The starting number of removed individuals in a group.
            time: The length of time over which to simulate the SIR model (in days).
            dt: Timestep to use. Analogous to tau in the DyMES model.
        """
        simulation_sols = solve_ivp(
            self.calculate_derivatives, [0, time], [s, n, r], t_eval=np.arange(0, time, dt)
        )

        self.t = simulation_sols.t
        self.sim_s = simulation_sols.y[0]
        self.sim_n = simulation_sols.y[1]
        self.sim_r = simulation_sols.y[2]

        for val_s, val_n, val_r in zip(self.sim_s, self.sim_n, self.sim_r):
            self.sim_ds_dt.append(-self.c_0 * val_s * val_n)
            self.sim_dn_dt.append(self.c_0 * val_s * val_n - self.d_0 * val_n)
            self.sim_dr_dt.append(self.d_0 * val_n)

    def graph_scaled_infected(self):
        fig, axs = plt.subplots(1, 2, figsize = (14, 6))
        
        scaled_I_vals = [i * self.G for i in self.sim_n]
        axs[0].plot(self.t, scaled_I_vals)
        axs[0].set_title('Total Number Infected (Approx.)')

        scaled_dI_dt = []
        print("First (scaled) number of infected individuals:", scaled_I_vals[0])

        for i in range(len(scaled_I_vals)):
            scaled_S, scaled_I, scaled_R = self.sim_s[i] * self.G, self.sim_n[i] * self.G, self.sim_r[i] * self.G

            dS_approx = -self.c_0 * scaled_S * scaled_I
            dI_approx = self.c_0* scaled_S * scaled_I - self.d_0 * scaled_I
            dR_approx = self.d_0 * scaled_I

            if i == 0:
                print("First (scaled) derivative:", dI_approx)

            scaled_dI_dt.append(dI_approx)
        print("Final (scaled) number of infected individuals:", scaled_I_vals[-1])
        print("Final (scaled) derivative:",scaled_dI_dt[-1])

        axs[1].plot(self.t, scaled_dI_dt)
        axs[1].set_title('Scaled Derivative')

        for ax in axs.flat:
            ax.set(xlabel='Time')

        plt.show()
        
    
    def graph_infected(self):
        """
        Displays graph of number of infected individuals per group vs time, plus derivative of this number vs time.
        """
        fig, axs = plt.subplots(1, 2, figsize = (12, 6))

        axs[0].plot(self.t, self.sim_n)
        axs[0].set_title('State')
        axs[1].plot(self.t, self.sim_dn_dt)
        axs[1].set_title('Derivative')

        for ax in axs.flat:
            ax.set(xlabel='Time')

        plt.show()
    
    def graph_susceptible(self):
        """
        Displays graph of number of susceptible individuals per group vs time, plus derivative of this number vs time.
        """
        fig, axs = plt.subplots(1, 2, figsize = (12, 6))

        axs[0].plot(self.t, self.sim_s)
        axs[0].set_title('State')
        axs[1].plot(self.t, self.sim_ds_dt)
        axs[1].set_title('Derivative')

        for ax in axs.flat:
            ax.set(xlabel='Time')

        plt.show()






