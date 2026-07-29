import numpy as np
from scipy.optimize import fsolve
import sys, os
from typing import Callable
import matplotlib.pyplot as plt

import dyMES.rfunctions as rf
import dyMES.default_models as dm


class model:
    def __init__(self, initial_state:float, params:dict = {}, transition_function:Callable = None, num_groups:str = None) -> None:
        """
        Args:
            initial_state: Value of state at t=0
            params: Dict storing all parameters used in transition function
            transition_function: Transition function as described in paper, look at run_model.ipynb for example usage
            num_groups: key to variable in "params" which represents the number of groups in the system
        """

        self.states = [initial_state]
        self.num_dead = [0]
        self.lambdas = [[-0.000001,0]] #Default lambdas used as a starting point to find true values
        self.params = params
        self.time = [0]
        self.num_groups = num_groups
        self.probability_distributions = {}

        if transition_function is not None:
            self.func = transition_function

        #Set default model to the original pandemic toy model
        else:
            print("Using Default Transition Function")
            print("Steady State at N =", initial_state)
            print("Parameters:", dm.pandemic_params)
            self.func = dm.pandemic_transition_function
            self.params = dm.pandemic_params
            self.num_groups = 'S'
            
        if self.num_groups == None:
            self.num_groups = "No Groups"
            self.params[self.num_groups] = 1
        


        #Initialize lambdas, lambda 2 should be 0, so self.derivatives is set to 0
        self.derivatives = [0]
        self.lambdas = [self.lambda_update(init=True)]
        self.lambdas[0][1] = 0 
        

        #Calculate initial derivatives, should be 0 at steady state
        # The initial derivatives and calculations use <n>_(t-1) = 0.
        m_effective = self.params['m_0']
        Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1])
        f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1], lambda n: self.func(n, self.states[-1], self.params))/Z 
        # self.derivatives[-1] = f_mean  # should this be multiplied by the number of groups?
        self.derivatives[-1] = f_mean * self.params[self.num_groups]

        # Initialize m_(t-1) to hold m_0 and <n>_(t-1) to hold <n>_0 after initialization.
        self.params['m_(t-1)'] = self.params['m_0']
        self.m_effs = [self.params['m_0']]
        self.params['<n>_(t-1)'] = self.states[-1]/self.params[self.num_groups]

        self.probability_distributions[0] = rf.R(self.func, self.states[-1], self.params['m_(t-1)'], self.params, self.lambdas[-1])
        self.total_num_timesteps = 0

        self.constraint_errors = []
    
    def only_brute_force_update(self, time = float, dt = 0.1, error_lim=float("inf")) -> None:
        num_timesteps = int(time / dt)

        tenth_day = int(0.1/dt)
        quarter_day = int(0.25/dt)
        hundredth_day = int(0.01/dt)

        # Main Update Loop
        for timestep in range(num_timesteps):
            # print(timestep)
            
            self.time.append(self.time[-1] + dt)
            self.states.append(self.states[-1] + dt * self.derivatives[-1])
            # based off what was discussed with Pranav last year we are assming I = <n> * G
            # and we can do this computation straightforward to calculate the change in the number dead because
            # the term representing deaths is linear
            self.num_dead.append(self.num_dead[-1] + dt * self.params['d_0'] * (self.states[-2]))

            # Keep track of old m_(t-1) value for further computation.
            old_m = self.params['m_(t-1)']
            m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']

            Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1])
    
            # Calculate probabilities after derivative
            # if (self.total_num_timesteps + 1) % (5 * (1/dt)) == 0:

            to_add = ((self.total_num_timesteps + 1) % tenth_day == 0) or ((self.total_num_timesteps + 1) % quarter_day == 0)
            if to_add or (timestep == 0 and len(self.time) > 2): 
              self.probability_distributions[round(self.time[-1], 2)] = rf.R(self.func, self.states[-1], self.params['m_(t-1)'], self.params, self.lambdas[-1])

            # Update derivatives
            # Derivative update should use old mean so we don't update stored <n>_(t-1)
          
            f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1], lambda n: self.func(n, self.states[-1], self.params))/Z 
           
            self.derivatives.append(f_mean * self.params[self.num_groups])

            n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1])/Z

            # Compute new lambdas using old values.
            new_lambda = self.brute_force_update()
            print_toggle = False
            if to_add or timestep == 0:
                print_toggle = True
                print(self.time[-1])

            self.check_constraints(new_lambda, error_lim = error_lim, print_toggle = print_toggle)
                
  
            self.lambdas.append(new_lambda)

            # Update m_(t-1) to be set to m_effective, as this is the next iteration's value of m_(t-1).
            self.params['m_(t-1)'] = m_effective
            self.m_effs.append(m_effective)
            # Then, update <n>_(t-1) to be n_mean, the mean calculated using the old value of m and old lambdas.
            self.params['<n>_(t-1)'] = self.states[-1]/self.params[self.num_groups]
            # self.params['<n>_(t-1)'] = n_mean
            self.params['I_(t-1)'] = self.states[-2]
            self.total_num_timesteps += 1


    def update(self, time:float, dt=0.1, error_lim=float("inf")) -> None:
        """Updates model for "time" duration using timesteps of size "dt"

        Args:
            time: Length of time to perform updates for
            dt: Timestep to use, dt=0.1 seems to be a good value to start off with
            error_lim: Error limit to accept before printing out a warning.
        """

        num_timesteps = int(time/dt)

        tenth_day = int(0.1/dt)
        quarter_day = int(0.25/dt)
        hundredth_day = int(0.01/dt)

        # Main Update Loop
        for timestep in range(num_timesteps):
            # print(timestep)
            
            self.time.append(self.time[-1] + dt)
            self.states.append(self.states[-1] + dt * self.derivatives[-1])
            self.num_dead.append(self.num_dead[-1] + dt * self.params['d_0'] * (self.states[-2]))

            # Keep track of old m_(t-1) value for further computation.
            old_m = self.params['m_(t-1)']
            m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']

            Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1])
    
            # Calculate probabilities after derivative
            # if (self.total_num_timesteps + 1) % (5 * (1/dt)) == 0:

            to_add = ((self.total_num_timesteps + 1) % tenth_day == 0) or ((self.total_num_timesteps + 1) % quarter_day == 0)
            if to_add or (timestep == 0 and len(self.time) > 2): 
              self.probability_distributions[round(self.time[-1], 2)] = rf.R(self.func, self.states[-1], self.params['m_(t-1)'], self.params, self.lambdas[-1])

            # Update derivatives
            # Derivative update should use old mean so we don't update stored <n>_(t-1)
          
            f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1], lambda n: self.func(n, self.states[-1], self.params))/Z 
           
            self.derivatives.append(f_mean * self.params[self.num_groups])

            n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1])/Z

            # Compute new lambdas using old values.
            # If the determinant is too close to zero then use brute force instead
            new_lambda = self.lambda_update()
            determinant_D = self.calculate_determinant(self.lambdas[-1])

            print_toggle = False
            if to_add or timestep == 0:
                print_toggle = True
                print(self.time[-1])
            if (self.total_num_timesteps + 1) % hundredth_day == 0:
              print("Determinant:",determinant_D)
            
            if determinant_D <= 0.05:
              new_lambda = self.brute_force_update()
              # print(new_lambda)

            self.check_constraints(new_lambda, error_lim = error_lim, print_toggle = print_toggle)
  
            self.lambdas.append(new_lambda)

            # Update m_(t-1) to be set to m_effective, as this is the next iteration's value of m_(t-1).
            self.params['m_(t-1)'] = m_effective
            self.m_effs.append(m_effective)
            # Then, update <n>_(t-1) to be n_mean, the mean calculated using the old value of m and old lambdas.
            self.params['<n>_(t-1)'] = self.states[-1]/self.params[self.num_groups]
            # self.params['<n>_(t-1)'] = n_mean
            self.params['I_(t-1)'] = self.states[-2]
            self.total_num_timesteps += 1
    
    def pandemic_dfdI(self, n, I, params):
        # (params['c0'] * params['h'] * n * np.exp(- 1 * I / params['K'])) * (m_effective - n) - (params['d_0'] * n)
        #  m_effective = (params['m_(t-1)'] - params['c1'] * params['tau'] * params['d_0'] * params['<n>_(t-1)'])

        # F(n, I) = (c0 * n * e^-I/K) * ((m_(t-1) - c1 * tau * d_0 * <n>_(t-1)) - n) - d_0 * n
        # F(n, I) = (c0 * n * e^-I/K) * ((m_(t-1) - c1 * tau * d_0 * I/G) - n) - d_0 * n

        # F1(n, I) = (c0 * n * e^-I/K)
        # F2(n, I) = ((m_(t-1) - c1 * tau * d_0 * I/G) - n)

        f1 = self.params['c0'] * n * np.exp(-1 * I / self.params['K'])
        # When I was working with Pranav he said to use <n>_(t-1) instead of I/G so I'm using that instead
        # even though they should work out to be the same
        f2 = self.params['m_(t-1)'] - self.params['c1'] * self.params['tau'] * self.params['c0'] * self.params['m_(t-1)'] * self.params['<n>_(t-1)'] - n

        # F(n, I) = F1(n, I) * F2(n, I) - d_0 * n

        # dFdI = dF1dI * F2 + dF2dI * F1
        # dF1dI = (c0 * n * -1/K * e^-I/K)
        # dF2dI = (-c1 * tau * d_0 * 1/G)
        df1dI = self.params['c0'] * n * (-1/self.params['K']) * np.exp(-1 * I / self.params['K'])
        df2dI = -1 * self.params['c1'] * self.params['tau'] * self.params['c0'] * self.params['m_(t-1)'] * (1/self.params[self.num_groups])

        return df1dI * f2 + df2dI * f1
    
    def calculate_determinant(self, lambdas: list, verbose = False):
        old_m = self.params['m_(t-1)']
        m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']
        params_copy = dict(self.params)
        params_copy['m_(t-1)'] = m_effective

        # since using previous lambdas to calculate Z, use old_m instead of m_eff
        Z = rf.R_mean(self.func, self.states[-1], old_m, self.params, lambdas = self.lambdas[-1])

        # use potential new lambdas, so new m_eff
        n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z
        f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z

        n2_mean = rf.Rn2_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z
        f2_mean = rf.Rf2_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z
        nf_mean = rf.Rnf_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z

        # There were some alternate formulas but this might be the best way to do this
        cov_nn = n2_mean - (n_mean ** 2)
        cov_ff = f2_mean - (f_mean ** 2)
        cov_nf = nf_mean - (n_mean * f_mean)

        return (cov_nn * cov_ff) - (cov_nf ** 2)

    
    def lambda_update(self, init=False, verbose=False):
        if init:
            new_lambdas = [np.log(
                ((self.states[0]/self.params[self.num_groups]) + 1)/(self.states[0]/self.params[self.num_groups])), 0]
            return np.array(new_lambdas)
        if verbose:
            print("I = ", self.states[-1])
        
        curr_lambdas = self.lambdas[-1]
        curr_lambda1, curr_lambda2 = self.lambdas[-1]
        curr_derivative = self.derivatives[-1]
        dt = self.params['tau']

        old_m = self.params['m_(t-1)']
        m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']
        params_copy = dict(self.params)
        params_copy['m_(t-1)'] = m_effective

        D = self.calculate_determinant(self.lambdas[-1])

        # since using previous lambdas to calculate Z, use old_m instead of m_eff
        Z = rf.R_mean(self.func, self.states[-1], old_m, self.params, lambdas = self.lambdas[-1])

        # use potential new lambdas, so new m_eff
        n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas)/Z
        f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas,
          lambda n: self.func(n, self.states[-1], self.params))/Z

        n2_mean = rf.Rn2_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas)/Z
        f2_mean = rf.Rf2_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas)/Z
        nf_mean = rf.Rnf_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas,
          lambda n: self.func(n, self.states[-1], self.params))/Z

        # print(n_mean, f_mean)
        # print(n2_mean, f2_mean, nf_mean)

        # There were some alternate formulas but this might be the best way to do this
        cov_nn = n2_mean - (n_mean ** 2)
        cov_ff = f2_mean - (f_mean ** 2)
        cov_nf = nf_mean - (n_mean * f_mean)

        # print(cov_nn, cov_ff, cov_nf)

        # Calculate derivative means
        dfdt_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas, 
                              mean_func = lambda n: curr_derivative * self.pandemic_dfdI(n, self.states[-1], self.params))
        n_dfdt_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, curr_lambdas, 
                              mean_func = lambda n: n * curr_derivative * self.pandemic_dfdI(n, self.states[-1], self.params))
        cov_n_dfdt = n_dfdt_mean - n_mean * dfdt_mean

        # print(dfdt_mean, n_dfdt_mean, cov_n_dfdt)

        delta_lambda1 = (-1 * ((cov_n_dfdt * curr_lambda2 + (curr_derivative / self.params[self.num_groups]))) * cov_ff) / D
        delta_lambda2 = ((cov_n_dfdt * curr_lambda2 + (curr_derivative / self.params[self.num_groups])) * cov_nf)/D

        # print(
        #   delta_lambda1,
        #   delta_lambda2
        # )
        return np.array([
            curr_lambda1 + dt * delta_lambda1,
            curr_lambda2 + dt * delta_lambda2
        ])

    def brute_force_update(self, init : bool = False) -> list:
        """Returns lambda calculated with brute force method

        Args:
            single_lambda: set to true when finding initial lambdas, indicates
                that lambda2 should be 0
        Returns:
            list containing new lambdas
            

        """
        def constraints(lambdas: np.array, init_lambdas = False) -> list:
            """Returns error between actual <n>, <nf> and the value calculated by the given lambdas

            Args:
                lambdas: A 1d array of the two lambdas

            Returns:
                2 element list with the difference in <n> and <nf>
            
            """
            if(init_lambdas):
                lambdas[1] = 0
            if lambdas[0] <= 0:
                lambdas[0] = -1 * lambdas[0]
            if lambdas[1] <= 0:
                lambdas[1] = -1 * lambdas[1]
            old_m = self.params['m_(t-1)']
            m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']
            Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, lambdas) #Calculate normalization factor

            n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z
            f_mean = rf.R_mean(self.func, self.states[-1],  m_effective, self.params, lambdas, mean_func=lambda n: self.func(n, self.states[-1], self.params))/Z

            if init_lambdas:
                return [n_mean * self.params[self.num_groups] - self.states[-1], 0]
            
            return [n_mean * self.params[self.num_groups] - self.states[-1], f_mean * self.params[self.num_groups] - self.derivatives[-1]]
        

        if init:
            new_lambdas = fsolve(constraints, self.lambdas[-1], args=True)
            return new_lambdas

        temp_lambdas = [self.lambdas[-1][0], self.lambdas[-1][1]]
        if temp_lambdas[0] <= 0:
            temp_lambdas[0] = -1 * temp_lambdas[0]
        if temp_lambdas[1] <= 0:
            temp_lambdas[1] = -1 * temp_lambdas[1]
        
        starting_point = [temp_lambdas[0] - 0.0000000001, temp_lambdas[1] - 0.001]
        new_lambdas = fsolve(constraints, starting_point)

        if new_lambdas[0] <= 0:
            new_lambdas[0] = -1 * new_lambdas[0]
        if new_lambdas[1] <= 0:
            new_lambdas[1] = -1 * new_lambdas[1]

        return new_lambdas
    
    def find_steady_state_params(self, param_key : str) -> float:
        """Finds value for param_key such that <f> = 0

        Args: 
            param_key: key of parameter in self.params to be tuned

        Returns:
            value of parameter such that the system is in steady state.

        """
        
        params_copy = dict(self.params)
        old_m = self.params['m_(t-1)']
        m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']

        def get_derivatives(par_val):

            params_copy[param_key] = par_val
        
            lambdas = self.lambdas[0] #Lambda 1 is set to 0 at initial iteration

            Z = rf.R_mean(self.func, self.states[-1], m_effective, params_copy, lambdas) #Calculate normalization factor


            f_mean = rf.R_mean(self.func, self.states[-1], m_effective, params_copy, lambdas, mean_func=lambda n: self.func(n, self.states[-1], params_copy))/Z
            
            return f_mean #Want this to be 0
     
        return fsolve(get_derivatives, 0.02)[0]
    
    
    def update_param(self, param_key : str, new_val : float):
        """ Update parameters according to method outlined in paper

        Args:
            param_key: Key in self.params corresponding to value to be changed
            new_val: new value
        """
        
        self.params[param_key] = new_val
        old_m = self.params['m_(t-1)']
        m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']

        Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1]) #Calculate normalization factor 
        f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, self.lambdas[-1], mean_func=lambda n: self.func(n, self.states[-1], self.params))/Z
        self.derivatives[-1] = f_mean * self.params[self.num_groups]



    def check_constraints(self, lambdas: list, error_lim: float, print_toggle: bool = False):
        """ Asserts that calculated value of <n> using current lambdas is close to self.state[-1]
        
        Args:
            lambdas: list containing lambdas
            error_lim: error tolerance 
        """
        old_m = self.params['m_(t-1)']
        m_effective = old_m - self.params['tau'] * self.params['c1'] * self.params['c0'] * old_m * self.params['<n>_(t-1)']
        Z = rf.R_mean(self.func, self.states[-1], m_effective, self.params, lambdas) #Calculate normalization factor

        n_mean = rf.Rn_mean(self.func, self.states[-1], m_effective, self.params, lambdas)/Z
        f_mean = rf.R_mean(self.func, self.states[-1], m_effective, self.params, lambdas, mean_func=lambda n: self.func(n, self.states[-1], self.params))/Z

        iter_num = len(self.states)
        if print_toggle:
          print("Error:",(n_mean * self.params[self.num_groups] - self.states[-1]) ** 2)
          print(n_mean, self.lambdas[-1], n_mean, Z)
        self.constraint_errors.append((n_mean * self.params[self.num_groups] - self.states[-1]) ** 2)
        if not ((n_mean * self.params[self.num_groups] - self.states[-1]) ** 2 < error_lim**2):
          print(n_mean, self.states[-1])
        assert (n_mean * self.params[self.num_groups] - self.states[-1]) ** 2 < error_lim**2, "Constraints not satisfied at iteration " + str(iter_num)
        

    def graph(self) -> None:
        """ Displays graph of state vs time, derivatives vs time, lambdas vs time
        """


        fig, axs = plt.subplots(2, 2,constrained_layout=True)
       
        axs[0, 0].plot(self.time,self.states)
        axs[0, 0].set_title('State')
        axs[0, 1].plot(self.time,self.derivatives)
        axs[0, 1].set_title('Derivative')
        axs[1, 0].plot(self.time, np.array(self.lambdas)[:,0])
        axs[1, 0].set_title('Lambda 1')
        axs[1, 1].plot(self.time, np.array(self.lambdas)[:,1])
        axs[1, 1].set_title('Lambda 2')

        for ax in axs.flat:
            ax.set(xlabel='Time')

        plt.show()
    
    def graph_v2(self) -> None:
        """ Displays graph of state vs time, m_eff vs time,
            cumulative number of deaths vs time, derivatives vs time
        """
        fig, axs = plt.subplots(2, 2,constrained_layout=True)
       
        axs[0, 0].plot(self.time,self.states)
        axs[0, 0].set_title('Number Infected')
        axs[0, 1].plot(self.time,self.m_effs)
        axs[0, 1].set_title('Number Susceptible (per group)')
        axs[1, 0].plot(self.time, self.num_dead)
        axs[1, 0].set_title('Cumulative Number Dead')
        axs[1, 1].plot(self.time, np.array(self.derivatives))
        axs[1, 1].set_title('Derivative')

        for ax in axs.flat:
            ax.set(xlabel='Time')

        plt.show()

    def graph_m_eff(self) -> None:
        plt.plot(self.time,self.m_effs)
    
    def graph_probability_distributions(self, x_upper_bound = -1, interval = 1, legend = False) -> None:
        """
        Displays graph of probability distribution at the end of the simulation
        """
        if x_upper_bound == -1:
            x_upper_bound = self.params['m_(t-1)'] + 1
        
        x_axis = np.arange(0,x_upper_bound + 1)
        prob_dist = rf.R(self.func, self.states[-1], self.params['m_(t-1)'], self.params, self.lambdas[-1])

        key_list = list(self.probability_distributions.keys())
        num_distributions = len(key_list)

        prob_cmap = plt.get_cmap('viridis')
        prob_norm = plt.Normalize(vmin=0, vmax=max(key_list))

        for i in range(0, num_distributions):
            k = key_list[i]
            if k % interval == 0:
              normalized_pd = self.probability_distributions[k]/sum(self.probability_distributions[k])

              plt.plot(x_axis,
                normalized_pd[0:x_upper_bound + 1],
                label = "t =" + str(k),
                color = prob_cmap(prob_norm(k)),
                linewidth=0.5,
                marker = 'o',
                markersize = 3)
              plt.xlabel("n")
              plt.ylabel("P(n)")

              print(self.probability_distributions[k][0:10])

        if legend:
            plt.legend()
        plt.show()

    def calculate_mean_constraint_error(self):
        return np.mean(self.constraint_errors)

