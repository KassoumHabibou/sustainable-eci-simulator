import pandas as pd
import numpy as np
from scipy.linalg import eig
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import inv
from itertools import product
import pulp




## RCA FUNCTION

def rca(X):
    X = np.array(X)
    Xc = X.sum(axis=1, keepdims=True)
    Xp = X.sum(axis=0)
    Xall = X.sum()

    R = (X * Xall) / (Xc * Xp)
    return R

## ECI/PCI FUNCTION

def cplex_rank(RCAcp, Country, Product):
    # Handle NaN values
    RCAcp = np.nan_to_num(RCAcp)
    
    # Create Mcp matrix
    Mcp = np.array(RCAcp >= 1, dtype=float)

    # Ubiquity and diversity
    Kp0 = Mcp.sum(axis=0)
    Kc0 = Mcp.sum(axis=1)

    # Calculate proximity of products (PHIpp)
    PHIpp = np.zeros((len(Kp0), len(Kp0)))
    for i in range(len(Kp0)):
        for j in range(len(Kp0)):
            PHIpp[i, j] = np.dot(Mcp[:, i], Mcp[:, j]) / max(Kp0[i], Kp0[j])

    # Calculate Relatedness
    Relatedness = np.dot(Mcp, PHIpp) / PHIpp.sum(axis=1)

    # Economic Complexity Index (ECI) and Product Complexity Index (PCI)
    Mcc = np.zeros((RCAcp.shape[0], RCAcp.shape[0]))  # Country transition matrix
    Mpp = np.zeros((RCAcp.shape[1], RCAcp.shape[1]))  # Product transition matrix

    for i in range(Mcc.shape[0]):
        for j in range(Mcc.shape[0]):
            Mcc[i, j] = np.sum(Mcp[i, :] * Mcp[j, :] / (Kp0 * Kc0[i]))

    for i in range(Mpp.shape[0]):
        for j in range(Mpp.shape[0]):
            Mpp[i, j] = np.sum(Mcp[:, i] * Mcp[:, j] / (Kc0 * Kp0[i]))

    # Eigenvalues and eigenvectors
    Vcc = eig(Mcc)[1]
    Vpp = eig(Mpp)[1]
    Vc = np.real(Vcc[:, 1])
    Vp = np.real(Vpp[:, 1])
    ECI = (Vc - np.mean(Vc)) / np.std(Vc)
    PCI = (Vp - np.mean(Vp)) / np.std(Vp)
    
    
    Kc1 = (Mcp @ Kp0) / Mcp.sum(axis=1)

    # Adjust signs based on correlation
    if np.corrcoef(ECI, Kc1)[0, 1] > 0:
        ECI *= -1
    if np.corrcoef(PCI, Kp0)[0, 1] > 0:
        PCI *= -1

    # Country and Product rankings
    Countryrankings = pd.DataFrame({'Country': Country, 'ECI': ECI})
    Productrankings = pd.DataFrame({'Product': Product, 'PCI': PCI})
    
    Countryrankings['COI'] = np.dot(Relatedness * (1-Mcp),PCI)
    
    # Calculate opportunity gain
    OpportunityGain = np.dot((1-Mcp)*PCI, PHIpp) / PHIpp.sum(axis=1)

    return Countryrankings, Productrankings, Relatedness, OpportunityGain

## PGI/PEII function

def estimate_product_index(xx,X_product_index):


    non_nan_mask = ~xx.isna()

    # Use the boolean mask to filter `used_rows_in_X` to only include rows where `xx` isn't NaN
    X_product_index = X_product_index[non_nan_mask.values, :]
    xx = xx[non_nan_mask]
    X_product_index.shape

    Xc = X_product_index.sum(axis=1, keepdims=True)
    Scp = X_product_index / Xc

    RCAcp = rca(X_product_index)

    Mcp = np.array(RCAcp >= 1, dtype=float)

    int_val = Scp * Mcp
    Np = Scp * Mcp
    Np = Np.sum(axis=0)
    Np = Np.reshape(-1, 1)


    xx_vector = xx.to_numpy().reshape(-1, 1)

    # Matrix multiplication of the transpose of int_val with xx_vector
    vals = np.dot(int_val.T, xx_vector)
    product_index_vals = vals / Np

    return product_index_vals


def eci_optimization(target_country, ECI_target, CountryRankings, ProductRankings, indices_to_exclude, beta_entry, beta_exit, PHIpp):

    ECI_initial = CountryRankings.loc[CountryRankings['Country'] == target_country, 'ECI_not_normalized'].values[0]
    normalized_product = CountryRankings['ECI_not_normalized'].values
    sd_for_ad = np.std(normalized_product)
    mean_for_ad = np.mean(normalized_product)

    pci = ProductRankings['PCI'].copy().values
    X_start = ProductRankings['X_start'].copy().values
    Relatedness_start = ProductRankings['Relatedness_start'].copy().values
    Relative_relatedness_start = ProductRankings['Relative_relatedness_start'].copy().values
    predicted_prob = ProductRankings['predicted_prob'].copy().values
    X_p_start = ProductRankings['X_p_start'].copy().values
    W_p = ProductRankings['W_p'].copy().values
    RCA_start = ProductRankings['RCA_start'].copy().values
    M_start = ProductRankings['M_start'].copy().values

    X_c_start = X_start.sum()
    
    RCA_start_entry = RCA_start[RCA_start < 1]
    Relatedness_start_entry = Relatedness_start[RCA_start < 1]
    Relative_relatedness_start_entry = Relative_relatedness_start[RCA_start < 1]

    RCA_start_exit = RCA_start[RCA_start >= 1]
    Relatedness_start_exit = Relatedness_start[RCA_start >= 1]
    Relative_relatedness_start_exit = Relative_relatedness_start[RCA_start >= 1]
    

    Ycp_entry = np.exp((np.log(2)-(beta_entry[0] + beta_entry[2] * np.log(1+RCA_start_entry) + beta_entry[3] * Relatedness_start_entry + beta_entry[4] * Relative_relatedness_start_entry))/beta_entry[1]) - RCA_start_entry - 1
    Ycp_exit = np.exp((np.log(2)-(beta_exit[0] + beta_exit[2] * np.log(1+RCA_start_exit) + beta_exit[3] * Relatedness_start_exit + beta_exit[4] * Relative_relatedness_start_exit))/beta_exit[1]) - RCA_start_exit - 1

#    Ycp_entry = (np.log(2)-(beta_entry[0] + beta_entry[1] * np.log(1+RCA_start_entry) + beta_entry[2] * np.log(1+RCA_start_entry) + beta_entry[4] * Relative_relatedness_start_entry))/beta_entry[3] - Relatedness_start_entry - 1
#    Ycp_exit = (np.log(2)-(beta_exit[0] + beta_exit[1] * np.log(1+RCA_start_exit) + beta_exit[2] * np.log(1+RCA_start_exit) + beta_exit[4] * Relative_relatedness_start_exit))/beta_exit[3] - Relatedness_start_exit - 1

    Ycp = np.full(RCA_start.shape, np.nan)
    Ycp[RCA_start < 1] = Ycp_entry
    Ycp[RCA_start >= 1] = Ycp_exit

    # Create a linear programming minimization problem
    prob = pulp.LpProblem("Minimize_Ycp_M_cp", pulp.LpMinimize)

    # Define binary decision variables M_cp based on the length of Ycp (same size as M_start)
    M_cp = [pulp.LpVariable(f"M_cp_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(len(Ycp))]

    # Objective function: Minimize the sum of Ycp * M_cp
    prob += pulp.lpSum([Ycp[i] * (M_cp[i]-M_start[i]) for i in range(len(Ycp))])

    # Constraint: sum_p M_cp * (pci_p - ECI_target) >= 0
    prob += pulp.lpSum([M_cp[i] * (pci[i] - ECI_target) for i in range(len(M_cp))]) >= 0

    for i in range(len(M_cp)):
        # First condition: prioritize M_start[i] > 1 and set M_cp[i] to 1
        if M_start[i] > 0:
            prob += M_cp[i] == 1
        
        # Second condition: apply the check for predicted_prob[i] only when M_start[i] <= 1
        elif M_start[i] < 1 and RCA_start[i] > 1:

            prob += M_cp[i] == 0

    # Solve the problem
    prob.solve()

        # Assuming M_cp and Y_cp are already computed (from your optimization)
    # M_cp and Y_cp are lists, where each element corresponds to the respective product
    M_cp_solutions = [pulp.value(M_cp[i]) for i in range(len(M_cp))]
    Y_cp_values = Ycp.tolist()  # Convert Ycp from numpy array to list if necessary



    M_cp_solutions = [pulp.value(M_cp[i]) for i in range(len(M_cp))]

    M_change = M_cp_solutions - M_start
    z_not_zero = M_change > 0

    # Filter P_representative_country based on pci >= ECI_initial
    X_start_filtered = X_start[z_not_zero]

    
    # Similarly, apply filtering to Relatedness_start
    Relatedness_start_filtered = Relatedness_start[z_not_zero]
    Relative_relatedness_start_filtered = Relative_relatedness_start[z_not_zero]
    X_p_start_filtered = X_p_start[z_not_zero]
    W_p_filtered = W_p[z_not_zero]
    RCA_start_filtered = RCA_start[z_not_zero]

    RCA_start_filtered_entry = RCA_start_filtered[RCA_start_filtered<1]
    Relatedness_start_filtered_entry = Relatedness_start_filtered[RCA_start_filtered<1]
    Relative_relatedness_start_filtered_entry = Relative_relatedness_start_filtered[RCA_start_filtered<1]
    W_p_filtered_entry = W_p_filtered[RCA_start_filtered<1]

    RCA_start_filtered_exit = RCA_start_filtered[RCA_start_filtered>=1]
    Relatedness_start_filtered_exit = Relatedness_start_filtered[RCA_start_filtered>=1]
    Relative_relatedness_start_filtered_exit = Relative_relatedness_start_filtered[RCA_start_filtered>=1]
    W_p_filtered_exit = W_p_filtered[RCA_start_filtered>=1]

    # calculate parameters:
    alpha_p_entry = beta_entry[0] + beta_entry[2] * np.log(1+RCA_start_filtered_entry) + beta_entry[3] * Relatedness_start_filtered_entry + beta_entry[4] * Relative_relatedness_start_filtered_entry
    theta_p_entry = W_p_filtered_entry*(((1+1.2)/np.exp(alpha_p_entry))**(1/beta_entry[1])-1)

    alpha_p_exit = beta_exit[0] + beta_exit[2] * np.log(1+RCA_start_filtered_exit) + beta_exit[3] * Relatedness_start_filtered_exit + beta_exit[4] * Relative_relatedness_start_filtered_exit
    theta_p_exit = W_p_filtered_exit*(((1+1.2)/np.exp(alpha_p_exit))**(1/beta_exit[1])-1)

    alpha_p = np.full(RCA_start_filtered.shape, np.nan)
    alpha_p[RCA_start_filtered < 1] = alpha_p_entry
    alpha_p[RCA_start_filtered >= 1] = alpha_p_exit

    theta_p = np.full(RCA_start_filtered.shape, np.nan)
    theta_p[RCA_start_filtered < 1] = theta_p_entry
    theta_p[RCA_start_filtered >= 1] = theta_p_exit


    B = theta_p * X_c_start - X_start_filtered

    # Number of products or elements after filtering
    n = len(Relatedness_start_filtered)

    # Initialize Theta as a square matrix of zeros
    Theta = np.zeros((n, n))

    # Populate the matrix according to the given conditions
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal entries
                Theta[i, j] = 1 - theta_p[i]
            else:
                # Off-diagonal entries
                Theta[i, j] = - theta_p[i]

    # Calculate the inverse of Theta

    # Objective function coefficients
    c = np.ones(n)  # n is the number of variables in x

    # Since linprog does not directly support > constraints, we'll use >= by formulating it as -Theta * x <= -B
    A_ub = -Theta
    b_ub = -B

    # Bounds for x ensuring x >= 0
    bounds = [(0, None) for _ in range(n)]
    
    if len(A_ub)>0:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        # Multiply the inverse of Theta with B
        vals = np.zeros(len(X_start))
        vals[z_not_zero] = res.x
    else:
        vals = np.zeros(len(X_start))

    X_start_adjusted = X_start + vals
    X_c_start_adjusted = X_c_start + np.sum(vals)
    RCA_adjusted = (X_start_adjusted / X_c_start_adjusted) / W_p

    RCA_adjusted_entry = RCA_adjusted[RCA_start < 1]
    probit_input_entry = np.column_stack(( np.log1p(RCA_adjusted_entry),  np.log1p(RCA_start_entry), Relatedness_start_entry, Relative_relatedness_start_entry))
    linear_predictor_entry = probit_input_entry @ beta_entry[1:] + beta_entry[0]

    RCA_adjusted_exit = RCA_adjusted[RCA_start >= 1]
    probit_input_exit = np.column_stack(( np.log1p(RCA_adjusted_exit),  np.log1p(RCA_start_exit), Relatedness_start_exit, Relative_relatedness_start_exit))
    linear_predictor_exit = probit_input_exit @ beta_exit[1:] + beta_exit[0]

    linear_predictor = np.full(RCA_start.shape, np.nan)
    linear_predictor[RCA_start < 1] = linear_predictor_entry
    linear_predictor[RCA_start >= 1] = linear_predictor_exit

    RCA_final = np.expm1(linear_predictor)
    expected_P = (RCA_final >= 1).astype(int)
    Relatedness_final = np.dot(expected_P, PHIpp) / PHIpp.sum(axis=1)

    M_start = (RCA_start >= 1).astype(int)
    mar_eci = ((expected_P / np.sum(expected_P)) - (M_start / np.sum(M_start)))*pci



    # Assuming ProductRankings_for_AD has a 'Product Name' column with product names in the same order as the results
    product_names = ProductRankings['Product'].tolist()
    product_codes = ProductRankings['Product'].tolist()


    # Creating the DataFrame
    df = pd.DataFrame({
        'Code': product_codes,
        'Name': product_names,
        'Mcp': M_cp_solutions,
        'Added_vol': vals,
        'RCA_final': RCA_final,
        'Relatedness_final': Relatedness_final,
        'mar_ECI': mar_eci
    })



    return df


def find_products_criteria(target_country, ECI_target, CountryRankings, ProductRankings, indices_to_exclude, beta_entry, beta_exit, criteria):
    ECI_initial = CountryRankings.loc[CountryRankings['Country'] == target_country, 'ECI_not_normalized'].values[0]
    expected_eci = ECI_initial.copy()

    pci = ProductRankings['PCI'].copy().values
    Mcp = ProductRankings['M_start'].copy().values
    X_start = ProductRankings['X_start'].copy().values
    Relatedness_start = ProductRankings['Relatedness_start'].copy().values
    Relative_relatedness_start = ProductRankings['Relative_relatedness_start'].copy().values
    W_p = ProductRankings['W_p'].copy().values
    RCA_start = ProductRankings['RCA_start'].copy().values
    X_c_start = X_start.sum()
    optimal_threshold = 1

    # Define z_not_zero based on conditions
    z_not_zero = np.where((pci >= ECI_initial))[0]
    z_zero = np.where((Mcp > 0) | (RCA_start > 1))[0]
    z_not_zero = np.setdiff1d(z_not_zero, indices_to_exclude)
    z_not_zero = np.setdiff1d(z_not_zero, z_zero)

    # Sort ProductRankings by criteria from largest to smallest
    sorted_indices = np.argsort(-criteria)

    # Filter sorted_indices to include only rows in z_not_zero
    sorted_indices = [idx for idx in sorted_indices if idx in z_not_zero]

    # Initialize index for iteration
    k = 1  # Start from 1 to include at least one product

    # Variables to store the desired values when expected_eci >= ECI_target
    stored_vals = None
    stored_RCA_final = None
    stored_expected_P = None
    stored_k = None

    # Continue while the expected ECI is less than the target
    while k <= len(sorted_indices):
        selected_indices = sorted_indices[0:k]  # Select top k products

        # Filter data for selected products
        X_start_filtered = X_start[selected_indices]
        Relatedness_start_filtered = Relatedness_start[selected_indices]
        Relative_relatedness_start_filtered = Relative_relatedness_start[selected_indices]
        W_p_filtered = W_p[selected_indices]
        RCA_start_filtered = RCA_start[selected_indices]

        # Proceed only if there are selected products
        n = len(selected_indices)
        if n == 0:
            print("No products selected. Exiting loop.")
            break

        # Separate entry and exit products
        RCA_start_filtered_entry = RCA_start_filtered[RCA_start_filtered < 1]
        Relatedness_start_filtered_entry = Relatedness_start_filtered[RCA_start_filtered < 1]
        Relative_relatedness_start_filtered_entry = Relative_relatedness_start_filtered[RCA_start_filtered < 1]
        W_p_filtered_entry = W_p_filtered[RCA_start_filtered < 1]

        RCA_start_filtered_exit = RCA_start_filtered[RCA_start_filtered >= 1]
        Relatedness_start_filtered_exit = Relatedness_start_filtered[RCA_start_filtered >= 1]
        Relative_relatedness_start_filtered_exit = Relative_relatedness_start_filtered[RCA_start_filtered >= 1]
        W_p_filtered_exit = W_p_filtered[RCA_start_filtered >= 1]

        # Calculate parameters
        alpha_p_entry = beta_entry[0] + beta_entry[2] * np.log(1 + RCA_start_filtered_entry) + beta_entry[3] * Relatedness_start_filtered_entry + beta_entry[4] * Relative_relatedness_start_filtered_entry
        theta_p_entry = W_p_filtered_entry * (((1 + 1.2) / np.exp(alpha_p_entry)) ** (1 / beta_entry[1]) - 1)

        alpha_p_exit = beta_exit[0] + beta_exit[2] * np.log(1 + RCA_start_filtered_exit) + beta_exit[3] * Relatedness_start_filtered_exit + beta_exit[4] * Relative_relatedness_start_filtered_exit
        theta_p_exit = W_p_filtered_exit * (((1 + 1.2) / np.exp(alpha_p_exit)) ** (1 / beta_exit[1]) - 1)

        alpha_p = np.full(RCA_start_filtered.shape, np.nan)
        alpha_p[RCA_start_filtered < 1] = alpha_p_entry
        alpha_p[RCA_start_filtered >= 1] = alpha_p_exit

        theta_p = np.full(RCA_start_filtered.shape, np.nan)
        theta_p[RCA_start_filtered < 1] = theta_p_entry
        theta_p[RCA_start_filtered >= 1] = theta_p_exit

        B = theta_p * X_c_start - X_start_filtered

        # Initialize Theta as a square matrix of zeros
        Theta = np.zeros((n, n))

        # Populate the matrix according to the given conditions
        for i in range(n):
            for j in range(n):
                if i == j:
                    Theta[i, j] = 1 - theta_p[i]
                else:
                    Theta[i, j] = -theta_p[i]

        # Objective function coefficients
        c = np.ones(n)  # n is the number of variables in x

        # Since linprog does not directly support >= constraints, we'll use <= by formulating it as -Theta * x <= -B
        A_ub = -Theta
        b_ub = -B

        # Bounds for x ensuring x >= 0
        bounds = [(0, None) for _ in range(n)]

        if len(A_ub) > 0:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                vals = np.zeros(len(X_start))
                vals[selected_indices] = res.x
            else:
                print(f"Optimization failed at iteration {k}: {res.message}")
                break
        else:
            vals = np.zeros(len(X_start))

        # Update X_start_adjusted and calculate RCA_adjusted
        X_start_adjusted = X_start + vals
        X_c_start_adjusted = X_c_start + np.sum(vals)
        RCA_adjusted = (X_start_adjusted / X_c_start_adjusted) / W_p

        # Prepare data for probit model
        RCA_adjusted_entry = RCA_adjusted[RCA_start < 1]
        probit_input_entry = np.column_stack((np.log1p(RCA_adjusted_entry), np.log1p(RCA_start[RCA_start < 1]), Relatedness_start[RCA_start < 1], Relative_relatedness_start[RCA_start < 1]))
        linear_predictor_entry = probit_input_entry @ beta_entry[1:] + beta_entry[0]

        RCA_adjusted_exit = RCA_adjusted[RCA_start >= 1]
        probit_input_exit = np.column_stack((np.log1p(RCA_adjusted_exit), np.log1p(RCA_start[RCA_start >= 1]), Relatedness_start[RCA_start >= 1], Relative_relatedness_start[RCA_start >= 1]))
        linear_predictor_exit = probit_input_exit @ beta_exit[1:] + beta_exit[0]

        linear_predictor = np.full(RCA_start.shape, np.nan)
        linear_predictor[RCA_start < 1] = linear_predictor_entry
        linear_predictor[RCA_start >= 1] = linear_predictor_exit

        RCA_final = np.expm1(linear_predictor)
        expected_P = (RCA_final >= optimal_threshold).astype(int)
        M_new = ((expected_P-Mcp) > 0).astype(int)
        M_new = M_new + Mcp
        expected_eci = np.sum(M_new * pci) / np.sum(M_new)

        print(f"Iteration {k}, Expected ECI: {expected_eci}")

        # Check if expected_eci meets or exceeds ECI_target
        if expected_eci >= ECI_target:
            # Store the current vals and RCA_final
            stored_vals = vals.copy()
            stored_RCA_final = RCA_final.copy()
            stored_expected_P = expected_P.copy()
            stored_k = k
            break  # Exit the loop as we've met the condition

        k += 1  # Increment k for the next iteration

    # After the loop, check if we have stored values
    if stored_vals is not None and stored_RCA_final is not None:
        # Use the stored values
        vals = stored_vals
        RCA_final = stored_RCA_final
        expected_P = stored_expected_P
        k = stored_k
    else:
        # If the condition was never met, handle accordingly
        print("ECI_target was not reached. Returning the last computed values.")

    # Assuming ProductRankings has 'Product' column with product names
    product_names = ProductRankings['Product'].tolist()
    product_codes = ProductRankings['Product'].tolist()

    # Creating the DataFrame
    df = pd.DataFrame({
        'Code': product_codes,
        'Name': product_names,
        'Added_vol': vals,
        'RCA_final': RCA_final
    })

    return df
