# %%
'''
Quadratic Sieve Implementation by Niccolò Mattarei, 2022. Includes code for generating the chart and graphs in my report.
#Run in VSCODE for Jupyter Notebook style cells.
'''

import math
import sympy
import numpy as np
import itertools

def find_factor_base(n, smoothness_bound = 0, SURPRESS_WARNING = False):
    """
    Generates the factor base.
    """
    if smoothness_bound == 0:
        smoothness_bound = math.ceil(math.exp(0.5*math.sqrt(math.log(n)*math.log(math.log(n)))))
        print("Optimal smoothness bound of ", smoothness_bound, " used.")
    factor_base = [2]
    for p in sympy.primerange(3, smoothness_bound):
        legendre_symbol = pow(n, int((p-1)/2), p)
        if legendre_symbol == 0 and not SURPRESS_WARNING:
            print("A prime below the smoothness bound (", p, ") divides n")
        if legendre_symbol == 1:
            factor_base.append(p)
    return factor_base

def solve_congruence(n, p):
    """
    Solves x^2 = n (mod p) and returns x.
    """
    for t in range(p):
        if (t**2 - n) % p == 0:
            return t

def log_function(x):
    """
    Returns some approximation of the logarithm.
    """
    return round(math.log(x,2))

def print_r_i_info(r_i, f_r_i):
    print("r_i =", r_i, ", f(r_i) =", f_r_i, ", log\u2082(f(r_i)) =", log_function(f_r_i))

def print_latex_matrix(numpy_array):
    print("\\begin{bmatrix}")
    for row in numpy_array:
        print(" & ".join([str(element) for element in row]), "\\\\")
    print("\\end{bmatrix}")

def qs_sieve(n, factor_base, t, LOG_TOLERANCE, LOG_FUNCTION = log_function, OVERKILL = 1,
             SIEVE_INTERVAL_SIZE = 0, PRINT_INFO = True):
    """
    Sieves through f(r_i) using a factor base of odd primes, to find which are smooth
    Returns (smooth_f_r_i, found_sufficient, probably_smooth_count), where smooth_f_r_i is an array in
    which an element looks like:
        (r_i, f(r_i), list of exponents to which factor base is raised to get f(r_i)),
    for smooth f(r_i).

    Keyword arguments:
    LOG_TOLERANCE is how strict we are when finding candidate smooth f_r_i
    LOG_FUNCTION is the function used when sieving, this could be math.log or some custom function with rounding 
        (default base 2 logarithm rounded to nearest integer)
    OVERKILL is the amount of smooth f(r_i) to find more than the size of the factor base (can be negative) 
        (default 1)
    SIEVE_INTERVAL_SIZE is the amount of f(r_i) checked for smoothness (should be smaller than or equal to n) 
        (default n)
    """
    if SIEVE_INTERVAL_SIZE == 0:
        SIEVE_INTERVAL_SIZE = math.ceil((3-8**0.5)*n)
    found_sufficient = True
    probable_smooth_count = 0
    r_i = math.floor(n**0.5)+1
    smooth_f_r_i = []

    while len(smooth_f_r_i) < len(factor_base) + OVERKILL and r_i <= math.floor(n**0.5)+1 + SIEVE_INTERVAL_SIZE:
        #Iterate until we have enough for Gaussian elimination.
        f_r_i = r_i**2-n
        sum_of_prime_logs = 0
        for p_j in factor_base:
            r_i_mod_p_j = r_i % p_j
            if r_i_mod_p_j == t[p_j] or r_i_mod_p_j == p_j-t[p_j]:
                #p_j divides f(r_i)
                sum_of_prime_logs += LOG_FUNCTION(p_j)

        if abs(LOG_FUNCTION(f_r_i) - sum_of_prime_logs) < LOG_TOLERANCE:
            #Now we need to check that f(r_i) is actually smooth by trial division before adding it to smooth_f_r_i.
            #print("Testing probable smooth number", f_r_i)
            probable_smooth_count += 1
            f_r_i_factor_powers = [0]*len(factor_base)
            for j, p_j in enumerate(factor_base):
                a = 1 #Power to which p_j is raised
                try_next_a = True
                while try_next_a:
                    if f_r_i % (p_j**a) == 0:
                        f_r_i_factor_powers[j] = a
                        a += 1
                    else:
                        try_next_a = False
            if np.prod(list(map(pow, factor_base, f_r_i_factor_powers)), dtype='int64') == f_r_i:
                #Found smooth number f_r_i and its factorisation
                smooth_f_r_i.append((r_i, f_r_i, f_r_i_factor_powers))
                if PRINT_INFO:
                    print(len(smooth_f_r_i), "/", len(factor_base) + OVERKILL, "f(r_{}) = {} is smooth".format(r_i-math.floor(n**0.5), f_r_i))
        r_i += 1
    
    if r_i > (2*n)**0.5:
        found_sufficient = False

    return (smooth_f_r_i, found_sufficient, probable_smooth_count)

def gaussian_elimination_step(n, smooth_f_r_i, factor_base, PRINT_MATRICES = False, PRINT_INFO = True,
                              CHECK_EVERY_SQUARE_COMBINATION = False):
    """
    For each possible combination of smooth f(r_i), returns a product of a subset 
    of them which is a square, alongside its factorisation.
    Keyword arguments:
    PRINT_MATRICES can be set to True if Latex formatted matrices are needed (default False)
    PRINT_INFO can be set to false to turn off all printing besides displaying the final factors.
    """
    found_factor = False
    exp_matrix = np.transpose(np.array([x[2] for x in smooth_f_r_i]))
    exp_matrix_b2 = np.mod(exp_matrix, 2)   #Binary matrix of exponents, will be operated on in-place
    if PRINT_MATRICES:
        print("Initial exponent matrix")
        print_latex_matrix(exp_matrix)
        print("Exponent matrices base 2, with elimination")

    no_of_primes = exp_matrix.shape[0]
    no_of_smooth_f_r_i = exp_matrix.shape[1]

    #The kth row of marks is all of the f(r_i) columns that have contributed to the kth f(r_i) column
    marks = np.identity(no_of_smooth_f_r_i)   #[column affected by row operations, column with pivot]
    indices_of_zero_cols = []

    for i in range(no_of_smooth_f_r_i):   #For each column f(r_i)
        if PRINT_MATRICES:
            print_latex_matrix(exp_matrix_b2)
        
        #First check that we have not already achieved a square, i.e. a row of zeroes:
        #This prevents there being a zero column later, by which point we have a large marks array already.
        for i_ in range(i, no_of_smooth_f_r_i):
            if i_ not in indices_of_zero_cols and np.count_nonzero(exp_matrix_b2[:,i_]) == 0:
                #If column of zeroes is found, test the combination of f(r_i).
                free_index = [0]*(exp_matrix_b2.shape[1])   #numpy array with 1 at i_th index
                free_index[i_] = 1
                exponent_array = np.sum(np.multiply(marks[i_], exp_matrix), axis=1)
                if PRINT_INFO:
                    print("Found product of f(r_i) which is square: ",
                        [smooth_f_r_i[l][1] for l in range(no_of_smooth_f_r_i) if marks[i_,l] == 1])
                x = math.prod([pow(factor_base[l], int(exponent_array[l]/2), n) for l in range(no_of_primes)]) % n
                y = math.prod([smooth_f_r_i[l][0] for l in range(no_of_smooth_f_r_i) if marks[i_,l]  == 1]) % n
                factor1 = math.gcd(x-y,n)
                print("Gives factor", factor1)
                if factor1 != 1 and factor1 != n:
                    if PRINT_INFO:
                        print("^ produces non-trivial factor of n: \n  x\u00b2 = ",
                        '*'.join([str(smooth_f_r_i[l][1]) for l in range(no_of_smooth_f_r_i) if marks[i_][l]  == 1]),
                        " ≡ " , 
                        math.prod([pow(factor_base[l], int(exponent_array[l]), n) for l in range(no_of_primes)]) % n,
                        " ≡ ",
                        '*'.join([(str(smooth_f_r_i[l][0]) + "\u00b2") for l in range(no_of_smooth_f_r_i) if marks[i_][l]  == 1]),
                        " = y\u00b2 (product of r_i\u00b2)")
                    factor2 = math.gcd(x+y,n)
                    if PRINT_INFO:
                        print("  gcd(x-y,n) = ", factor1, ", gcd(x+y,n) = ", factor2)
                    print("  n = ", factor1, "*", n // factor1)
                    found_factor = True
                indices_of_zero_cols.append(i_)   #So the column is not checked again
                if not CHECK_EVERY_SQUARE_COMBINATION:
                        break
        if found_factor and not CHECK_EVERY_SQUARE_COMBINATION:
            break

        for j in range(no_of_primes):   #Find the first prime dividing it with an odd exponent
            if exp_matrix_b2[j,i] == 1:
                for k in range(no_of_smooth_f_r_i):   #For each column other than the current one
                    if k == i:# or k in indices_of_zero_cols:
                        continue
                    if exp_matrix_b2[j,k] == 1:   #If the column addition is needed
                        marks[k,:] = np.mod(marks[k,:] + marks[i,:], 2) #Marks which f(r_i) have contributed to the construction of the square
                        exp_matrix_b2[:,k] = np.mod(exp_matrix_b2[:,k] + exp_matrix_b2[:,i], 2)   #Perform it
                break
    if PRINT_INFO:
        print("Amount of square products of f(r_i) found and checked: ", len(indices_of_zero_cols))
    if found_factor:
        return True
    else:
        return False

def qs(n, smoothness_bound = 0, LOG_TOLERANCE = 20, LOG_FUNCTION = log_function, OVERKILL = 1,
 SIEVE_INTERVAL_SIZE = 0, CHECK_EVERY_SQUARE_COMBINATION = False, PRINT_MATRICES = False, PRINT_INFO = True,
 SURPRESS_WARNING = False):
    """
    Runs the quadratic sieve algorithm on n with a given smoothness bound (optimised if not provided).
    Keyword arguments:
    n is the number to be factorised
    smoothness_bound is the upper bound on the factor base (default optimised)
    LOG_TOLERANCE is how strict we are when finding candidate smooth f_r_i
    LOG_FUNCTION is the function used when sieving, this could be math.log or some custom function with rounding 
        (default base 2 logarithm rounded to nearest integer)
    OVERKILL is the amount of smooth f(r_i) to find more than the size of the factor base (can be negative) 
        (default 1)
    SIEVE_INTERVAL_SIZE is the amount of f(r_i) checked for smoothness (should be smaller than or equal to n) 
        (default n)
    CHECK_EVERY_SQUARE_COMBINATION should be set to True if a multiplier is being used, so the Gaussian
        elimination stage is completed even if a factor is found.
    PRINT_MATRICES can be set to True if Latex formatted matrices are needed (default False)
    PRINT_INFO can be set to False to turn off all printing besides displaying the final factors
    SURPRESS_WARNING can be set to True to surpress the 'A prime below the smoothness bound divides n' warning.
    """
    if PRINT_INFO:
        print("n = ", n)
    factor_base = find_factor_base(n, smoothness_bound, SURPRESS_WARNING = SURPRESS_WARNING)
    if PRINT_INFO:
        print(len(factor_base), "primes in our factor base: ", factor_base)
    t = {p:solve_congruence(n, p) for p in factor_base}
    if PRINT_INFO:
        print("Solution to congruence for each prime: ", t)
    (smooth_f_r_i, found_sufficient, probable_smooth_count) = qs_sieve(n, factor_base, t, LOG_TOLERANCE, LOG_FUNCTION, OVERKILL,
        SIEVE_INTERVAL_SIZE, PRINT_INFO)
    if PRINT_INFO:
        print("Sieving stage complete")
        print("Smooth f(r_i) found / probable smooth f(r_i) found:", len(smooth_f_r_i), "/", probable_smooth_count)
    if len(smooth_f_r_i) > 0:
        found_factor = gaussian_elimination_step(n, smooth_f_r_i, factor_base, PRINT_MATRICES, PRINT_INFO, CHECK_EVERY_SQUARE_COMBINATION)
    return (len(smooth_f_r_i), probable_smooth_count)

def qs_multiplier_test(n_list, two_multiplier = False, three_multiplier = False, smoothness_bound = 0, SIEVE_INTERVAL_SIZE=1000):
    """Given a list of n that have no divisors under the smoothness bound, applies the quadratic sieve using a range of tolerances.
    depending on the multiplier chosen, will apply it to those n that need it. A list
    Returns a 2D array with columns
    (n, multiplier, # of smooth f(r_i), # of probably smooth f(r_i))"""
    multiplier_results_table = np.ones((len(n_list), 4), np.int64)
    multiplier_results_table[:,0] = n_list
    for i, n in enumerate(n_list):
        print(two_multiplier, three_multiplier)
        if two_multiplier:
            multiplier_results_table[i,1] = n % 8
        if three_multiplier:
            if n % 3 == 2:
                multiplier_results_table[i,1] = 2
        kn = int(multiplier_results_table[i,1]*n)
        (smooth_count, probable_smooth_count) = qs(kn, smoothness_bound, LOG_TOLERANCE = 50, OVERKILL=100, SIEVE_INTERVAL_SIZE=SIEVE_INTERVAL_SIZE, PRINT_INFO=False, SURPRESS_WARNING = True)
        multiplier_results_table[i,2] = smooth_count
        multiplier_results_table[i,3] = probable_smooth_count
    return multiplier_results_table

def qs_parameter_test(n_list, LOG_TOLERANCE_list, smoothness_bound = 0, SIEVE_INTERVAL_SIZE=1000):
    """Given a list of n that have no divisors under the smoothness bound, applies the quadratic sieve using a range of tolerances.
    Returns a 1D array of average amounts of smooth f(r_i) per tolerance and one of average amount of probably smooth f(r_i) per 
    tolerance."""
    smooth_count_matrix = np.zeros((len(n_list),len(LOG_TOLERANCE_list)),np.int32)
    probable_smooth_count_matrix = np.zeros((len(n_list),len(LOG_TOLERANCE_list)),np.int32)
    for i, n in enumerate(n_list):
        for j, LOG_TOLERANCE in enumerate(LOG_TOLERANCE_list):
            (smooth_count, probable_smooth_count) = qs(n, smoothness_bound, OVERKILL=100, LOG_TOLERANCE=LOG_TOLERANCE, SIEVE_INTERVAL_SIZE=SIEVE_INTERVAL_SIZE, PRINT_INFO=False)
            smooth_count_matrix[i,j] = smooth_count
            probable_smooth_count_matrix[i,j] = probable_smooth_count
            print("n =", n, "TOL =", LOG_TOLERANCE, "Smooth/Probably smooth:", smooth_count, "/", probable_smooth_count)
    return (np.mean(smooth_count_matrix, axis=0), np.mean(probable_smooth_count_matrix, axis=0))

def all_primes(start, end):
    """Generates a list of all the primes in between start and end."""
    return list(sympy.sieve.primerange(start, end))

def difficult_semiprimes(primes_centre, prime_range):
    """Generates a list of all semiprimes in an interval 2*prime_range long, centred on prime_centre."""
    n_factor_range = all_primes(primes_centre-prime_range,primes_centre+prime_range)
    return [x * y for x, y in itertools.combinations(n_factor_range, r=2)]

# %% Run tolerance test
LOG_TOLERANCE_list = range(1,30)
n_list = difficult_semiprimes(10000,100)
n_list.sort()
(average_smooth_count, average_probable_smooth_count) = qs_parameter_test(n_list, LOG_TOLERANCE_list, smoothness_bound=0)

# %% Tolerance test plot
import matplotlib.pyplot as plt
#plt.plot(LOG_TOLERANCE_list, average_probable_smooth_count, "g--", label="Number of f(r_i) identified as probably smooth")
plt.plot(LOG_TOLERANCE_list, average_smooth_count, "r--", label="Number of f(r_i) identified and verified as smooth"),
plt.legend()
plt.xlabel("Tolerance")
plt.savefig('tolerance_graph_full8')
plt.show()

# %% Run multiplier test
import pandas as pd
n_list = difficult_semiprimes(1000000,100)
n_list.sort()
no_mul_results = qs_multiplier_test(n_list, smoothness_bound = 120)
two_mul_results = qs_multiplier_test(n_list, two_multiplier = True, smoothness_bound = 120)
three_mul_results = qs_multiplier_test(n_list, three_multiplier = True, smoothness_bound = 120)
full_results = np.concatenate((no_mul_results[:,:3], two_mul_results[:,1:3], three_mul_results[:,1:3]), axis=1)
full_results_df = pd.DataFrame(full_results, columns = ['n','1','R(1)','a','R(a)','b','R(b)'])

# %% Multiplier results plot
best_results = full_results_df.query('a != 1 and b != 1')
print(len(best_results))
print(best_results.to_latex(columns=['n','R(1)','a','R(a)','b','R(b)'],index=False))
bar_plot = best_results[['n','R(1)','R(a)','R(b)']].plot.barh(x = 'n',figsize=(8,16))
bar_plot.legend(['Quadratic sieve run on n','Quadratic sieve run on an','Quadratic sieve run on bn'], loc='lower right')
bar_plot.set_xlabel("Number of first thousand f(r_i) that are smooth")
bar_plot.set_ylabel("n")
bar_plot.get_figure().savefig('D:\Documents\#University\Year 3\Project\Report\multiplier_bar_chart', bbox_inches = 'tight')

# %% Example 1
qs(7001*70001, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True, PRINT_MATRICES=True)
print(len([120243275875674, 156760187023377, 381644205160337, 751162839115962, 1212035271744069, 1304797810522689, 1683720207277073, 1693317183760098, 1771599368143962, 2246953922435397, 2333399775428673, 3028611948822789, 3168558401054469, 3750818961978714, 3810490712144298, 3897738801792549, 3912826819911282, 4244763413657754, 4447030625168229, 5897430296359973, 7298029050832449, 7726475689987173, 9615812761355514, 10336383706282257, 10445014964333274, 12338086880882042, 12793899321003402, 14491393802430042, 15677220877793637, 18358974046560069, 21496650351376074, 21757268384962698, 22016768949091098, 27470342706347073]))

# %% Example 2
qs(2**67-1, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True)

# %%Example 3 part i
qs(11111111111111111111111111111, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True)

# %% %%Example 3 part ii
qs(3482015390508026045475121, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True)

# %% Example 3 part iii (with multiplier)
qs(7*721429231, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True)

# %%Example 3 part iv
qs(4826551574132191, OVERKILL=8, LOG_TOLERANCE=20, PRINT_INFO=True)

# %%
