import numpy as np
import time # To measure execution time
# import tracemalloc # For memory profiling - Disabled for performance optimization

# --- Problem Definition & Fitness Function Parameters ---
TOTAL_PERSONNEL = 100       # P: Total available emergency personnel
N_AREAS = 5                 # Z: Total number of flood-affected zones

# Parameters SPECIFICALLY for your 'calculate_fitness' function:
RISK_SCORES_PER_ZONE = np.array([5, 4, 3, 2, 1]) # r_i (ensure length matches N_AREAS)
POPULATION_PER_ZONE = np.array([1000, 1500, 800, 1200, 900]) # Example values for p_i, ensure length matches N_AREAS

# Fitness Weights (STARTING POINT - TUNE THESE CAREFULLY FOR BOTH ALGORITHMS)
W1 = 0.2  # Weight for Coverage Ratio 
W2 = 0.3  # Weight for Risk-Weighted Allocation (using log(1+r_i))
W3 = 0.1  # Weight for Imbalance (sigma / (mu + epsilon))
W4 = 0.4  # NEW Weight for Population-Based Allocation (using log(1+p_i))

# Small constant for imbalance calculation
EPSILON = 1e-6 # A small value to prevent division by zero if mu is 0

# --- PSO Parameters (User Updated Values) ---
N_PARTICLES_PSO = 100   
MAX_ITER_PSO = 300      

W_PSO = 0.729       
C1_PSO = 1.49445    
C2_PSO = 1.49445    

V_MAX_PSO = (TOTAL_PERSONNEL * 0.20) 
V_MIN_PSO = -V_MAX_PSO

# --- Firefly Algorithm (FA) Parameters (Suggested Set 1 for Tuning) ---
N_FIREFLIES_FA = 100      # Number of fireflies (solutions)
MAX_GEN_FA = 300          # Maximum number of generations
ALPHA_FA = 0.5            # Randomization parameter (Moderate)
BETA0_FA = 1.0            # Attractiveness at distance r=0 (Standard starting point)
GAMMA_FA = 0.01           # Light absorption coefficient (Allows longer-range attraction)

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# SECTION 0: SHARED REPAIR FUNCTION                                            #
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def repair_allocation_enforce_sum(allocation_array, target_total_personnel, risk_scores_ref):
    """
    Repairs an allocation to ensure:
    1. Personnel counts are non-negative integers.
    2. The sum of allocated personnel EXACTLY matches target_total_personnel (if target_total_personnel > 0).
       Uses risk scores to guide addition of personnel if sum is too low.
    """
    if target_total_personnel == 0:
        return np.zeros_like(allocation_array, dtype=int)
    if not isinstance(allocation_array, np.ndarray): 
        allocation_array = np.array(allocation_array)
    if len(allocation_array) == 0: 
        return allocation_array

    repaired_alloc = np.round(allocation_array).astype(int)
    repaired_alloc[repaired_alloc < 0] = 0
    
    n_areas = len(repaired_alloc)

    safety_iters = 0 
    max_safety_iters = (target_total_personnel + n_areas) * 2 

    while np.sum(repaired_alloc) != target_total_personnel and safety_iters < max_safety_iters:
        current_sum_loop = np.sum(repaired_alloc) 
        if current_sum_loop > target_total_personnel:
            if np.sum(repaired_alloc) == target_total_personnel: break
            positive_indices = np.where(repaired_alloc > 0)[0]
            if not positive_indices.size: break 
            idx_to_reduce = positive_indices[np.argmax(repaired_alloc[positive_indices])]
            repaired_alloc[idx_to_reduce] -= 1
        elif current_sum_loop < target_total_personnel:
            risk_scores_np = np.array(risk_scores_ref) 
            if len(risk_scores_np) == n_areas:
                sorted_indices_by_risk = np.argsort(risk_scores_np)[::-1] 
                if np.sum(repaired_alloc) == target_total_personnel: break
                repaired_alloc[sorted_indices_by_risk[safety_iters % n_areas]] += 1 
            else: 
                if np.sum(repaired_alloc) == target_total_personnel: break
                repaired_alloc[safety_iters % n_areas] += 1
        safety_iters += 1
    
    final_adjustment = target_total_personnel - np.sum(repaired_alloc)
    if final_adjustment != 0 and n_areas > 0:
        temp_alloc_for_final_adjust = np.copy(repaired_alloc) 
        if final_adjustment > 0: 
            for i_adj in range(final_adjustment): 
                risk_scores_np = np.array(risk_scores_ref)
                if len(risk_scores_np) == n_areas:
                    add_idx = np.argsort(risk_scores_np)[::-1][i_adj % n_areas]
                    temp_alloc_for_final_adjust[add_idx] +=1
                else:
                    temp_alloc_for_final_adjust[i_adj % n_areas] +=1
        elif final_adjustment < 0: 
             for _ in range(abs(final_adjustment)):
                positive_indices = np.where(temp_alloc_for_final_adjust > 0)[0]
                if not positive_indices.size: break
                idx_to_reduce = positive_indices[np.argmax(temp_alloc_for_final_adjust[positive_indices])] 
                temp_alloc_for_final_adjust[idx_to_reduce] -=1
        if np.sum(temp_alloc_for_final_adjust) == target_total_personnel:
            repaired_alloc = temp_alloc_for_final_adjust
            
    return repaired_alloc
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# END SECTION 0: SHARED REPAIR FUNCTION                                        #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# SECTION 1: SHARED CUSTOM FITNESS FUNCTION (OPTIMIZED 4-term version)         #
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def calculate_fitness(
    zones_count: int,                           # Z
    total_personnel: int,                       # P
    assigned_personnel_np: np.ndarray,          # a_i (Now expects NumPy array)
    risk_scores_np: np.ndarray,                 # r_i (Now expects NumPy array)
    population_np: np.ndarray,                  # p_i (Now expects NumPy array)
    w1: float,
    w2: float,
    w3: float,
    w4: float,                                  # NEW weight
    epsilon: float 
) -> float:
    # This function returns a score where HIGHER IS BETTER.
    if zones_count == 0: 
        return 0.0 
    if total_personnel <= 0 and zones_count > 0 : 
         raise ValueError("'total_personnel' must be positive if zones_count > 0.")
    # Assuming lengths are pre-validated or match N_AREAS due to how they are created/passed

    # Term 1: w1 * (C / Z)
    C_calculated = np.sum(assigned_personnel_np >= 1) # Vectorized
    coverage_ratio_calculated = C_calculated / zones_count if zones_count > 0 else 0.0
    term1 = w1 * coverage_ratio_calculated

    # Term 2: w2 * (1 / P) * Sum_i_to_Z(a_i * log(1 + r_i))
    # Ensure risk_scores are non-negative for log. Create a mask for valid entries.
    valid_risk_mask = risk_scores_np >= 0
    log_risk_values = np.zeros_like(risk_scores_np, dtype=float)
    if np.any(valid_risk_mask): # Calculate log only for valid entries
        log_risk_values[valid_risk_mask] = np.log(1 + risk_scores_np[valid_risk_mask])
    
    log_risk_weighted_sum = np.sum(assigned_personnel_np * log_risk_values) # Vectorized
    
    priority_fulfillment_component = 0.0
    if total_personnel > 0:
        priority_fulfillment_component = log_risk_weighted_sum / total_personnel
    term2 = w2 * priority_fulfillment_component

    # Term 3: w3 * (sigma / (mu + epsilon))
    sigma = 0.0
    mu = 0.0
    imbalance_component = 0.0
    if assigned_personnel_np.size > 0: # Check if array is not empty
        sigma = np.std(assigned_personnel_np)
        mu = np.mean(assigned_personnel_np)
    
    if (mu + epsilon) != 0: 
        imbalance_component = sigma / (mu + epsilon)
    term3_for_subtraction = w3 * imbalance_component
        
    # Term 4: w4 * (1 / P) * Sum_i_to_Z(a_i * log(1 + p_i)) (NEW TERM)
    valid_pop_mask = population_np >= 0
    log_pop_values = np.zeros_like(population_np, dtype=float)
    if np.any(valid_pop_mask):
        log_pop_values[valid_pop_mask] = np.log(1 + population_np[valid_pop_mask])

    log_population_weighted_sum = np.sum(assigned_personnel_np * log_pop_values) # Vectorized
    
    population_component = 0.0
    if total_personnel > 0:
        population_component = log_population_weighted_sum / total_personnel
    term4_for_addition = w4 * population_component
    
    # Final Fitness: Term1 + Term2 - Term3 + Term4
    fitness_calculated = term1 + term2 - term3_for_subtraction + term4_for_addition
    return fitness_calculated
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# END SECTION 1: SHARED CUSTOM FITNESS FUNCTION (OPTIMIZED 4-term version)     #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# SECTION 2: SHARED WRAPPER OBJECTIVE FUNCTION (for minimization algorithms)   #
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def objective_function_wrapper_for_minimizers(allocation_particle_np: np.ndarray): 
    # No longer needs .tolist() as calculate_fitness now expects NumPy arrays
    # Global arrays RISK_SCORES_PER_ZONE and POPULATION_PER_ZONE are already NumPy arrays

    actual_score = calculate_fitness(
        zones_count=N_AREAS, 
        total_personnel=TOTAL_PERSONNEL, 
        assigned_personnel_np=allocation_particle_np, # Pass NumPy array directly
        risk_scores_np=RISK_SCORES_PER_ZONE,          # Pass NumPy array directly
        population_np=POPULATION_PER_ZONE,            # Pass NumPy array directly
        w1=W1, 
        w2=W2, 
        w3=W3,
        w4=W4, 
        epsilon=EPSILON 
    )
    return -actual_score 
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# END SECTION 2: SHARED WRAPPER OBJECTIVE FUNCTION                             #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# --- PSO Algorithm Implementation ---
def run_pso_allocation():
    print("\n--- Starting PSO Algorithm ---")
    # tracemalloc.start() # Memory tracing disabled for performance
    start_time_pso = time.time()

    if len(RISK_SCORES_PER_ZONE) != N_AREAS or len(POPULATION_PER_ZONE) != N_AREAS: 
        raise ValueError(f"PSO: Parameter array lengths (RISK_SCORES, POPULATION_PER_ZONE) must match N_AREAS ({N_AREAS}).")
    if N_AREAS == 0:
        print("PSO: N_AREAS is 0. No allocation possible.")
        # current_mem, peak_mem = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        # return np.array([]), 0, 0, peak_mem 
        return np.array([]), 0, 0, 0 # Return 0 for peak_mem if tracemalloc is off

    particle_positions_pso = np.zeros((N_PARTICLES_PSO, N_AREAS), dtype=int)
    if N_AREAS > 0 : 
        for i in range(N_PARTICLES_PSO):
            if TOTAL_PERSONNEL == 0:
                particle_positions_pso[i] = np.zeros(N_AREAS, dtype=int)
                continue
            proportions = np.random.dirichlet(np.ones(N_AREAS)) 
            temp_alloc = np.zeros(N_AREAS, dtype=int)
            assigned_sum = 0
            for j in range(N_AREAS - 1):
                val = int(round(proportions[j] * TOTAL_PERSONNEL))
                temp_alloc[j] = val
                assigned_sum += val
            temp_alloc[N_AREAS - 1] = TOTAL_PERSONNEL - assigned_sum
            particle_positions_pso[i] = repair_allocation_enforce_sum(temp_alloc, 
                                                                  TOTAL_PERSONNEL, 
                                                                  RISK_SCORES_PER_ZONE)

    particle_velocities_pso = np.random.uniform(V_MIN_PSO, V_MAX_PSO, (N_PARTICLES_PSO, N_AREAS))
    
    pbest_positions_pso = np.copy(particle_positions_pso)
    pbest_scores_pso = np.array([objective_function_wrapper_for_minimizers(p) for p in pbest_positions_pso])
    
    gbest_idx_pso = np.argmin(pbest_scores_pso)
    gbest_position_pso = np.copy(pbest_positions_pso[gbest_idx_pso])
    gbest_score_pso = pbest_scores_pso[gbest_idx_pso] 

    print(f"PSO Initial Actual Fitness (higher is better): {-gbest_score_pso:.4f}")
    print(f"PSO Initial Global Best Allocation: {gbest_position_pso.astype(int)}, Sum: {np.sum(gbest_position_pso)}\n")

    for iteration in range(MAX_ITER_PSO):
        for i in range(N_PARTICLES_PSO):
            current_pos = particle_positions_pso[i]
            current_vel = particle_velocities_pso[i]
            r1 = np.random.rand(N_AREAS)
            r2 = np.random.rand(N_AREAS)
            cognitive_component = C1_PSO * r1 * (pbest_positions_pso[i] - current_pos)
            social_component = C2_PSO * r2 * (gbest_position_pso - current_pos)
            new_velocity = W_PSO * current_vel + cognitive_component + social_component
            new_velocity = np.clip(new_velocity, V_MIN_PSO, V_MAX_PSO)
            particle_velocities_pso[i] = new_velocity
            new_position = current_pos + new_velocity
            repaired_position = repair_allocation_enforce_sum(new_position, 
                                                              TOTAL_PERSONNEL, 
                                                              RISK_SCORES_PER_ZONE)
            particle_positions_pso[i] = repaired_position
            current_score = objective_function_wrapper_for_minimizers(repaired_position)
            if current_score < pbest_scores_pso[i]:
                pbest_scores_pso[i] = current_score
                pbest_positions_pso[i] = np.copy(repaired_position)
                if current_score < gbest_score_pso:
                    gbest_score_pso = current_score
                    gbest_position_pso = np.copy(repaired_position)
        
        if (iteration + 1) % 50 == 0: 
            print(f"PSO Iteration {iteration + 1}/{MAX_ITER_PSO}, "
                  f"Best Actual Fitness: {-gbest_score_pso:.4f}, "
                  f"Allocation Sum: {np.sum(gbest_position_pso)}")

    end_time_pso = time.time()
    execution_time_pso = end_time_pso - start_time_pso
    
    # current_mem_pso, peak_mem_pso = tracemalloc.get_traced_memory() 
    # tracemalloc.stop() 
    peak_mem_pso = 0 # Placeholder if tracemalloc is off
    
    print(f"--- PSO Algorithm Finished in {execution_time_pso:.2f} seconds ---")
    # print(f"PSO Peak Memory Usage: {peak_mem_pso / 1024**2:.2f} MiB") 
    return gbest_position_pso, -gbest_score_pso, execution_time_pso, peak_mem_pso


# --- Firefly Algorithm (FA) Implementation ---
def run_firefly_allocation():
    print("\n--- Starting Firefly Algorithm (FA) ---")
    # tracemalloc.start() # Memory tracing disabled for performance
    start_time_fa = time.time()

    if len(RISK_SCORES_PER_ZONE) != N_AREAS or len(POPULATION_PER_ZONE) != N_AREAS: 
        raise ValueError(f"FA: Parameter array lengths (RISK_SCORES, POPULATION_PER_ZONE) must match N_AREAS ({N_AREAS}).")
    if N_AREAS == 0:
        print("FA: N_AREAS is 0. No allocation possible.")
        # current_mem, peak_mem = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        # return np.array([]), 0, 0, peak_mem 
        return np.array([]), 0, 0, 0 # Placeholder if tracemalloc is off


    firefly_positions_fa = np.zeros((N_FIREFLIES_FA, N_AREAS), dtype=int)
    if N_AREAS > 0:
        for i in range(N_FIREFLIES_FA):
            if TOTAL_PERSONNEL == 0:
                firefly_positions_fa[i] = np.zeros(N_AREAS, dtype=int)
                continue
            proportions = np.random.dirichlet(np.ones(N_AREAS))
            temp_alloc = np.zeros(N_AREAS, dtype=int)
            assigned_sum = 0
            for j in range(N_AREAS - 1):
                val = int(round(proportions[j] * TOTAL_PERSONNEL))
                temp_alloc[j] = val
                assigned_sum += val
            temp_alloc[N_AREAS - 1] = TOTAL_PERSONNEL - assigned_sum
            firefly_positions_fa[i] = repair_allocation_enforce_sum(temp_alloc,
                                                                   TOTAL_PERSONNEL,
                                                                   RISK_SCORES_PER_ZONE)
    
    light_intensities_fa = np.array([objective_function_wrapper_for_minimizers(p) for p in firefly_positions_fa])
    gbest_idx_fa = np.argmin(light_intensities_fa)
    gbest_position_fa = np.copy(firefly_positions_fa[gbest_idx_fa])
    gbest_score_fa = light_intensities_fa[gbest_idx_fa]

    print(f"FA Initial Actual Fitness (higher is better): {-gbest_score_fa:.4f}")
    print(f"FA Initial Global Best Allocation: {gbest_position_fa.astype(int)}, Sum: {np.sum(gbest_position_fa)}\n")

    for gen in range(MAX_GEN_FA):
        current_gen_positions = np.copy(firefly_positions_fa)
        current_gen_intensities = np.copy(light_intensities_fa)
        for i in range(N_FIREFLIES_FA):
            for j in range(N_FIREFLIES_FA):
                if current_gen_intensities[j] < current_gen_intensities[i]: 
                    # Calculate squared Euclidean distance only if needed (positions are integers)
                    # For integer arrays, direct subtraction and sum of squares is fine.
                    r_sq = np.sum((current_gen_positions[i].astype(float) - current_gen_positions[j].astype(float))**2)
                    beta = BETA0_FA * np.exp(-GAMMA_FA * r_sq)
                    # Generate random step for each dimension
                    random_step = ALPHA_FA * (np.random.rand(N_AREAS) - 0.5) 
                    
                    # Movement equation
                    # Ensure intermediate calculations can handle floats before repair
                    moved_position_i_float = current_gen_positions[i].astype(float) + \
                                       beta * (current_gen_positions[j].astype(float) - current_gen_positions[i].astype(float)) + \
                                       random_step
                    
                    repaired_moved_position_i = repair_allocation_enforce_sum(moved_position_i_float,
                                                                              TOTAL_PERSONNEL,
                                                                              RISK_SCORES_PER_ZONE)
                    new_intensity_i = objective_function_wrapper_for_minimizers(repaired_moved_position_i)
                    if new_intensity_i < light_intensities_fa[i]: 
                        firefly_positions_fa[i] = repaired_moved_position_i
                        light_intensities_fa[i] = new_intensity_i
                        if new_intensity_i < gbest_score_fa:
                            gbest_score_fa = new_intensity_i
                            gbest_position_fa = np.copy(repaired_moved_position_i)
        
        if (gen + 1) % 1 == 0: 
            print(f"FA Generation {gen + 1}/{MAX_GEN_FA}, "
                  f"Best Actual Fitness: {-gbest_score_fa:.4f}, "
                  f"Allocation Sum: {np.sum(gbest_position_fa)}")

    end_time_fa = time.time()
    execution_time_fa = end_time_fa - start_time_fa
    
    # current_mem_fa, peak_mem_fa = tracemalloc.get_traced_memory() 
    # tracemalloc.stop() 
    peak_mem_fa = 0 # Placeholder if tracemalloc is off

    print(f"--- Firefly Algorithm (FA) Finished in {execution_time_fa:.2f} seconds ---")
    # print(f"FA Peak Memory Usage: {peak_mem_fa / 1024**2:.2f} MiB") 
    return gbest_position_fa, -gbest_score_fa, execution_time_fa, peak_mem_fa


# --- Function to Test Your Fitness Function Independently (Optional) ---
def test_fitness_function_examples():
    print("\n--- Testing calculate_fitness with an example for the NEW (4-term) formula ---")
    test_zones = N_AREAS
    test_total_personnel = TOTAL_PERSONNEL
    # Use NumPy arrays for testing as the function now expects them
    test_assigned_np = np.array([20, 25, 30, 15, 10])
    
    test_risk_np = RISK_SCORES_PER_ZONE 
    test_population_np = POPULATION_PER_ZONE 

    print(f"Test Parameters: W1={W1}, W2={W2}, W3={W3}, W4={W4}, EPSILON={EPSILON}") 
    print(f"Test Allocation: {test_assigned_np}")
    print(f"Test Risk Scores: {test_risk_np}")
    print(f"Test Population Scores: {test_population_np}") 

    try:
        fitness_val_test = calculate_fitness(
            test_zones, test_total_personnel, test_assigned_np, test_risk_np,
            test_population_np, 
            W1, W2, W3, W4, 
            EPSILON
        )
        print(f"Test Example Fitness (New 4-term Formula): {fitness_val_test:.4f}")

        C_t = np.sum(test_assigned_np >= 1)
        CR_t = C_t / test_zones if test_zones > 0 else 0
        T1_t = W1 * CR_t

        valid_risk_mask_t = test_risk_np >= 0
        log_risk_values_t = np.zeros_like(test_risk_np, dtype=float)
        if np.any(valid_risk_mask_t):
            log_risk_values_t[valid_risk_mask_t] = np.log(1 + test_risk_np[valid_risk_mask_t])
        LRS_t = np.sum(test_assigned_np * log_risk_values_t)
        PF_t = LRS_t / test_total_personnel if test_total_personnel > 0 else 0
        T2_t = W2 * PF_t
        
        sigma_t = np.std(test_assigned_np) if test_assigned_np.size > 0 else 0
        mu_t = np.mean(test_assigned_np) if test_assigned_np.size > 0 else 0
        IC_t = sigma_t / (mu_t + EPSILON) if (mu_t + EPSILON) != 0 else 0
        T3_t = W3 * IC_t

        valid_pop_mask_t = test_population_np >= 0
        log_pop_values_t = np.zeros_like(test_population_np, dtype=float)
        if np.any(valid_pop_mask_t):
            log_pop_values_t[valid_pop_mask_t] = np.log(1 + test_population_np[valid_pop_mask_t])
        LPS_t = np.sum(test_assigned_np * log_pop_values_t)
        PC_t = LPS_t / test_total_personnel if test_total_personnel > 0 else 0 
        T4_t = W4 * PC_t 

        print(f"  Term1 (Coverage)  : {T1_t:.4f} (CR={CR_t:.2f})")
        print(f"  Term2 (Risk Alloc): {T2_t:.4f} (PF_val={PF_t:.2f}, log_sum_r={LRS_t:.2f})")
        print(f"  Term3 (Imbalance) : {T3_t:.4f} (IC_val={IC_t:.2f}, sigma={sigma_t:.2f}, mu={mu_t:.2f})")
        print(f"  Term4 (Population): {T4_t:.4f} (PC_val={PC_t:.2f}, log_sum_p={LPS_t:.2f})") 
        print(f"  Calculated Total  : {(T1_t + T2_t - T3_t + T4_t):.4f}") 

    except ValueError as e:
        print(f"Test Example Error: {e}")
    print("--- End of Fitness Function Test ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # test_fitness_function_examples() 

    print("\n" + "="*50)
    print(" RUNNING PARTICLE SWARM OPTIMIZATION (PSO) ".center(50, "="))
    print("="*50)
    pso_best_alloc, pso_best_fitness, pso_time, pso_peak_mem = run_pso_allocation() 
    
    if N_AREAS > 0 : 
        print("\nPSO FINAL RESULTS:")
        print(f"  Best Allocation: {pso_best_alloc}")
        print(f"  Best Actual Fitness (higher is better): {pso_best_fitness:.4f}")
        print(f"  Execution Time: {pso_time:.2f} seconds")
        # print(f"  Peak Memory (Python Internal): {pso_peak_mem / 1024**2:.2f} MiB") 
    else:
        print("\nPSO: No allocation performed as N_AREAS was 0.")

    print("\n" + "="*50)
    print(" RUNNING FIREFLY ALGORITHM (FA) ".center(50, "="))
    print("="*50)
    fa_best_alloc, fa_best_fitness, fa_time, fa_peak_mem = run_firefly_allocation() 

    if N_AREAS > 0 :
        print("\nFA FINAL RESULTS:")
        print(f"  Best Allocation: {fa_best_alloc}")
        print(f"  Best Actual Fitness (higher is better): {fa_best_fitness:.4f}")
        print(f"  Execution Time: {fa_time:.2f} seconds")
        # print(f"  Peak Memory (Python Internal): {fa_peak_mem / 1024**2:.2f} MiB") 
    else:
        print("\nFA: No allocation performed as N_AREAS was 0.")

    if N_AREAS > 0:
        print("\n" + "="*50)
        print(" COMPARISON SUMMARY ".center(50, "="))
        print("="*50)
        print(f"PSO Best Fitness : {pso_best_fitness:.4f} (Time: {pso_time:.2f}s)") # Mem removed for now
        print(f"FA Best Fitness  : {fa_best_fitness:.4f} (Time: {fa_time:.2f}s)")  # Mem removed for now
        if abs(pso_best_fitness - fa_best_fitness) < 1e-6 : 
             print("PSO and FA found solutions with essentially the same fitness score.")
        elif pso_best_fitness > fa_best_fitness:
            print("PSO found a better solution based on this run.")
        else: 
            print("FA found a better solution based on this run.")
        
        print("\nDetailed Breakdown for PSO Best Solution:")
        # Ensure pso_best_alloc is a NumPy array for these calculations
        pso_best_alloc_np = np.array(pso_best_alloc)
        C_pso = np.sum(pso_best_alloc_np >= 1)
        CR_pso = C_pso / N_AREAS if N_AREAS > 0 else 0.0
        
        valid_risk_mask_pso = RISK_SCORES_PER_ZONE >= 0
        log_risk_values_pso = np.zeros_like(RISK_SCORES_PER_ZONE, dtype=float)
        if np.any(valid_risk_mask_pso):
            log_risk_values_pso[valid_risk_mask_pso] = np.log(1 + RISK_SCORES_PER_ZONE[valid_risk_mask_pso])
        LRS_pso = np.sum(pso_best_alloc_np * log_risk_values_pso)
        PF_pso = LRS_pso / TOTAL_PERSONNEL if TOTAL_PERSONNEL > 0 else 0.0
        
        sigma_pso = np.std(pso_best_alloc_np) if pso_best_alloc_np.size > 0 else 0.0
        mu_pso = np.mean(pso_best_alloc_np) if pso_best_alloc_np.size > 0 else 0.0
        IC_pso = sigma_pso / (mu_pso + EPSILON) if (mu_pso + EPSILON) != 0 else 0.0
        
        valid_pop_mask_pso = POPULATION_PER_ZONE >= 0
        log_pop_values_pso = np.zeros_like(POPULATION_PER_ZONE, dtype=float)
        if np.any(valid_pop_mask_pso):
            log_pop_values_pso[valid_pop_mask_pso] = np.log(1 + POPULATION_PER_ZONE[valid_pop_mask_pso])
        LPS_pso = np.sum(pso_best_alloc_np * log_pop_values_pso)
        PC_pso = LPS_pso / TOTAL_PERSONNEL if TOTAL_PERSONNEL > 0 else 0.0 
        print(f"  T1 (Cov)  : {W1*CR_pso:.4f}, T2 (Risk): {W2*PF_pso:.4f}, T3 (Imb): {W3*IC_pso:.4f}, T4 (Pop): {W4*PC_pso:.4f}")

        print("\nDetailed Breakdown for FA Best Solution:")
        fa_best_alloc_np = np.array(fa_best_alloc)
        C_fa = np.sum(fa_best_alloc_np >= 1)
        CR_fa = C_fa / N_AREAS if N_AREAS > 0 else 0.0

        valid_risk_mask_fa = RISK_SCORES_PER_ZONE >= 0
        log_risk_values_fa = np.zeros_like(RISK_SCORES_PER_ZONE, dtype=float)
        if np.any(valid_risk_mask_fa):
            log_risk_values_fa[valid_risk_mask_fa] = np.log(1 + RISK_SCORES_PER_ZONE[valid_risk_mask_fa])
        LRS_fa = np.sum(fa_best_alloc_np * log_risk_values_fa)
        PF_fa = LRS_fa / TOTAL_PERSONNEL if TOTAL_PERSONNEL > 0 else 0.0

        sigma_fa = np.std(fa_best_alloc_np) if fa_best_alloc_np.size > 0 else 0.0
        mu_fa = np.mean(fa_best_alloc_np) if fa_best_alloc_np.size > 0 else 0.0
        IC_fa = sigma_fa / (mu_fa + EPSILON) if (mu_fa + EPSILON) != 0 else 0.0

        valid_pop_mask_fa = POPULATION_PER_ZONE >= 0
        log_pop_values_fa = np.zeros_like(POPULATION_PER_ZONE, dtype=float)
        if np.any(valid_pop_mask_fa):
            log_pop_values_fa[valid_pop_mask_fa] = np.log(1 + POPULATION_PER_ZONE[valid_pop_mask_fa])
        LPS_fa = np.sum(fa_best_alloc_np * log_pop_values_fa)
        PC_fa = LPS_fa / TOTAL_PERSONNEL if TOTAL_PERSONNEL > 0 else 0.0 
        print(f"  T1 (Cov)  : {W1*CR_fa:.4f}, T2 (Risk): {W2*PF_fa:.4f}, T3 (Imb): {W3*IC_fa:.4f}, T4 (Pop): {W4*PC_fa:.4f}")

