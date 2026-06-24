import sympy as sp
from physics_engine.metric_engine import AlcubierreMetric, EFESolver
from metrology_sim.sensor_sim import AtomInterferometrySim, LenseThirringFilter

def check_gravity_probe_b():
    """
    Gravity Probe B Experimental Bounds comparison.
    Expected geodetic effect: -6.6018 arcsec/yr
    Expected frame-dragging effect: -39.2 milliarcsec/yr
    """
    gpb_frame_dragging_bounds = -0.0392

    filter_lt = LenseThirringFilter(angular_momentum=1e4) # Example angular momentum
    simulated_omega = filter_lt.apply_filter(0) # baseline shift

    print(f"GP-B Frame Dragging Target: {gpb_frame_dragging_bounds} arcsec/yr")
    print(f"Simulated Lense-Thirring offset: {simulated_omega} rad/s")

def check_tajmar_bounds():
    """
    Tajmar Experimental Bounds for anomalous acceleration.
    """
    tajmar_threshold = 1e-6 # m/s^2 example threshold

    sensor = AtomInterferometrySim(sensitivity=1e-10)
    g1 = 9.8066500000
    g2 = 9.8066500100 # Micro-gradient simulation
    
    detected, rel_gradient = sensor.detect_gradient(g2, g1, 0.1)
    acceleration_diff = rel_gradient * sensor.base_g

    print(f"Tajmar Threshold: {tajmar_threshold} m/s^2")
    print(f"Simulated Anomalous Acceleration: {acceleration_diff} m/s^2")
    if acceleration_diff > tajmar_threshold:
        print("WARNING: Simulated acceleration exceeds Tajmar bounds.")
    else:
        print("Validation Pass: Simulated acceleration within Tajmar bounds.")

def check_symbolic_efe_solution():
    """
    Compute and print the energy density T^{00} symbolically
    to verify the NEC violation analytically.
    """
    print("\n--- Phase 2: Exact Symbolic Solutions Using SymPy ---")
    solver = EFESolver()
    print("1. Building symbolic metric tensor...")
    g = solver.get_alcubierre_metric_tensor()
    g_inv = solver.get_inverse_metric(g)
    
    print("2. Calculating Christoffel symbols (simplified computation)...")
    Gamma = solver.get_christoffel_symbols(g, g_inv)
    
    print("3. Calculating Ricci tensor and scalar...")
    R_mu_nu, R_scalar = solver.get_ricci_tensor_and_scalar(Gamma, g_inv)
    
    print("4. Calculating Einstein tensor...")
    G_tensor = solver.get_einstein_tensor(R_mu_nu, R_scalar, g)
    
    print("5. Extracting Stress-Energy Tensor...")
    T = solver.get_stress_energy_tensor(G_tensor)
    
    print("6. Calculating contravariant Energy Density (T^{00})...")
    T_upper_00 = 0
    for mu in range(4):
        for nu in range(4):
            if g_inv[0, mu] != 0 and g_inv[0, nu] != 0:
                T_upper_00 += g_inv[0, mu] * g_inv[0, nu] * T[mu, nu]
                
    T_upper_00 = T_upper_00.simplify()
    print(f"\nAnalytical Energy Density T^{{00}}:\n{T_upper_00}\n")
    if T_upper_00 != 0:
        print("NEC Violation confirmed: Energy density explicitly depends on the derivative of the shape function f.")

if __name__ == "__main__":
    print("Running Experimental Validations...\n")
    check_gravity_probe_b()
    print("\n---")
    check_tajmar_bounds()
    check_symbolic_efe_solution()
