import numpy as np
import sympy as sp

class AlcubierreMetric:
    r"""
    Calculates the Alcubierre Metric line element numerically.
    ds^2 = -c^2 dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2
    """
    def __init__(self, c=3.0e8, G=6.67430e-11):
        self.c = c
        self.G = G

    def form_function(self, r_s, R, sigma):
        """
        Shaping function f(r_s)
        """
        return (np.tanh(sigma * (r_s + R)) - np.tanh(sigma * (r_s - R))) / (2 * np.tanh(sigma * R))

    def evaluate_line_element(self, dt, dx, dy, dz, v_s, r_s, R, sigma):
        """
        Evaluates the metric line element mathematically.
        """
        f_rs = self.form_function(r_s, R, sigma)
        ds2 = -(self.c**2) * (dt**2) + (dx - v_s * f_rs * dt)**2 + dy**2 + dz**2
        return ds2

class EFESolver:
    r"""
    Solver for the Einstein Field Equations:
    G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}
    """
    def __init__(self):
        # Symbolic definitions
        self.t, self.x, self.y, self.z = sp.symbols('t x y z', real=True)
        self.c, self.G, self.v_s = sp.symbols('c G v_s', real=True, positive=True)
        
        # We model f as an arbitrary function of space and time to evaluate the metric symbolically
        self.f = sp.Function('f')(self.t, self.x, self.y, self.z)

        # Coordinates tuple
        self.coords = (self.t, self.x, self.y, self.z)

    def get_alcubierre_metric_tensor(self):
        r"""Returns covariant metric tensor g_{\mu\nu}"""
        g = sp.zeros(4, 4)
        g[0, 0] = -self.c**2 + self.v_s**2 * self.f**2
        g[0, 1] = g[1, 0] = -self.v_s * self.f
        g[1, 1] = 1
        g[2, 2] = 1
        g[3, 3] = 1
        return g
        
    def get_inverse_metric(self, g):
        r"""Returns contravariant metric tensor g^{\mu\nu}"""
        return g.inv()

    def get_christoffel_symbols(self, g, g_inv):
        r"""
        \Gamma^\rho_{\mu\nu} = \frac{1}{2} g^{\rho\sigma} (\partial_\mu g_{\sigma\nu} + \partial_\nu g_{\sigma\mu} - \partial_\sigma g_{\mu\nu})
        """
        Gamma = sp.MutableDenseNDimArray.zeros(4, 4, 4)
        for rho in range(4):
            for mu in range(4):
                for nu in range(4):
                    term = 0
                    for sigma in range(4):
                        if g_inv[rho, sigma] != 0:
                            term += 0.5 * g_inv[rho, sigma] * (
                                sp.diff(g[sigma, nu], self.coords[mu]) +
                                sp.diff(g[sigma, mu], self.coords[nu]) -
                                sp.diff(g[mu, nu], self.coords[sigma])
                            )
                    # Expand instead of simplify for speed on abstract functions
                    Gamma[rho, mu, nu] = term.expand()
        return Gamma

    def get_ricci_tensor_and_scalar(self, Gamma, g_inv):
        r"""
        R_{\mu\nu} = \partial_\rho \Gamma^\rho_{\mu\nu} - \partial_\nu \Gamma^\rho_{\mu\rho} + \Gamma^\rho_{\mu\nu}\Gamma^\lambda_{\rho\lambda} - \Gamma^\lambda_{\mu\rho}\Gamma^\rho_{\nu\lambda}
        R = g^{\mu\nu} R_{\mu\nu}
        """
        R_mu_nu = sp.zeros(4, 4)
        for mu in range(4):
            for nu in range(4):
                term1 = sum(sp.diff(Gamma[rho, mu, nu], self.coords[rho]) for rho in range(4))
                term2 = sum(sp.diff(Gamma[rho, mu, rho], self.coords[nu]) for rho in range(4))
                term3 = sum(Gamma[rho, mu, nu] * Gamma[lambda_, rho, lambda_] for rho in range(4) for lambda_ in range(4))
                term4 = sum(Gamma[lambda_, mu, rho] * Gamma[rho, nu, lambda_] for rho in range(4) for lambda_ in range(4))
                R_mu_nu[mu, nu] = (term1 - term2 + term3 - term4).expand()
                
        R_scalar = sum(g_inv[mu, nu] * R_mu_nu[mu, nu] for mu in range(4) for nu in range(4)).expand()
        return R_mu_nu, R_scalar

    def get_einstein_tensor(self, R_mu_nu, R_scalar, g):
        r"""
        G_{\mu\nu} = R_{\mu\nu} - {1\over 2} g_{\mu\nu} R
        """
        G_tensor = sp.zeros(4, 4)
        for mu in range(4):
            for nu in range(4):
                G_tensor[mu, nu] = (R_mu_nu[mu, nu] - 0.5 * g[mu, nu] * R_scalar).expand()
        return G_tensor
        
    def get_stress_energy_tensor(self, G_tensor):
        r"""
        T_{\mu\nu} = \frac{c^4}{8\pi G} G_{\mu\nu}
        """
        return (self.c**4 / (8 * sp.pi * self.G)) * G_tensor

    def check_nec_violation(self, T_mu_nu, u_mu, u_nu):
        r"""
        Checks for Null Energy Condition (NEC) violations.
        T_{\mu\nu} u^\mu u^\nu < 0
        """
        # Kept for numeric legacy code compatibility
        contraction = np.einsum('ij,i,j->', T_mu_nu, u_mu, u_nu)
        if contraction < 0:
            return True, contraction
        return False, contraction
