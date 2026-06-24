import numpy as np

class AtomInterferometrySim:
    r"""
    Simulation for Atom Interferometry capable of detecting
    micro-gravitational gradients (\Delta g/g \approx 10^{-10}).
    """
    def __init__(self, sensitivity=1e-10):
        self.sensitivity = sensitivity
        self.base_g = 9.80665 # m/s^2

    def detect_gradient(self, simulated_g2, simulated_g1, distance):
        """
        Calculates the gravitational gradient over a distance.
        """
        delta_g = abs(simulated_g2 - simulated_g1)
        relative_gradient = delta_g / self.base_g

        if relative_gradient >= self.sensitivity:
            return True, relative_gradient
        return False, relative_gradient

class LenseThirringFilter:
    """
    Filter for Gravitomagnetic frame-dragging effects.
    """
    def __init__(self, angular_momentum, earth_radius=6371000):
        self.J = angular_momentum
        self.G = 6.67430e-11
        self.c = 3.0e8
        self.R = earth_radius

    def apply_filter(self, raw_signal):
        r"""
        Isolates the Lense-Thirring precession from background noise.
        \Omega_{LT} = \frac{G J}{c^2 r^3}
        """
        # simplified theoretical angular velocity simulation
        omega_lt = (self.G * self.J) / (self.c**2 * self.R**3)
        return raw_signal - omega_lt

