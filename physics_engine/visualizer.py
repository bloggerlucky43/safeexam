import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for server deployment without GUI
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

def generate_energy_density_plot(output_dir="models", filename="metric_plot.png"):
    """
    Numerically evaluates the T^{00} energy density based on the explicit Phase 2 analytical solution.
    Generates a 3D structural plot mapped across the localized spatial cross-section.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, filename)
    
    # Physics constants
    v_s_val = 3.0e8   # 1c
    G_val = 6.67430e-11
    R_val = 50.0      # Bubble radius
    sigma_val = 8.0   # Shell thickness scaling
    
    # 2D Grid mapping (y, z coordinates orthogonal to travel direction x)
    # Using float64 precision to prevent underflow
    Y, Z = np.meshgrid(np.linspace(-100, 100, 200), np.linspace(-100, 100, 200))
    rs_grid = np.sqrt(Y**2 + Z**2)
    
    # Avoiding division by zero at origin (rs=0 limit)
    rs_grid[rs_grid == 0] = 1e-10 
    
    # Analytical derivative df/dr_s for the top-hat cutoff function f(r_s)
    def df_dr(rs):
        norm = 2 * np.tanh(sigma_val * R_val)
        # Using cosh for the derivative of tanh
        term1 = sigma_val / (np.cosh(sigma_val * (rs + R_val))**2)
        term2 = sigma_val / (np.cosh(sigma_val * (rs - R_val))**2)
        return (term1 - term2) / norm
        
    d_f = df_dr(rs_grid)
    
    # Evaluates T^{00} = - (v_s^2 / (32 * pi * G)) * ( (df/dy)^2 + (df/dz)^2 )
    # Notice the explicit negative multiplier mapping to the NEC violation requirement 
    T_00 = -(v_s_val**2 / (32 * np.pi * G_val)) * (d_f**2)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the Exotic Matter localized ring
    surf = ax.plot_surface(Y, Z, T_00, cmap='magma', edgecolor='none', alpha=0.9)
    ax.set_title("Alcubierre Metric: Negative Energy Density Structural Ring", pad=20)
    ax.set_xlabel("y / meters")
    ax.set_ylabel("z / meters")
    ax.set_zlabel("T^{00} Density Extent")
    
    # Customize the visual for a premium mathematical representation
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    fig.colorbar(surf, shrink=0.4, aspect=10, label="Energy Density Gradient")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_plotly_energy_density(v_s_val=3.0e8, R_val=50.0, sigma_val=8.0):
    """
    Dynamically generates an interactive Plotly 3D HTML figure for Streamlit.
    """
    G_val = 6.67430e-11
    
    Y, Z = np.meshgrid(np.linspace(-100, 100, 200), np.linspace(-100, 100, 200))
    rs_grid = np.sqrt(Y**2 + Z**2)
    rs_grid[rs_grid == 0] = 1e-10 
    
    def df_dr(rs):
        norm = 2 * np.tanh(sigma_val * R_val)
        term1 = sigma_val / (np.cosh(sigma_val * (rs + R_val))**2)
        term2 = sigma_val / (np.cosh(sigma_val * (rs - R_val))**2)
        return (term1 - term2) / norm
        
    d_f = df_dr(rs_grid)
    T_00 = -(v_s_val**2 / (32 * np.pi * G_val)) * (d_f**2)
    
    fig = go.Figure(data=[go.Surface(z=T_00, x=Y, y=Z, colorscale='Magma')])
    fig.update_layout(
        title='Alcubierre Metric: Negative Energy Density Structure',
        scene=dict(
            xaxis_title='y (meters)',
            yaxis_title='z (meters)',
            zaxis_title='T^{00} (J/m^3)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
