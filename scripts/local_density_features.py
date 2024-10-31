import numpy as np

def calculate_quantum_features(n_up, n_down, phi_orbitals, coordinates):
    """
    Calculate quantum mechanical parameters based on electron densities and orbitals.
    
    Parameters:
    -----------
    n_up : numpy.ndarray
        Spin-up electron density
    n_down : numpy.ndarray
        Spin-down electron density
    phi_orbitals : list of numpy.ndarray
        List of occupied orbital wavefunctions
    coordinates : numpy.ndarray
        Grid coordinates for calculating gradients
        
    Returns:
    --------
    dict
        Dictionary containing calculated parameters:
        - n_s: Reduced density
        - zeta: Spin polarization
        - s: Reduced density gradient
        - tau: Kinetic energy density
    """
    # Total density
    n = n_up + n_down
    
    # Calculate n_s (reduced density)
    n_s = np.power(n, 1/3)
    
    # Calculate zeta (spin polarization)
    zeta = np.divide(n_up - n_down, n, out=np.zeros_like(n), where=n!=0)
    
    # Calculate s (reduced density gradient)
    grad_n = np.gradient(n, coordinates)
    grad_n_magnitude = np.sqrt(sum(x*x for x in grad_n))
    s = np.divide(grad_n_magnitude, np.power(n, 4/3), out=np.zeros_like(n), where=n!=0)
    
    # Calculate tau (kinetic energy density)
    tau = np.zeros_like(n)
    for phi in phi_orbitals:
        grad_phi = np.gradient(phi, coordinates)
        grad_phi_squared = sum(x*x for x in grad_phi)
        tau += 0.5 * grad_phi_squared
    
    return {
        'n_s': n_s,
        'zeta': zeta,
        's': s,
        'tau': tau
    }

# Example usage
def create_example():
    """
    Create an example calculation using a simple 1D r-coordinates system.
    """
    # Create a 1D grid
    x = np.linspace(-5, 5, 100)
    
    # Create example densities (Gaussian distributions)
    n_up = np.exp(-x**2)
    n_down = 0.8 * np.exp(-x**2)
    
    # Create example orbitals (normalized Gaussian functions)
    phi1 = np.sqrt(1/np.sqrt(np.pi)) * np.exp(-x**2/2)
    phi2 = np.sqrt(1/np.sqrt(np.pi)) * x * np.exp(-x**2/2)
    phi_orbitals = [phi1, phi2]
    
    # Calculate parameters
    results = calculate_quantum_features(n_up, n_down, phi_orbitals, x)
    
    return x, results

if __name__ == "__main__":
    # Run example calculation
    x, results = create_example()
    
    # Print some results
    print("Example results at x=0:")
    for param, value in results.items():
        print(f"{param}: {value[50]:.6f}")  # Value at middle of grid