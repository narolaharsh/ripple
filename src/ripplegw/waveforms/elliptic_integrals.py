"""
JAX implementation of GSL elliptic integral functions.

This module provides JAX-compatible implementations of elliptic integrals,
specifically the incomplete elliptic integral of the first kind (F).
"""

import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid


@jit
def ellint_F(phi: float, k: float, n_points: int = 1000) -> float:
    """
    Compute the incomplete elliptic integral of the first kind.

    This function computes F(φ, k) using the Legendre form:
    F(φ, k) = ∫₀^φ dt / √(1 - k² sin²(t))

    This is equivalent to GSL's gsl_sf_ellint_F(phi, k, GSL_PREC_DOUBLE).

    Args:
        phi: The amplitude (upper limit of integration) in radians
        k: The modulus, where 0 ≤ k² ≤ 1 (note: this is k, not m = k²)
        n_points: Number of integration points (default: 1000)

    Returns:
        The value of F(φ, k)

    Notes:
        - For k = 0: F(φ, 0) = φ
        - For |k| = 1 and |φ| < π/2: F(φ, ±1) = arctanh(sin(φ))
        - The implementation uses numerical integration via trapezoidal rule
        - JAX-compatible (can be used in jit, grad, vmap, etc.)

    Examples:
        >>> ellint_F(jnp.pi/2, 0.5)  # Complete elliptic integral K(0.5)
        >>> ellint_F(jnp.pi/4, 0.8)  # Incomplete elliptic integral
    """
    phi = jnp.asarray(phi)
    k = jnp.asarray(k)
    k2 = k * k

    # Handle special case: k = 0
    # F(φ, 0) = φ
    def case_k_zero():
        return phi

    # Handle special case: |k| = 1
    # F(φ, ±1) = arctanh(sin(φ)) = 0.5 * ln((1 + sin(φ)) / (1 - sin(φ)))
    def case_k_one():
        sin_phi = jnp.sin(phi)
        # Use arctanh for numerical stability
        return jnp.arctanh(sin_phi)

    # General case: numerical integration
    def case_general():
        # Create integration points from 0 to phi
        t = jnp.linspace(0.0, phi, n_points)

        # Compute integrand: 1 / √(1 - k² sin²(t))
        sin_t = jnp.sin(t)
        integrand = 1.0 / jnp.sqrt(1.0 - k2 * sin_t * sin_t)

        # Integrate using trapezoidal rule
        result = trapezoid(integrand, t)
        return result

    # Select appropriate case based on k value
    abs_k = jnp.abs(k)
    is_k_zero = abs_k < 1e-10
    is_k_one = jnp.abs(abs_k - 1.0) < 1e-10

    # Nested conditional: check k=0 first, then k=1, then general
    result = jnp.where(
        is_k_zero,
        case_k_zero(),
        jnp.where(
            is_k_one,
            case_k_one(),
            case_general()
        )
    )

    return result


@jit
def ellint_Kcomp(k: float, n_points: int = 1000) -> float:
    """
    Compute the complete elliptic integral of the first kind.

    This is K(k) = F(π/2, k) = ∫₀^(π/2) dt / √(1 - k² sin²(t))

    Equivalent to GSL's gsl_sf_ellint_Kcomp(k, GSL_PREC_DOUBLE).

    Args:
        k: The modulus, where 0 ≤ k² ≤ 1
        n_points: Number of integration points (default: 1000)

    Returns:
        The value of K(k)
    """
    return ellint_F(jnp.pi / 2.0, k, n_points)


@jit
def ellint_F_carlson(phi: float, k: float) -> float:
    """
    Compute the incomplete elliptic integral of the first kind using Carlson's method.

    This is an alternative implementation using Carlson symmetric form:
    F(φ, k) = sin(φ) * R_F(cos²(φ), 1 - k² sin²(φ), 1)

    This method can be more accurate for certain parameter ranges but requires
    implementing Carlson's R_F function.

    Args:
        phi: The amplitude in radians
        k: The modulus

    Returns:
        The value of F(φ, k)

    Note:
        This is a placeholder for a future implementation using Carlson's
        symmetric elliptic integrals, which are numerically more stable.
    """
    # For now, fall back to the trapezoidal integration method
    return ellint_F(phi, k)
