import jax.numpy as jnp


def deltaF_mergerBin(fdamp: float, alpha4: float, abserror: float) -> float:
    """
    Right hand side of eq. 2.27 in arXiv:2001.10897.

    Calculates the frequency bin size for the merger region.

    Args:
        fdamp: Damping frequency (float)
        alpha4: Alpha4 coefficient (float)
        abserror: Absolute error tolerance (float)

    Returns:
        float: Frequency bin size for merger
    """
    aux = jnp.sqrt(jnp.sqrt(3.0) * 3.0)
    return 4.0 * fdamp * jnp.sqrt(abserror / jnp.abs(alpha4)) / aux


def deltaF_ringdownBin(fdamp: float, alpha4: float, LAMBDA: float, abserror: float) -> float:
    """
    Correspond to eqs. 2.28 and 2.31 in arXiv:2001.10897.

    Calculates the frequency bin size for the ringdown region based on
    phase and amplitude contributions, returning the minimum.

    Args:
        fdamp: Damping frequency (float)
        alpha4: Alpha4 coefficient (float)
        LAMBDA: Lambda parameter (float)
        abserror: Absolute error tolerance (float)

    Returns:
        float: Frequency bin size for ringdown (minimum of phase and amplitude contributions)
    """
    dfphase = 5.0 * fdamp * jnp.sqrt(abserror * 0.5 / jnp.abs(alpha4))
    dfamp = jnp.sqrt(2.0 * abserror) / jnp.abs(LAMBDA)

    return jnp.where(dfphase <= dfamp, dfphase, dfamp)
