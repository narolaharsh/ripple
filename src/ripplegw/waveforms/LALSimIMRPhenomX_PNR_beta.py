import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from .LALSimIMRPhenomX_PNR_coefficients import (
    IMRPhenomX_PNR_beta_B0_coefficient,
    IMRPhenomX_PNR_beta_B1_coefficient,
    IMRPhenomX_PNR_beta_B2_coefficient,
    IMRPhenomX_PNR_beta_B3_coefficient,
    IMRPhenomX_PNR_beta_B4_coefficient,
    IMRPhenomX_PNR_beta_B5_coefficient,
)


@pytree_dataclass
class IMRPhenomX_PNR_beta_parameters:
    """
    Parameter structure for beta angle coefficients.
    Reference: Sec. 8D in arXiv:2107.08876
    """
    B0: float
    B1: float
    B2: float
    B3: float
    B4: float
    B5: float


def IMRPhenomX_PNR_precompute_beta_coefficients(
    pWF: dict,
    pPrec
) -> IMRPhenomX_PNR_beta_parameters:
    """
    Precompute beta coefficients for PNR angles.

    JAX-friendly implementation without if-else branches.
    Reference: Sec. 8D in arXiv:2107.08876

    Parameters
    ----------
    pWF : dict
        PhenomX waveform struct
    pPrec : IMRPhenomXGetAndSetPrecessionVariables
        PhenomX precession struct

    Returns
    -------
    IMRPhenomX_PNR_beta_parameters
        Beta parameter struct with B0, B1, B2, B3, B4, B5 coefficients
    """

    # Determine eta based on precession version
    # If version==330 and eta < 0.09, use 0.09; otherwise use pWF['eta']
    eta_330 = jnp.where(pWF['eta'] >= 0.09, pPrec.eta, 0.09)
    eta = jnp.where(pPrec.IMRPhenomXPrecVersion == 330, eta_330, pWF['eta'])

    # Compute chi boundary
    chiboundary = 0.80 - 0.20 * jnp.exp(-jnp.power((pWF['q'] - 6.0) / 1.5, 8))

    # Determine chi based on precession version
    # If version==330 and chi_singleSpin > chiboundary, use chiboundary
    chi_330 = jnp.where(pPrec.chi_singleSpin <= chiboundary, pPrec.chi_singleSpin, chiboundary)
    chi = jnp.where(pPrec.IMRPhenomXPrecVersion == 330, chi_330, pPrec.chi_singleSpin)

    costheta = pPrec.costheta_singleSpin

    # Approximate orientation of final spin
    costhetaf = pPrec.costheta_final_singleSpin

    # Compute B4 and ensure it's sufficiently large (>= 175.0)
    B4_raw = IMRPhenomX_PNR_beta_B4_coefficient(eta, chi, costheta)
    B4 = jnp.where(B4_raw <= 175.0, 175.0, B4_raw)

    # Compute all beta coefficients
    B0 = jnp.arccos(costhetaf) - IMRPhenomX_PNR_beta_B0_coefficient(eta, chi, costheta)
    B1 = IMRPhenomX_PNR_beta_B1_coefficient(eta, chi, costheta)
    B2 = IMRPhenomX_PNR_beta_B2_coefficient(eta, chi, costheta)
    B3_coeff = IMRPhenomX_PNR_beta_B3_coefficient(eta, chi, costheta)
    B3 = B2 * B3_coeff
    B5 = IMRPhenomX_PNR_beta_B5_coefficient(eta, chi, costheta)

    return IMRPhenomX_PNR_beta_parameters(
        B0=B0,
        B1=B1,
        B2=B2,
        B3=B3,
        B4=B4,
        B5=B5
    )
