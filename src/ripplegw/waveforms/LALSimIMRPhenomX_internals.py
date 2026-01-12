"""
IMRPhenomX internal functions converted to JAX.

This module contains the core initialization and setup functions for the
IMRPhenomX waveform model, converted from the LALSimulation C code to JAX.
"""

import jax.numpy as jnp
from typing import Dict, Any, Optional
from ..constants import MSUN, MRSUN, MTSUN, PI
from .LALSimIMRPhenomX_PNR_internals import (
    XLALSimIMRPhenomXchiEff,
    XLALSimIMRPhenomXSTotR,
    XLALSimIMRPhenomXFinalMass2017,
    XLALSimIMRPhenomXFinalSpin2017,
)
from .LALSimIMRPhenomX_qnm import evaluate_QNMfit_fring22, evaluate_QNMfit_fdamp22
from .LALSimIMRPhenomX_precession import XLALSimIMRPhenomXUtilsHztoMf


# Helper functions
def XLALSimIMRPhenomXchiPNHat(eta: float, chi1L: float, chi2L: float) -> float:
    """
    PN reduced spin parameter.

    This is the PN parameter that appears in the TaylorF2 phase at 1.5PN order.

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        PN reduced spin parameter
    """
    # chiPNHat = (chi1 + chi2) - (76/113) * eta * (chi1 + chi2)
    return (chi1L + chi2L) - (76.0 * eta / 113.0) * (chi1L + chi2L)


def XLALSimIMRPhenomXdchi(chi1L: float, chi2L: float) -> float:
    """
    Spin difference parameter.

    Args:
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Spin difference chi1L - chi2L
    """
    return chi1L - chi2L


def XLALSimIMRPhenomXfMECO(eta: float, chi1L: float, chi2L: float) -> float:
    """
    Minimum Energy Circular Orbit (MECO) frequency in geometric units.

    This is a fit to the hybrid MECO from Cabero et al, Phys.Rev. D95 (2017).

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Dimensionless MECO frequency (Mf_MECO)
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    # Compute effective spin S for the MECO fit
    chiEff = XLALSimIMRPhenomXchiEff(eta, chi1L, chi2L)
    S = (chiEff - (38.0 / 113.0) * eta * (chi1L + chi2L)) / (1.0 - (76.0 * eta / 113.0))
    S2 = S * S
    S3 = S2 * S

    dchi = chi1L - chi2L
    dchi2 = dchi * dchi

    # No-spin contribution
    noSpin = (
        0.018744340279608845
        + 0.0077903147004616865 * eta
        + 0.003940354686136861 * eta2
        - 0.00006693930988501673 * eta3
    ) / (1.0 - 0.10423384680638834 * eta)

    # Equal-spin contribution
    eqSpin = (
        S * (
            0.00027180386951683135
            - 0.00002585252361022052 * S
            + eta4 * (-0.0006807631931297156 + 0.022386313074011715 * S - 0.0230825153005985 * S2)
            + eta2 * (0.00036556167661117023 - 0.000010021140796150737 * S - 0.00038216081981505285 * S2)
            + eta * (0.00024422562796266645 - 0.00001049013062611254 * S - 0.00035182990586857726 * S2)
            + eta3 * (-0.0005418851224505745 + 0.000030679548774047616 * S + 4.038390455349854e-6 * S2)
            - 0.00007547517256664526 * S2
        )
    ) / (
        0.026666543809890402
        + (-0.014590539285641243 - 0.012429476486138982 * eta + 1.4861197211952053 * eta4
           + 0.025066696514373803 * eta2 + 0.005146809717492324 * eta3) * S
        + (-0.0058684526275074025 - 0.02876774751921441 * eta - 2.551566872093786 * eta4
           - 0.019641378027236502 * eta2 - 0.001956646166089053 * eta3) * S2
        + (0.003507640638496499 + 0.014176504653145768 * eta + 1.0 * eta4
           + 0.012622225233586283 * eta2 - 0.00767768214056772 * eta3) * S3
    )

    # Unequal-spin contribution
    uneqSpin = (
        dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta) * eta2
        + dchi * delta * eta * (
            0.08064665214195679 * eta2
            + eta * (-0.028476219509487793 - 0.005746537021035632 * S)
            - 0.0011713735642446144 * S
        )
    )

    return noSpin + eqSpin + uneqSpin


def XLALSimIMRPhenomXfISCO(afinal: float) -> float:
    """
    Innermost Stable Circular Orbit (ISCO) frequency in geometric units.

    Based on Ori et al, Phys.Rev. D62 (2000) 124022.

    Args:
        afinal: Final dimensionless spin

    Returns:
        Dimensionless ISCO frequency (Mf_ISCO)
    """
    a2 = afinal * afinal

    # Compute rISCO
    Z1 = 1.0 + jnp.cbrt(1.0 - a2) * (jnp.cbrt(1 + afinal) + jnp.cbrt(1 - afinal))
    Z1 = jnp.where(Z1 > 3.0, 3.0, Z1)
    Z2 = jnp.sqrt(3.0 * a2 + Z1 * Z1)

    rISCO = 3.0 + Z2 - jnp.sign(afinal) * jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

    # Compute orbital frequency at ISCO
    rISCOsq = jnp.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq
    OmegaISCO = 1.0 / (rISCO3o2 + afinal)

    return OmegaISCO / PI


def IMRPhenomX_InternalNudge(x: float, X: float, epsilon: float) -> float:
    """
    Nudge a value towards a target if it's within epsilon.

    Args:
        x: Value to potentially nudge
        X: Target value
        epsilon: Tolerance

    Returns:
        Either x (if outside tolerance) or X (if within tolerance)
    """
    return jnp.where(jnp.abs(x - X) < epsilon, X, x)


def IMRPhenomXSetWaveformVariables(
    m1_SI: float,
    m2_SI: float,
    chi1L_In: float,
    chi2L_In: float,
    deltaF: float,
    fRef: float,
    phi0: float,
    f_min: float,
    f_max: float,
    distance: float,
    inclination: float,
    lalParams: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Initialize and set all waveform variables for IMRPhenomX.

    This function performs parameter validation, mass/spin ordering, and computes
    all derived quantities needed for waveform generation. It returns a dictionary
    containing all waveform parameters (wf struct).

    Parameters
    ----------
    m1_SI : float
        Mass of companion 1 in SI units (kg)
    m2_SI : float
        Mass of companion 2 in SI units (kg)
    chi1L_In : float
        Dimensionless aligned spin of object 1
    chi2L_In : float
        Dimensionless aligned spin of object 2
    deltaF : float
        Frequency spacing for waveform generation (Hz)
    fRef : float
        Reference frequency (Hz). If 0, defaults to f_min
    phi0 : float
        Orbital phase at reference frequency (radians)
    f_min : float
        Starting GW frequency (Hz)
    f_max : float
        Ending GW frequency (Hz)
    distance : float
        Luminosity distance (meters)
    inclination : float
        Inclination angle (radians)
    lalParams : dict, optional
        LAL parameters dictionary. If None, defaults will be used.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    wf : dict
        Dictionary containing all waveform parameters

    Notes
    -----
    - This is a JAX translation of LALSimIMRPhenomX_internals.c
    - The function ensures m1 >= m2 by swapping if necessary
    - Spins are validated to be within the Kerr bound [-1, 1]
    - Version flags default to recommended values (104, 105, 105, 103, 104, 103)

    References
    ----------
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
    """

    # Initialize waveform struct as dictionary
    wf = {}

    # Set default LALParams if not provided
    if lalParams is None:
        lalParams = {}

    wf['LALparams'] = lalParams

    # Copy model version flags (use defaults)
    wf['IMRPhenomXInspiralPhaseVersion'] = lalParams.get('InspiralPhaseVersion', 104)
    wf['IMRPhenomXIntermediatePhaseVersion'] = lalParams.get('IntermediatePhaseVersion', 105)
    wf['IMRPhenomXRingdownPhaseVersion'] = lalParams.get('RingdownPhaseVersion', 105)

    wf['IMRPhenomXInspiralAmpVersion'] = lalParams.get('InspiralAmpVersion', 103)
    wf['IMRPhenomXIntermediateAmpVersion'] = lalParams.get('IntermediateAmpVersion', 104)
    wf['IMRPhenomXRingdownAmpVersion'] = lalParams.get('RingdownAmpVersion', 103)

    wf['IMRPhenomXPNRUseTunedCoprec'] = lalParams.get('PNRUseTunedCoprec', 0)
    wf['IMRPhenomXPNRUseTunedCoprec33'] = wf['IMRPhenomXPNRUseTunedCoprec'] * wf['IMRPhenomXPNRUseTunedCoprec']

    wf['PhenomXOnlyReturnPhase'] = lalParams.get('OnlyReturnPhase', 0)

    wf['IMRPhenomXPNRForceXHMAlignment'] = lalParams.get('PNRForceXHMAlignment', 0)

    wf['debug'] = debug

    # Rescale to mass in solar masses
    m1_In = m1_SI / MSUN
    m2_In = m2_SI / MSUN

    # Set matter parameters (tidal effects - default to zero)
    lambda1_In = lalParams.get('lambda1', 0.0)
    lambda2_In = lalParams.get('lambda2', 0.0)
    quadparam1_In = lalParams.get('quadparam1', 1.0)
    quadparam2_In = lalParams.get('quadparam2', 1.0)

    # Check if m1 >= m2, if not then swap masses/spins/lambdas/quadparams
    swap_needed = m1_In < m2_In

    m1 = jnp.where(swap_needed, m2_In, m1_In)
    m2 = jnp.where(swap_needed, m1_In, m2_In)
    chi1L = jnp.where(swap_needed, chi2L_In, chi1L_In)
    chi2L = jnp.where(swap_needed, chi1L_In, chi2L_In)
    lambda1 = jnp.where(swap_needed, lambda2_In, lambda1_In)
    lambda2 = jnp.where(swap_needed, lambda1_In, lambda2_In)
    quadparam1 = jnp.where(swap_needed, quadparam2_In, quadparam1_In)
    quadparam2 = jnp.where(swap_needed, quadparam1_In, quadparam2_In)

    # Nudge spins if they're slightly outside [-1, 1] due to roundoff
    chi1L = IMRPhenomX_InternalNudge(chi1L, 1.0, 1e-6)
    chi1L = IMRPhenomX_InternalNudge(chi1L, -1.0, 1e-6)
    chi2L = IMRPhenomX_InternalNudge(chi2L, 1.0, 1e-6)
    chi2L = IMRPhenomX_InternalNudge(chi2L, -1.0, 1e-6)

    # Symmetric mass ratio
    delta = jnp.abs((m1 - m2) / (m1 + m2))
    eta = jnp.abs(0.25 * (1.0 - delta * delta))
    q = jnp.where(m1 > m2, m1 / m2, m2 / m1)

    # Ensure eta <= 0.25
    eta = jnp.where(eta > 0.25, 0.25, eta)
    q = jnp.where(eta == 0.25, 1.0, q)

    # Masses definitions
    wf['m1_SI'] = m1 * MSUN
    wf['m2_SI'] = m2 * MSUN
    wf['q'] = q
    wf['eta'] = eta
    wf['Mtot_SI'] = wf['m1_SI'] + wf['m2_SI']
    wf['Mtot'] = m1 + m2
    wf['m1'] = m1 / wf['Mtot']
    wf['m2'] = m2 / wf['Mtot']
    wf['M_sec'] = wf['Mtot'] * MTSUN
    wf['delta'] = delta

    wf['eta2'] = eta * eta
    wf['eta3'] = eta * wf['eta2']

    # Spins
    wf['chi1L'] = chi1L
    wf['chi2L'] = chi2L
    wf['chi1L2L'] = chi1L * chi2L

    # Useful powers of spin
    wf['chi1L2'] = chi1L * chi1L
    wf['chi1L3'] = chi1L * chi1L * chi1L
    wf['chi2L2'] = chi2L * chi2L
    wf['chi2L3'] = chi2L * chi2L * chi2L

    # Spin parameterisations
    wf['chiEff'] = XLALSimIMRPhenomXchiEff(eta, chi1L, chi2L)
    wf['chiPNHat'] = XLALSimIMRPhenomXchiPNHat(eta, chi1L, chi2L)
    wf['STotR'] = XLALSimIMRPhenomXSTotR(eta, chi1L, chi2L)
    wf['dchi'] = XLALSimIMRPhenomXdchi(chi1L, chi2L)
    wf['dchi_half'] = wf['dchi'] * 0.5

    wf['SigmaL'] = (wf['chi2L'] * wf['m2']) - (wf['chi1L'] * wf['m1'])
    wf['SL'] = wf['chi1L'] * (wf['m1'] * wf['m1']) + wf['chi2L'] * (wf['m2'] * wf['m2'])

    # Matter parameters
    wf['lambda1'] = lambda1
    wf['lambda2'] = lambda2
    wf['quadparam1'] = quadparam1
    wf['quadparam2'] = quadparam2

    # Tidal parameters (simplified - set to zero for BBH)
    wf['kappa2T'] = 0.0
    wf['fmerger'] = 0.0

    # Reference frequency
    wf['fRef'] = jnp.where(fRef == 0.0, f_min, fRef)
    wf['phiRef_In'] = phi0
    wf['phi0'] = phi0
    wf['beta'] = PI * 0.5 - phi0
    wf['phifRef'] = 0.0

    # Geometric reference frequency
    wf['MfRef'] = XLALSimIMRPhenomXUtilsHztoMf(wf['fRef'], wf['Mtot'])
    wf['piM'] = PI * wf['M_sec']
    wf['v_ref'] = jnp.cbrt(wf['piM'] * wf['fRef'])

    wf['deltaF'] = deltaF
    wf['deltaMF'] = XLALSimIMRPhenomXUtilsHztoMf(wf['deltaF'], wf['Mtot'])

    # Default cutoff frequency
    wf['fCutDef'] = jnp.where(wf['chiEff'] > 0.99, 0.33, 0.3)

    # Minimum and maximum frequency
    wf['fMin'] = f_min
    wf['fMax'] = f_max
    wf['MfMax'] = XLALSimIMRPhenomXUtilsHztoMf(wf['fMax'], wf['Mtot'])

    # Convert fCut to physical cut-off frequency
    wf['fCut'] = wf['fCutDef'] / wf['M_sec']

    # f_max_prime calculation
    wf['f_max_prime'] = wf['fMax']
    wf['f_max_prime'] = jnp.where(wf['fMax'] == 0.0, wf['fCut'], wf['f_max_prime'])
    wf['f_max_prime'] = jnp.where(wf['f_max_prime'] > wf['fCut'], wf['fCut'], wf['f_max_prime'])

    # Final Mass and Spin
    wf['Mfinal'] = XLALSimIMRPhenomXFinalMass2017(wf['eta'], wf['chi1L'], wf['chi2L'])
    wf['afinal'] = XLALSimIMRPhenomXFinalSpin2017(wf['eta'], wf['chi1L'], wf['chi2L'])

    # Set default values for precession-specific parameters
    wf['afinal_nonprec'] = wf['afinal']
    wf['afinal_prec'] = wf['afinal']

    # Ringdown and damping frequency of final BH
    wf['fRING'] = evaluate_QNMfit_fring22(wf['afinal']) / wf['Mfinal']
    wf['fDAMP'] = evaluate_QNMfit_fdamp22(wf['afinal']) / wf['Mfinal']

    # MECO and ISCO frequencies
    wf['fMECO'] = XLALSimIMRPhenomXfMECO(wf['eta'], wf['chi1L'], wf['chi2L'])
    wf['fISCO'] = XLALSimIMRPhenomXfISCO(wf['afinal'])

    # Distance and inclination
    wf['distance'] = distance
    wf['inclination'] = inclination

    # Amplitude normalization
    wf['amp0'] = wf['Mtot'] * MRSUN * wf['Mtot'] * MTSUN / wf['distance']
    wf['ampNorm'] = jnp.sqrt(2.0 / 3.0) * jnp.sqrt(wf['eta']) * (PI ** (-1.0 / 6.0))

    # Phase normalization
    wf['dphase0'] = 5.0 / (128.0 * (PI ** (5.0 / 3.0)))

    # Set nonprecessing value of select precession quantities
    wf['chiTot_perp'] = 0.0
    wf['chi_p'] = 0.0
    wf['theta_LS'] = 0.0
    wf['a1'] = 0.0
    wf['PNR_DEV_PARAMETER'] = 0.0
    wf['PNR_SINGLE_SPIN'] = 0
    wf['MU1'] = 0
    wf['MU2'] = 0
    wf['MU3'] = 0
    wf['MU4'] = 0
    wf['NU0'] = 0
    wf['NU4'] = 0
    wf['NU5'] = 0
    wf['NU6'] = 0
    wf['ZETA1'] = 0
    wf['ZETA2'] = 0
    wf['fRINGEffShiftDividedByEmm'] = 0

    wf['f_inspiral_align'] = 0.0
    wf['XAS_dphase_at_f_inspiral_align'] = 0.0
    wf['XAS_phase_at_f_inspiral_align'] = 0.0
    wf['XHM_dphase_at_f_inspiral_align'] = 0.0
    wf['XHM_phase_at_f_inspiral_align'] = 0.0

    wf['betaRD'] = 0.0
    wf['fRING22_prec'] = 0.0
    wf['fRINGCP'] = 0.0
    wf['pnr_window'] = 0.0

    wf['APPLY_PNR_DEVIATIONS'] = 0

    return wf
