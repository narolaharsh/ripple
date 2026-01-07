"""
IMRPhenomX internal functions converted to JAX.

This module contains the core initialization and setup functions for the
IMRPhenomX waveform model, converted from the LALSimulation C code to JAX.
"""

import jax.numpy as jnp
from typing import Dict, Any, Optional
from ..constants import MSUN, MRSUN, MTSUN_SI, PI
from .LALSimIMRPhenomX_PNR_internals import (
    XLALSimIMRPhenomXchiEff,
    XLALSimIMRPhenomXSTotR,
    XLALSimIMRPhenomXFinalMass2017,
    XLALSimIMRPhenomXFinalSpin2017,
)

from .LALSimIMRPhenomX_internals import (IMRPhenomX_Inspiral_Phase_22_d13, IMRPhenomX_Inspiral_Phase_22_d23, IMRPhenomX_Inspiral_Phase_22_v3, IMRPhenomX_Inspiral_Phase_22_d43, IMRPhenomX_Inspiral_Phase_22_d53)

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
    wf['M_sec'] = wf['Mtot'] * MTSUN_SI
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
    wf['amp0'] = wf['Mtot'] * MRSUN * wf['Mtot'] * MTSUN_SI / wf['distance']
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






def _compute_powers_of_pi():
    """
    Compute useful powers of pi (analogous to powers_of_lalpi in C code).

    Returns:
        Dictionary containing various powers of pi
    """
    return {
        'seven_sixths': jnp.pi ** (7.0 / 6.0),
        'one_sixth': jnp.pi ** (1.0 / 6.0),
        'ten_thirds': jnp.pi ** (10.0 / 3.0),
        'eight_thirds': jnp.pi ** (8.0 / 3.0),
        'seven_thirds': jnp.pi ** (7.0 / 3.0),
        'five_thirds': jnp.pi ** (5.0 / 3.0),
        'four_thirds': jnp.pi ** (4.0 / 3.0),
        'two_thirds': jnp.pi ** (2.0 / 3.0),
        'one_third': jnp.pi ** (1.0 / 3.0),
        'five': jnp.pi ** 5.0,
        'four': jnp.pi ** 4.0,
        'three': jnp.pi ** 3.0,
        'two': jnp.pi ** 2.0,
        'sqrt': jnp.sqrt(jnp.pi),
        'itself': jnp.pi,
        'm_sqrt': jnp.pi ** (-0.5),
        'm_one': 1.0 / jnp.pi,
        'm_two': jnp.pi ** (-2.0),
        'm_three': jnp.pi ** (-3.0),
        'm_four': jnp.pi ** (-4.0),
        'm_five': jnp.pi ** (-5.0),
        'm_six': jnp.pi ** (-6.0),
        'm_one_third': jnp.pi ** (-1.0 / 3.0),
        'm_two_thirds': jnp.pi ** (-2.0 / 3.0),
        'm_four_thirds': jnp.pi ** (-4.0 / 3.0),
        'm_five_thirds': jnp.pi ** (-5.0 / 3.0),
        'm_seven_thirds': jnp.pi ** (-7.0 / 3.0),
        'm_eight_thirds': jnp.pi ** (-8.0 / 3.0),
        'm_ten_thirds': jnp.pi ** (-10.0 / 3.0),
        'm_one_sixth': jnp.pi ** (-1.0 / 6.0),
        'm_seven_sixths': jnp.pi ** (-7.0 / 6.0),
        'log': jnp.log(jnp.pi),
    }


# Pre-compute powers of pi (used throughout phase calculations)
powers_of_lalpi = _compute_powers_of_pi()


def IMRPhenomXGetPhaseCoefficients(pWF: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate phenomenological phase coefficients for IMRPhenomX.

    This function computes the phase coefficients for the inspiral, intermediate,
    and ringdown regions of the IMRPhenomX waveform model. It sets up and solves
    systems of linear equations using collocation methods to determine coefficients
    that match calibrated values at specific frequency points.

    The function implements:
    - Ringdown phase collocation (5 points)
    - Inspiral phase collocation (4 or 5 points depending on version)
    - TaylorF2 Post-Newtonian coefficients up to 4.5PN
    - Pseudo-PN coefficient conversions

    Parameters
    ----------
    pWF : dict
        Waveform parameter dictionary containing masses, spins, and version flags.
        Must contain keys like 'eta', 'chi1L', 'chi2L', 'fRING', 'fDAMP', 'fMECO',
        'fISCO', 'IMRPhenomXRingdownPhaseVersion', 'IMRPhenomXInspiralPhaseVersion', etc.

    Returns
    -------
    pPhase : dict
        Dictionary containing all phase coefficients including:
        - Matching frequencies (fPhaseMatchIN, fPhaseMatchIM)
        - Inspiral/intermediate/ringdown frequency bounds
        - Phenomenological coefficients (a0-a4, b0-b4, c0-c4, etc.)
        - PN phase coefficients (phi0-phi12, dphi0-dphi12)
        - Connection coefficients (initialized to zero, computed later)
        - Collocation points and values

    References
    ----------
    - Pratten et al., PRD 102, 064001 (2020) [arXiv:2001.11412]
    - LALSimulation: LALSimIMRPhenomX_internals.c

    Notes
    -----
    This is a JAX translation of the C function IMRPhenomXGetPhaseCoefficients.
    The linear systems are solved using jnp.linalg.solve instead of GSL's LU decomposition.
    """

    # Initialize phase coefficient dictionary
    pPhase = {}

    # Get useful constants
    LAL_GAMMA = 0.577215664901532860606512090082  # Euler-Mascheroni constant
    LAL_PI = jnp.pi

    # Matching regions
    # Eq. 5.11 in arXiv:2001.11412
    fIMmatch = 0.6 * (0.5 * pWF['fRING'] + pWF['fISCO'])

    # MECO frequency
    fINmatch = pWF['fMECO']

    # Eq. 5.10 in arXiv:2001.11412
    deltaf = (fIMmatch - fINmatch) * 0.03

    # Transition frequencies (Eq. 7.7)
    pPhase['fPhaseMatchIN'] = fINmatch - 1.0 * deltaf  # f_L
    pPhase['fPhaseMatchIM'] = fIMmatch + 0.5 * deltaf  # f_H

    # Inspiral region bounds (Eq. 7.4)
    pPhase['fPhaseInsMin'] = 0.0026  # f_L
    pPhase['fPhaseInsMax'] = 1.020 * pWF['fMECO']  # f_H

    # Ringdown region bounds (Eq. 7.12)
    pPhase['fPhaseRDMin'] = fIMmatch  # f_L
    pPhase['fPhaseRDMax'] = pWF['fRING'] + 1.25 * pWF['fDAMP']  # f_H

    # Phase normalization
    pPhase['phiNorm'] = -(3.0 * powers_of_lalpi['m_five_thirds']) / 128.0

    # Extract useful quantities from waveform struct
    chi1L = pWF['chi1L']
    chi2L = pWF['chi2L']
    chi1L2L = chi1L * chi2L
    chi1L2 = chi1L * chi1L
    chi1L3 = chi1L * chi1L2
    chi2L2 = chi2L * chi2L
    chi2L3 = chi2L * chi2L2
    eta = pWF['eta']
    eta2 = eta * eta
    eta3 = eta * eta2
    delta = pWF['delta']

    # Pre-initialize all phenomenological coefficients
    pPhase['a0'] = 0.0
    pPhase['a1'] = 0.0
    pPhase['a2'] = 0.0
    pPhase['a3'] = 0.0
    pPhase['a4'] = 0.0

    pPhase['b0'] = 0.0
    pPhase['b1'] = 0.0
    pPhase['b2'] = 0.0
    pPhase['b3'] = 0.0
    pPhase['b4'] = 0.0

    pPhase['c0'] = 0.0
    pPhase['c1'] = 0.0
    pPhase['c2'] = 0.0
    pPhase['c3'] = 0.0
    pPhase['c4'] = 0.0
    pPhase['cL'] = 0.0
    pPhase['cRD'] = 0.0

    pPhase['c2PN_tidal'] = 0.0
    pPhase['c3PN_tidal'] = 0.0
    pPhase['c3p5PN_tidal'] = 0.0

    pPhase['sigma0'] = 0.0
    pPhase['sigma1'] = 0.0
    pPhase['sigma2'] = 0.0
    pPhase['sigma3'] = 0.0
    pPhase['sigma4'] = 0.0
    pPhase['sigma5'] = 0.0

    # ========================================================================
    # RINGDOWN PHASE COEFFICIENTS
    # ========================================================================

    # Gauss-Chebyshev collocation points
    gpoints5 = jnp.array([
        0.0,
        1.0/2.0 - 1.0/(2.0*jnp.sqrt(2.0)),
        1.0/2.0,
        1.0/2.0 + 1.0/(2.0*jnp.sqrt(2.0)),
        1.0
    ])

    # Set up ringdown collocation points
    deltax = pPhase['fPhaseRDMax'] - pPhase['fPhaseRDMin']
    xmin = pPhase['fPhaseRDMin']

    CollocationPointsPhaseRD = gpoints5 * deltax + xmin
    # Collocation point 4 is set to the ringdown frequency
    CollocationPointsPhaseRD = CollocationPointsPhaseRD.at[3].set(pWF['fRING'])

    pPhase['CollocationPointsPhaseRD'] = CollocationPointsPhaseRD

    # Check phase version
    if pWF['IMRPhenomXRingdownPhaseVersion'] == 105:
        pPhase['NCollocationPointsRD'] = 5
    else:
        raise ValueError(f"IMRPhenomXRingdownPhaseVersion {pWF['IMRPhenomXRingdownPhaseVersion']} is not valid")

    # NOTE: The calibrated collocation values would need to be imported from
    # the appropriate coefficient functions (IMRPhenomX_Ringdown_Phase_22_*).
    # For now, I'm setting placeholders that should be replaced with actual
    # function calls to the calibrated fits.

    # These functions need to be implemented separately - they contain the
    # calibrated fits from NR simulations (see LALSimIMRPhenomX_ringdown.c)
    # v4 = IMRPhenomX_Ringdown_Phase_22_v4(eta, STotR, dchi, delta, version)
    # For now, use placeholder values - these MUST be replaced with actual calibration

    # Placeholder for calibrated values (NEEDS ACTUAL IMPLEMENTATION)
    RDv4 = 0.0  # Should call IMRPhenomX_Ringdown_Phase_22_v4(...)
    CollocationValuesPhaseRD = jnp.zeros(5)
    # CollocationValuesPhaseRD[0] = IMRPhenomX_Ringdown_Phase_22_d12(...)
    # CollocationValuesPhaseRD[1] = IMRPhenomX_Ringdown_Phase_22_d24(...)
    # CollocationValuesPhaseRD[2] = IMRPhenomX_Ringdown_Phase_22_d34(...)
    # CollocationValuesPhaseRD[3] = RDv4
    # CollocationValuesPhaseRD[4] = IMRPhenomX_Ringdown_Phase_22_d54(...)

    # Accumulate collocation values: v_j = d_{j4} + v4
    # CollocationValuesPhaseRD = CollocationValuesPhaseRD.at[4].set(CollocationValuesPhaseRD[4] + CollocationValuesPhaseRD[3])
    # CollocationValuesPhaseRD = CollocationValuesPhaseRD.at[2].set(CollocationValuesPhaseRD[2] + CollocationValuesPhaseRD[3])
    # CollocationValuesPhaseRD = CollocationValuesPhaseRD.at[1].set(CollocationValuesPhaseRD[1] + CollocationValuesPhaseRD[3])
    # CollocationValuesPhaseRD = CollocationValuesPhaseRD.at[0].set(CollocationValuesPhaseRD[0] + CollocationValuesPhaseRD[1])

    pPhase['CollocationValuesPhaseRD'] = CollocationValuesPhaseRD

    # Set up the linear system for ringdown: A x = b
    # Ansatz (Eq. 7.12): phi_RD(f) = a_0 + a_1 f^{-1/3} + a_2 f^{-2} + a_4 f^{-4} + cL / (f_damp^2 + (f - f_ring)^2)

    A_RD = jnp.zeros((5, 5))
    b_RD = CollocationValuesPhaseRD

    for i in range(5):
        ff = CollocationPointsPhaseRD[i]
        invff = 1.0 / ff
        ff1 = jnp.cbrt(invff)  # f^{-1/3}
        ff2 = invff * invff  # f^{-2}
        ff3 = ff2 * ff2  # f^{-4}
        ff4 = -pWF['dphase0'] / (pWF['fDAMP']**2 + (ff - pWF['fRING'])**2)

        A_RD = A_RD.at[i, 0].set(1.0)  # Constant term
        A_RD = A_RD.at[i, 1].set(ff1)  # f^{-1/3}
        A_RD = A_RD.at[i, 2].set(ff2)  # f^{-2}
        A_RD = A_RD.at[i, 3].set(ff3)  # f^{-4}
        A_RD = A_RD.at[i, 4].set(ff4)  # Lorentzian

    # Solve the linear system using JAX
    x_RD = jnp.linalg.solve(A_RD, b_RD)

    pPhase['c0'] = x_RD[0]  # a0
    pPhase['c1'] = x_RD[1]  # a1
    pPhase['c2'] = x_RD[2]  # a2
    pPhase['c4'] = x_RD[3]  # a4
    pPhase['cRD'] = x_RD[4]  # a_RD
    pPhase['cL'] = -pWF['dphase0'] * pPhase['cRD']

    # Apply NR tuning for precessing cases (if applicable)
    # pPhase['cL'] = pPhase['cL'] + (pWF['PNR_DEV_PARAMETER'] * pWF['NU4'])

    # ========================================================================
    # INSPIRAL PHASE COEFFICIENTS
    # ========================================================================

    deltax = pPhase['fPhaseInsMax'] - pPhase['fPhaseInsMin']
    xmin = pPhase['fPhaseInsMin']

    # Determine number of pseudo-PN coefficients based on version
    version = pWF['IMRPhenomXInspiralPhaseVersion']
    if version == 104:
        NPseudoPN = 4
        NCollocationPointsPhaseIns = 4
    elif version == 105:
        NPseudoPN = 5
        NCollocationPointsPhaseIns = 5
    elif version == 114:
        NPseudoPN = 4
        NCollocationPointsPhaseIns = 4
    elif version == 115:
        NPseudoPN = 5
        NCollocationPointsPhaseIns = 5
    else:
        raise ValueError(f"IMRPhenomXInspiralPhaseVersion {version} is not valid")

    pPhase['NPseudoPN'] = NPseudoPN
    pPhase['NCollocationPointsPhaseIns'] = NCollocationPointsPhaseIns

    # Gauss-Chebyshev points for inspiral
    if NPseudoPN == 4:
        gpoints = jnp.array([0.0, 1.0/4.0, 3.0/4.0, 1.0])
    else:  # NPseudoPN == 5
        gpoints = gpoints5

    # Set up inspiral collocation points
    CollocationPointsPhaseIns = gpoints * deltax + xmin
    pPhase['CollocationPointsPhaseIns'] = CollocationPointsPhaseIns

    # Placeholder for calibrated inspiral values (NEEDS ACTUAL IMPLEMENTATION)
    CollocationValuesPhaseIns = jnp.zeros(NCollocationPointsPhaseIns)
    # These functions need to be imported from LALSimIMRPhenomX_inspiral.c
    CollocationValuesPhaseIns[0] = IMRPhenomX_Inspiral_Phase_22_d13(pWF['eta'],pWF['chiPNHat'],pWF['dchi'],pWF['delta'],pWF['IMRPhenomXInspiralPhaseVersion'])
    CollocationValuesPhaseIns[1] = IMRPhenomX_Inspiral_Phase_22_d23(pWF['eta'],pWF['chiPNHat'],pWF['dchi'],pWF['delta'],pWF['IMRPhenomXInspiralPhaseVersion'])
    CollocationValuesPhaseIns[2] = IMRPhenomX_Inspiral_Phase_22_v3(pWF['eta'],pWF['chiPNHat'],pWF['dchi'],pWF['delta'],pWF['IMRPhenomXInspiralPhaseVersion'])
    CollocationValuesPhaseIns[3] = IMRPhenomX_Inspiral_Phase_22_d43(pWF['eta'],pWF['chiPNHat'],pWF['dchi'],pWF['delta'],pWF['IMRPhenomXInspiralPhaseVersion'])
    if NPseudoPN == 5:
         CollocationValuesPhaseIns[4] = IMRPhenomX_Inspiral_Phase_22_d53(pWF['eta'],pWF['chiPNHat'],pWF['dchi'],pWF['delta'],pWF['IMRPhenomXInspiralPhaseVersion'])

    # Accumulate: v_j = d_j3 + v_3
    CollocationValuesPhaseIns = CollocationValuesPhaseIns.at[0].set(CollocationValuesPhaseIns[0] + CollocationValuesPhaseIns[2])
    CollocationValuesPhaseIns = CollocationValuesPhaseIns.at[1].set(CollocationValuesPhaseIns[1] + CollocationValuesPhaseIns[2])
    CollocationValuesPhaseIns = CollocationValuesPhaseIns.at[3].set(CollocationValuesPhaseIns[3] + CollocationValuesPhaseIns[2])
    if NPseudoPN == 5:
        CollocationValuesPhaseIns = CollocationValuesPhaseIns.at[4].set(CollocationValuesPhaseIns[4] + CollocationValuesPhaseIns[2])

    pPhase['CollocationValuesPhaseIns'] = CollocationValuesPhaseIns

    # Set up the linear system for inspiral: A x = b
    # Ansatz (Eq. 7.4): phi_Ins(f) = a_0 + a_1 f^{1/3} + a_2 f^{2/3} + a_3 f + a_4 f^{4/3}

    A_Ins = jnp.zeros((NCollocationPointsPhaseIns, NCollocationPointsPhaseIns))
    b_Ins = CollocationValuesPhaseIns

    for i in range(NCollocationPointsPhaseIns):
        ff = CollocationPointsPhaseIns[i]
        ff1 = jnp.cbrt(ff)  # f^{1/3}
        ff2 = ff1 * ff1  # f^{2/3}
        ff3 = ff  # f^{1}

        A_Ins = A_Ins.at[i, 0].set(1.0)
        A_Ins = A_Ins.at[i, 1].set(ff1)
        A_Ins = A_Ins.at[i, 2].set(ff2)
        A_Ins = A_Ins.at[i, 3].set(ff3)

        if NPseudoPN == 5:
            ff4 = ff * ff1  # f^{4/3}
            A_Ins = A_Ins.at[i, 4].set(ff4)

    # Solve the linear system
    x_Ins = jnp.linalg.solve(A_Ins, b_Ins)

    pPhase['a0'] = x_Ins[0]
    pPhase['a1'] = x_Ins[1]
    pPhase['a2'] = x_Ins[2]
    pPhase['a3'] = x_Ins[3]
    if NPseudoPN == 5:
        pPhase['a4'] = x_Ins[4]
    else:
        pPhase['a4'] = 0.0

    # Convert pseudo-PN coefficients (normalized by (dphase0 / eta) * f^{8/3})
    # Re-scale by f^{-8/3} in the PN phasing
    pPhase['sigma1'] = (-5.0/3.0) * pPhase['a0']
    pPhase['sigma2'] = (-5.0/4.0) * pPhase['a1']
    pPhase['sigma3'] = (-5.0/5.0) * pPhase['a2']
    pPhase['sigma4'] = (-5.0/6.0) * pPhase['a3']
    pPhase['sigma5'] = (-5.0/7.0) * pPhase['a4']

    # ========================================================================
    # TAYLORF2 POST-NEWTONIAN COEFFICIENTS
    # ========================================================================

    # Initialize all PN coefficients to zero
    for key in ['dphi0', 'dphi1', 'dphi2', 'dphi3', 'dphi4', 'dphi5', 'dphi6',
                'dphi7', 'dphi8', 'dphi9', 'dphi10', 'dphi11', 'dphi12',
                'dphi5L', 'dphi6L', 'dphi8L', 'dphi9L',
                'phi0', 'phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6',
                'phi7', 'phi8', 'phi9', 'phi10', 'phi11', 'phi12',
                'phi5L', 'phi6L', 'phi8L', 'phi9L']:
        pPhase[key] = 0.0

    # Split into non-spinning and spinning coefficients
    # These are the TaylorF2 PN coefficients normalized by 3 / (128 * eta * [pi M f]^{5/3})

    # 0.0 PN (Newtonian)
    phi0NS = 1.0

    # 0.5 PN
    phi1NS = 0.0

    # 1.0 PN - Non-Spinning
    phi2NS = (3715.0/756.0 + (55.0*eta)/9.0) * powers_of_lalpi['two_thirds']

    # 1.5 PN
    phi3NS = -16.0 * powers_of_lalpi['two']
    phi3S = ((113.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) -
              76.0*(chi1L + chi2L)*eta) / 6.0) * powers_of_lalpi['itself']

    # 2.0 PN
    phi4NS = (15293365.0/508032.0 + (27145.0*eta)/504.0 +
              (3085.0*eta2)/72.0) * powers_of_lalpi['four_thirds']
    phi4S = ((-5.0*(81.0*chi1L2*(1.0 + delta - 2.0*eta) + 316.0*chi1L2L*eta -
                    81.0*chi2L2*(-1.0 + delta + 2.0*eta))) / 16.0) * powers_of_lalpi['four_thirds']

    # 2.5 PN
    phi5NS = 0.0
    phi5S = 0.0

    # 2.5 PN - Log terms
    phi5LNS = ((5.0*(46374.0 - 6552.0*eta)*LAL_PI) / 4536.0) * powers_of_lalpi['five_thirds']
    phi5LS = ((-732985.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) -
               560.0*(-1213.0*(chi1L + chi2L) + 63.0*(chi1L - chi2L)*delta)*eta +
               85680.0*(chi1L + chi2L)*eta2) / 4536.0) * powers_of_lalpi['five_thirds']

    # 3.0 PN
    phi6NS = (11583231236531.0/4.69421568e9 -
              (5.0*eta*(3147553127.0 + 588.0*eta*(-45633.0 + 102260.0*eta))) / 3.048192e6 -
              (6848.0*LAL_GAMMA) / 21.0 - (640.0*powers_of_lalpi['two']) / 3.0 +
              (2255.0*eta*powers_of_lalpi['two']) / 12.0 - (13696.0*jnp.log(2.0)) / 21.0 -
              (6848.0*powers_of_lalpi['log']) / 63.0) * powers_of_lalpi['two']
    phi6S = ((5.0*(227.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) -
                   156.0*(chi1L + chi2L)*eta)*LAL_PI) / 3.0) * powers_of_lalpi['two']
    phi6S += ((5.0*(20.0*chi1L2L*eta*(11763.0 + 12488.0*eta) +
                    7.0*chi2L2*(-15103.0*(-1.0 + delta) + 2.0*(-21683.0 + 6580.0*delta)*eta - 9808.0*eta2) -
                    7.0*chi1L2*(-15103.0*(1.0 + delta) + 2.0*(21683.0 + 6580.0*delta)*eta + 9808.0*eta2))) / 4032.0) * powers_of_lalpi['two']

    # 3.0 PN - Log terms
    phi6LNS = (-6848.0/63.0) * powers_of_lalpi['two']
    phi6LS = 0.0

    # 3.5 PN
    phi7NS = ((5.0*(15419335.0 + 168.0*(75703.0 - 29618.0*eta)*eta)*LAL_PI) / 254016.0) * powers_of_lalpi['seven_thirds']
    phi7S = ((5.0*(-5030016755.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) +
                   4.0*(2113331119.0*(chi1L + chi2L) + 675484362.0*(chi1L - chi2L)*delta)*eta -
                   1008.0*(208433.0*(chi1L + chi2L) + 25011.0*(chi1L - chi2L)*delta)*eta2 +
                   90514368.0*(chi1L + chi2L)*eta3)) / 6.096384e6) * powers_of_lalpi['seven_thirds']
    phi7S += (-5.0*(57.0*chi1L2*(1.0 + delta - 2.0*eta) + 220.0*chi1L2L*eta -
                    57.0*chi2L2*(-1.0 + delta + 2.0*eta))*LAL_PI) * powers_of_lalpi['seven_thirds']
    phi7S += ((14585.0*(-(chi2L3*(-1.0 + delta)) + chi1L3*(1.0 + delta)) -
               5.0*(chi2L3*(8819.0 - 2985.0*delta) + 8439.0*chi1L*chi2L2*(-1.0 + delta) -
                    8439.0*chi1L2*chi2L*(1.0 + delta) + chi1L3*(8819.0 + 2985.0*delta))*eta +
               40.0*(chi1L + chi2L)*(17.0*chi1L2 - 14.0*chi1L2L + 17.0*chi2L2)*eta2) / 48.0) * powers_of_lalpi['seven_thirds']

    # 4.0 PN
    phi8NS = 0.0
    phi8S = ((-5.0*(1263141.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) -
                    2.0*(794075.0*(chi1L + chi2L) + 178533.0*(chi1L - chi2L)*delta)*eta +
                    94344.0*(chi1L + chi2L)*eta2)*LAL_PI*(-1.0 + powers_of_lalpi['log'])) / 9072.0) * powers_of_lalpi['eight_thirds']

    # 4.0 PN - Log terms
    phi8LNS = 0.0
    phi8LS = ((-5.0*(1263141.0*(chi1L + chi2L + chi1L*delta - chi2L*delta) -
                     2.0*(794075.0*(chi1L + chi2L) + 178533.0*(chi1L - chi2L)*delta)*eta +
                     94344.0*(chi1L + chi2L)*eta2)*LAL_PI) / 9072.0) * powers_of_lalpi['eight_thirds']

    # 4.5 PN
    phi9NS = 0.0
    phi9S = 0.0
    phi9LNS = 0.0
    phi9LS = 0.0

    # Additional terms for versions 114/115
    if version == 114 or version == 115:
        # 3.5PN Leading Order Spin-Spin Tail Term
        phi7S += ((5.0*(65.0*chi1L2*(1.0 + delta - 2.0*eta) + 252.0*chi1L2L*eta -
                        65.0*chi2L2*(-1.0 + delta + 2.0*eta))*LAL_PI) / 4.0) * powers_of_lalpi['seven_thirds']

        # 4.5PN Tail Term
        phi9NS += ((5.0*(-256.0 + 451.0*eta)*powers_of_lalpi['three']) / 6.0 +
                   (LAL_PI*(105344279473163.0 + 700.0*eta*(-298583452147.0 +
                            96.0*eta*(99645337.0 + 14453257.0*eta)) -
                            12246091038720.0*LAL_GAMMA - 24492182077440.0*jnp.log(2.0))) / 1.877686272e10 -
                   (13696.0*LAL_PI*powers_of_lalpi['log']) / 63.0) * powers_of_lalpi['three']

        phi9LNS += ((-13696.0*LAL_PI) / 63.0) * powers_of_lalpi['three']

    # Assign PN phase coefficients
    pPhase['phi0'] = phi0NS
    pPhase['phi1'] = phi1NS
    pPhase['phi2'] = phi2NS
    pPhase['phi3'] = phi3NS + phi3S
    pPhase['phi4'] = phi4NS + phi4S
    pPhase['phi5'] = phi5NS + phi5S
    pPhase['phi5L'] = phi5LNS + phi5LS
    pPhase['phi6'] = phi6NS + phi6S
    pPhase['phi6L'] = phi6LNS + phi6LS
    pPhase['phi7'] = phi7NS + phi7S
    pPhase['phi8'] = phi8NS + phi8S
    pPhase['phi8L'] = phi8LNS + phi8LS
    pPhase['phi9'] = phi9NS + phi9S
    pPhase['phi9L'] = phi9LNS + phi9LS

    # Higher order terms (10PN, 11PN, 12PN) - not used in standard model
    pPhase['phi10'] = 0.0
    pPhase['phi11'] = 0.0
    pPhase['phi12'] = 0.0

    # Tidal corrections (placeholder - assumes BBH with no tides)
    pPhase['phi_minus2'] = 0.0
    pPhase['phi_minus1'] = 0.0

    # Calculate phase derivatives (for intermediate region matching)
    # These follow from the definition: dphi_n = (N/5) * phi_n where N is the PN order
    pPhase['dphi_minus2'] = (7.0 / 5.0) * pPhase['phi_minus2']
    pPhase['dphi_minus1'] = (6.0 / 5.0) * pPhase['phi_minus1']
    pPhase['dphi0'] = (5.0 / 5.0) * pPhase['phi0']
    pPhase['dphi1'] = (4.0 / 5.0) * pPhase['phi1']
    pPhase['dphi2'] = (3.0 / 5.0) * pPhase['phi2']
    pPhase['dphi3'] = (2.0 / 5.0) * pPhase['phi3']
    pPhase['dphi4'] = (1.0 / 5.0) * pPhase['phi4']
    pPhase['dphi5'] = -(3.0 / 5.0) * pPhase['phi5L']
    pPhase['dphi6'] = -(1.0 / 5.0) * pPhase['phi6'] - (3.0 / 5.0) * pPhase['phi6L']
    pPhase['dphi6L'] = -(1.0 / 5.0) * pPhase['phi6L']
    pPhase['dphi7'] = -(2.0 / 5.0) * pPhase['phi7']
    pPhase['dphi8'] = -(3.0 / 5.0) * pPhase['phi8'] - (3.0 / 5.0) * pPhase['phi8L']
    pPhase['dphi8L'] = -(3.0 / 5.0) * pPhase['phi8L']
    pPhase['dphi9'] = -(4.0 / 5.0) * pPhase['phi9'] - (3.0 / 5.0) * pPhase['phi9L']
    pPhase['dphi9L'] = -(3.0 / 5.0) * pPhase['phi9L']

    # Initialize connection coefficients (computed later by separate function)
    pPhase['C1Int'] = 0.0
    pPhase['C2Int'] = 0.0
    pPhase['C1MRD'] = 0.0
    pPhase['C2MRD'] = 0.0

    return pPhase