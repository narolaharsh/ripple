import jax.numpy as jnp
from .LALSimIMRPhenomTHM_fits import (evaluate_QNMfit_fring21,
                                      evaluate_QNMfit_fring33,
                                      evaluate_QNMfit_fring32,
                                      evaluate_QNMfit_fring44
                                      )

from .LALSimIMRPhenomTHM_fits import (evaluate_QNMfit_fdamp21,
                                      evaluate_QNMfit_fdamp33,
                                      evaluate_QNMfit_fdamp32,
                                      evaluate_QNMfit_fdamp44
                                      )

from .LALSimIMRPhenomTHM_fits import (evaluate_QNMfit_re_l2m2lp2,
                                      evaluate_QNMfit_im_l2m2lp2,
                                      evaluate_QNMfit_re_l2m2lp3,
                                      evaluate_QNMfit_im_l2m2lp3,
                                      evaluate_QNMfit_re_l3m2lp2,
                                      evaluate_QNMfit_im_l3m2lp2,
                                      evaluate_QNMfit_re_l3m2lp3,
                                      evaluate_QNMfit_im_l3m2lp3
                                      )

from .LALSimIMRPhenomXUtilities import IMRPhenomXPsi4ToStrain

import jax
from typing import Dict, Callable

def IMRPhenomXHM_GenerateRingdownFrequency(ell: int, emm: int, wf22: dict) -> float:
    """
    Wrapper function to return ringdown frequency
    
    Args:
        ell: Spherical harmonic l index (int)
        emm: Spherical harmonic m index (int) - guaranteed to be positive
        wf22: Waveform structure dictionary (dict)
        
    Returns:
        float: Ringdown frequency
    """
    # emm is guaranteed to be positive
    modeTag = ell * 10 + emm
    
    # if the tuned coprecessing tuning is activated, use the precessing final spin
    afinal = jnp.where(
        wf22['IMRPhenomXPNRUseTunedCoprec'],
        wf22['afinal_prec'],
        wf22['afinal']
    )
    
    def case_21():
        return evaluate_QNMfit_fring21(afinal) / wf22['Mfinal']
    
    def case_22():
        return wf22['fRING']
    
    def case_33():
        return evaluate_QNMfit_fring33(afinal) / wf22['Mfinal']
    
    def case_32():
        return evaluate_QNMfit_fring32(afinal) / wf22['Mfinal']
    
    def case_44():
        return evaluate_QNMfit_fring44(afinal) / wf22['Mfinal']
    
    def default_case():
        # In JAX, we can't raise errors in jit-compiled code
        # Return NaN or handle error upstream
        return jnp.nan
    
    # Create branches for each case
    fRING = jax.lax.switch(
        jnp.where(modeTag == 21, 0,
        jnp.where(modeTag == 22, 1,
        jnp.where(modeTag == 33, 2,
        jnp.where(modeTag == 32, 3,
        jnp.where(modeTag == 44, 4, 5))))),
        [case_21, case_22, case_33, case_32, case_44, default_case]
    )
    
    # if the coprecessing tuning is activated, return Effective RD frequency
    fRING = jnp.where(
        jnp.logical_and(
            wf22['IMRPhenomXPNRUseTunedCoprec'],
            jnp.logical_or(ell != 2, emm != 2)
        ),
        fRING - emm * wf22['fRINGEffShiftDividedByEmm'],
        fRING
    )
    
    return fRING


def IMRPhenomXHM_Initialize_QNMs() -> Dict[str, Dict[int, Callable]]:
    """
    Initialize QNM (Quasi-Normal Mode) fit functions for different modes.

    This function creates dictionaries mapping mode indices to their corresponding
    fring and fdamp QNM fit functions. The mode indices are:
        0: (2,1) mode
        1: (3,3) mode
        2: (3,2) mode
        3: (4,4) mode

    Returns:
        dict: Dictionary containing 'fring_lm' and 'fdamp_lm' subdictionaries,
              each mapping mode indices to their respective fit functions.

    
    """
    # Note: In Python/JAX, we use dictionaries instead of C-style function pointer arrays
    # This is more Pythonic and works well with JAX's functional programming style

    qnms = {
        'fring_lm': {
            0: evaluate_QNMfit_fring21,
            1: evaluate_QNMfit_fring33,
            2: evaluate_QNMfit_fring32,
            3: evaluate_QNMfit_fring44
        },
        'fdamp_lm': {
            0: evaluate_QNMfit_fdamp21,
            1: evaluate_QNMfit_fdamp33,
            2: evaluate_QNMfit_fdamp32,
            3: evaluate_QNMfit_fdamp44
        }
    }

    return qnms


def IMRPhenomXHM_Initialize_MixingCoeffs(wf: dict, wf22: dict) -> dict:
    """
    Initialize mixing coefficients for (3,2) mode.

    The mixing coefficients are used to transform the spheroidal-harmonic
    ringdown ansatz back to spherical-harmonic representation.
    Only the (3,2) mode has mixing with other modes.

    Args:
        wf: Waveform dictionary for the higher mode
        wf22: Waveform dictionary for the (2,2) mode

    Returns:
        dict: Updated waveform dictionary with mixing coefficients
    """
    # Initialize mixing coefficients array (4 complex numbers)
    # In C: wf->mixingCoeffs[0] through wf->mixingCoeffs[3]
    # In Python/JAX: store as array of complex numbers

    afinal = wf22['afinal']

    # Compute complex mixing coefficients
    # mixingCoeffs[0] corresponds to l=2, m=2, lp=2
    a222 = evaluate_QNMfit_re_l2m2lp2(afinal) + 1j * evaluate_QNMfit_im_l2m2lp2(afinal)

    # mixingCoeffs[1] corresponds to l=2, m=2, lp=3
    a223 = evaluate_QNMfit_re_l2m2lp3(afinal) + 1j * evaluate_QNMfit_im_l2m2lp3(afinal)

    # mixingCoeffs[2] corresponds to l=3, m=2, lp=2
    a322 = evaluate_QNMfit_re_l3m2lp2(afinal) + 1j * evaluate_QNMfit_im_l3m2lp2(afinal)

    # mixingCoeffs[3] corresponds to l=3, m=2, lp=3
    a323 = evaluate_QNMfit_re_l3m2lp3(afinal) + 1j * evaluate_QNMfit_im_l3m2lp3(afinal)

    # Adjust conventions so that they match the ones used for the hybrids
    # In the C code: wf->mixingCoeffs[2] = -1.* wf->mixingCoeffs[2]
    #                wf->mixingCoeffs[3] = -1.* wf->mixingCoeffs[3]
    a322 = -1.0 * a322
    a323 = -1.0 * a323

    # Store in waveform dictionary as a list
    wf['mixingCoeffs'] = jnp.array([a222, a223, a322, a323])

    return wf


def IMRPhenomXHM_SetHMWaveformVariables(
    ell: int,
    emm: int,
    wf22: dict,
    qnms: dict,
    LALParams: dict = None
) -> dict:
    """
    Set higher mode waveform variables for IMRPhenomXHM.

    This function initializes all the parameters needed for generating
    a specific higher mode (ell, emm) waveform based on the 22 mode
    properties and QNM fits.

    Args:
        ell: Spherical harmonic l index
        emm: Spherical harmonic m index (absolute value)
        wf22: Dictionary containing 22 mode waveform parameters
        qnms: Dictionary containing QNM fit functions
        LALParams: Optional dictionary for LAL parameters

    Returns:
        dict: Dictionary containing all waveform variables for the (ell, emm) mode
    """
    # Initialize LALParams if not provided
    if LALParams is None:
        LALParams = {}

    # Initialize waveform dictionary
    wf = {}

    # Read in which mode is being generated
    wf['ell'] = ell
    wf['emm'] = emm
    wf['modeTag'] = ell * 10 + emm  # 21, 33, 32, 44
    wf['ampNorm'] = wf22['ampNorm']
    wf['fMECOlm'] = wf22['fMECO'] * emm * 0.5
    wf['Ampzero'] = 0  # Ampzero = 1 (true) for odd modes and equal black holes
    wf['Amp0'] = wf22['amp0']
    wf['useFAmpPN'] = 0  # Only true for the 21, this mode has a different inspiral ansatz
    wf['AmpEMR'] = 0  # Only one intermediate region
    wf['InspiralAmpVeto'] = 0
    wf['IntermediateAmpVeto'] = 0
    wf['RingdownAmpVeto'] = 0

    # Mode-specific settings
    modeTag = wf['modeTag']

    # Determine modeInt and MixingOn based on modeTag
    def set_mode_21():
        return 0, 0, jnp.where(
            jnp.logical_and(wf22['q'] == 1.0, wf22['chi1L'] == wf22['chi2L']),
            1, 0
        )

    def set_mode_33():
        return 1, 0, jnp.where(
            jnp.logical_and(wf22['q'] == 1.0, wf22['chi1L'] == wf22['chi2L']),
            1, 0
        )

    def set_mode_32():
        return 2, 1, 0

    def set_mode_44():
        return 3, 0, 0

    def default_mode():
        return -1, 0, 0

    modeInt, MixingOn, Ampzero_mode = jax.lax.switch(
        jnp.where(modeTag == 21, 0,
        jnp.where(modeTag == 33, 1,
        jnp.where(modeTag == 32, 2,
        jnp.where(modeTag == 44, 3, 4)))),
        [set_mode_21, set_mode_33, set_mode_32, set_mode_44, default_mode]
    )

    wf['modeInt'] = modeInt
    wf['MixingOn'] = MixingOn
    wf['Ampzero'] = Ampzero_mode

    # Version numbers - default to 122019 if not specified in LALParams
    wf['IMRPhenomXHMReleaseVersion'] = LALParams.get('IMRPhenomXHMReleaseVersion', 122019)

    wf['IMRPhenomXHMInspiralPhaseVersion'] = LALParams.get('IMRPhenomXHMInspiralPhaseVersion', 122019)
    wf['IMRPhenomXHMIntermediatePhaseVersion'] = LALParams.get('IMRPhenomXHMIntermediatePhaseVersion', 122019)
    wf['IMRPhenomXHMRingdownPhaseVersion'] = LALParams.get('IMRPhenomXHMRingdownPhaseVersion', 122019)

    wf['IMRPhenomXHMInspiralAmpFitsVersion'] = LALParams.get('IMRPhenomXHMInspiralAmpFitsVersion', 122018)
    wf['IMRPhenomXHMIntermediateAmpFitsVersion'] = LALParams.get('IMRPhenomXHMIntermediateAmpFitsVersion', 122018)
    wf['IMRPhenomXHMRingdownAmpFitsVersion'] = LALParams.get('IMRPhenomXHMRingdownAmpFitsVersion', 122018)

    wf['IMRPhenomXHMInspiralAmpFreqsVersion'] = LALParams.get('IMRPhenomXHMInspiralAmpFreqsVersion', 122018)
    wf['IMRPhenomXHMIntermediateAmpFreqsVersion'] = LALParams.get('IMRPhenomXHMIntermediateAmpFreqsVersion', 122018)
    wf['IMRPhenomXHMRingdownAmpFreqsVersion'] = LALParams.get('IMRPhenomXHMRingdownAmpFreqsVersion', 122018)

    wf['IMRPhenomXHMInspiralAmpVersion'] = LALParams.get('IMRPhenomXHMInspiralAmpVersion', 3)
    wf['IMRPhenomXHMIntermediateAmpVersion'] = LALParams.get('IMRPhenomXHMIntermediateAmpVersion', 2)
    wf['IMRPhenomXHMRingdownAmpVersion'] = LALParams.get('IMRPhenomXHMRingdownAmpVersion', 0)

    # Initialize HM tuning parameters (all zeros by default)
    wf['MU1'] = 0.0
    wf['MU2'] = 0.0
    wf['MU3'] = 0.0
    wf['MU4'] = 0.0
    wf['NU0'] = 0.0
    wf['NU4'] = 0.0
    wf['NU5'] = 0.0
    wf['NU6'] = 0.0
    wf['ZETA1'] = 0.0
    wf['ZETA2'] = 0.0

    # Special handling for (3,3) mode tuning
    if modeTag == 33:
        # Set PNR deviation parameter
        wf['PNR_DEV_PARAMETER'] = wf22['delta'] * wf22.get('PNR_DEV_PARAMETER', 0.0)

        # Check if tuned coprecessing model should be used
        IMRPhenomXPNRUseTunedCoprec33 = wf22.get('IMRPhenomXPNRUseTunedCoprec33', False)

        if IMRPhenomXPNRUseTunedCoprec33:
            # These would need to be imported from PhenomXCP fits
            # For now, we'll use LALParams values
            # In full implementation, these should call:
            # XLALSimIMRPhenomXCP_MU1_l3m3, XLALSimIMRPhenomXCP_MU2_l3m3, etc.
            wf['MU1'] = LALParams.get('PhenomXCPMU1l3m3', 0.0)
            wf['MU2'] = LALParams.get('PhenomXCPMU2l3m3', 0.0)
            wf['MU3'] = LALParams.get('PhenomXCPMU3l3m3', 0.0)
            wf['MU4'] = LALParams.get('PhenomXCPMU4l3m3', 0.0)
            wf['NU4'] = LALParams.get('PhenomXCPNU4l3m3', 0.0)
            wf['NU5'] = LALParams.get('PhenomXCPNU5l3m3', 0.0)
            wf['NU6'] = LALParams.get('PhenomXCPNU6l3m3', 0.0)
            wf['ZETA1'] = LALParams.get('PhenomXCPZETA1l3m3', 0.0)
            wf['ZETA2'] = LALParams.get('PhenomXCPZETA2l3m3', 0.0)
        else:
            # Use values from LALParams
            wf['MU1'] = LALParams.get('PhenomXCPMU1l3m3', 0.0)
            wf['MU2'] = LALParams.get('PhenomXCPMU2l3m3', 0.0)
            wf['MU3'] = LALParams.get('PhenomXCPMU3l3m3', 0.0)
            wf['MU4'] = LALParams.get('PhenomXCPMU4l3m3', 0.0)
            wf['NU0'] = LALParams.get('PhenomXCPNU0l3m3', 0.0)
            wf['NU4'] = LALParams.get('PhenomXCPNU4l3m3', 0.0)
            wf['NU5'] = LALParams.get('PhenomXCPNU5l3m3', 0.0)
            wf['NU6'] = LALParams.get('PhenomXCPNU6l3m3', 0.0)
            wf['ZETA1'] = LALParams.get('PhenomXCPZETA1l3m3', 0.0)
            wf['ZETA2'] = LALParams.get('PhenomXCPZETA2l3m3', 0.0)
    else:
        wf['PNR_DEV_PARAMETER'] = 0.0

    # Set phase version variables
    wf['IMRPhenomXHMInspiralPhaseFitsVersion'] = wf['IMRPhenomXHMInspiralPhaseVersion']
    wf['IMRPhenomXHMIntermediatePhaseFitsVersion'] = wf['IMRPhenomXHMIntermediatePhaseVersion']
    wf['IMRPhenomXHMRingdownPhaseFitsVersion'] = wf['IMRPhenomXHMRingdownPhaseVersion']

    wf['IMRPhenomXHMInspiralPhaseFreqsVersion'] = wf['IMRPhenomXHMInspiralPhaseVersion']
    wf['IMRPhenomXHMIntermediatePhaseFreqsVersion'] = wf['IMRPhenomXHMIntermediatePhaseVersion']
    wf['IMRPhenomXHMRingdownPhaseFreqsVersion'] = wf['IMRPhenomXHMRingdownPhaseVersion']

    wf['nCollocPtsRDPhase'] = 0

    # Release-specific settings
    if wf['IMRPhenomXHMReleaseVersion'] == 122019:
        # For q>70 and chi1<0.9 use two intermediate regions
        if wf22['eta'] < 0.013886133703630232 and wf22['chi1L'] <= 0.9:
            wf['AmpEMR'] = 1

        # Mode-specific vetos and settings for 122019
        if modeTag == 21:
            if wf22['q'] < 8.0:
                wf['InspiralAmpVeto'] = 1
                wf['IntermediateAmpVeto'] = 1
            wf['RingdownAmpVeto'] = 1
            if wf22['eta'] >= 0.0237954:  # for EMR (q>40)
                wf['useFAmpPN'] = 1
        elif modeTag == 33:
            if wf22['q'] == 1.0 and wf22['chi1L'] == wf22['chi2L']:
                wf['Ampzero'] = 1
        elif modeTag == 32:
            wf['RingdownAmpVeto'] = 1
            wf['nCollocPtsRDPhase'] = 4

        wf['nCollocPtsInspAmp'] = wf['IMRPhenomXHMInspiralAmpVersion']
        wf['nCollocPtsInterAmp'] = wf['IMRPhenomXHMIntermediateAmpVersion']
        wf['IMRPhenomXHMRingdownPhaseFreqsVersion'] = 122019

    elif wf['IMRPhenomXHMReleaseVersion'] == 122022:
        # Amplitude versions for 122022 release
        wf['IMRPhenomXHMInspiralAmpFreqsVersion'] = 122022
        wf['IMRPhenomXHMIntermediateAmpFreqsVersion'] = 0
        wf['IMRPhenomXHMRingdownAmpFreqsVersion'] = 122022

        wf['IMRPhenomXHMInspiralAmpFitsVersion'] = 122022
        wf['IMRPhenomXHMIntermediateAmpFitsVersion'] = 122022
        wf['IMRPhenomXHMRingdownAmpFitsVersion'] = 122022

        wf['IMRPhenomXHMInspiralAmpVersion'] = 123
        wf['IMRPhenomXHMIntermediateAmpVersion'] = 211112
        wf['IMRPhenomXHMRingdownAmpVersion'] = 2

        # Mode-specific tweaks
        if modeTag == 21:
            wf['IMRPhenomXHMIntermediateAmpVersion'] = 110102
        elif modeTag == 32:
            wf['IMRPhenomXHMRingdownAmpVersion'] = 1
            wf['nCollocPtsRDPhase'] = 4

        # Calculate number of collocation points from version number
        wf['nCollocPtsInspAmp'] = len(str(wf['IMRPhenomXHMInspiralAmpVersion']))
        wf['nCollocPtsInterAmp'] = len(str(wf['IMRPhenomXHMIntermediateAmpVersion']))
    else:
        raise ValueError(f"IMRPhenomXHMReleaseVersion={wf['IMRPhenomXHMReleaseVersion']} is not valid.")

    # Common for all releases
    if modeTag == 32:
        wf['nCollocPtsInterPhase'] = 6
    else:
        wf['nCollocPtsInterPhase'] = 5

    # Limit between comparable and extreme mass ratios for the phase
    wf['etaEMR'] = 0.05

    # Spin parameterizations
    wf['chi_s'] = (wf22['chi1L'] + wf22['chi2L']) * 0.5
    wf['chi_a'] = (wf22['chi1L'] - wf22['chi2L']) * 0.5

    # Ringdown and damping frequencies
    afinal = wf22['afinal']
    wf['fRING'] = qnms['fring_lm'][modeInt](afinal) / wf22['Mfinal']
    wf['fDAMP'] = qnms['fdamp_lm'][modeInt](afinal) / wf22['Mfinal']

    # Apply precessing corrections if using tuned coprecessing model
    IMRPhenomXPNRUseTunedCoprec = wf22.get('IMRPhenomXPNRUseTunedCoprec', False)

    if IMRPhenomXPNRUseTunedCoprec:
        afinal_prec = wf22.get('afinal_prec', afinal)
        wf['fRING'] = qnms['fring_lm'][modeInt](afinal_prec) / wf22['Mfinal']
        wf['fDAMP'] = qnms['fdamp_lm'][modeInt](afinal_prec) / wf22['Mfinal']

        # Apply effective ringdown frequency shift
        fRINGEffShiftDividedByEmm = wf22.get('fRINGEffShiftDividedByEmm', 0.0)
        wf['fRING'] = wf['fRING'] - emm * fRINGEffShiftDividedByEmm

    # Apply (3,3) mode specific corrections
    IMRPhenomXPNRUseTunedCoprec33 = wf22.get('IMRPhenomXPNRUseTunedCoprec33', False)
    if IMRPhenomXPNRUseTunedCoprec33 and modeTag == 33:
        # Apply PNR CoPrec deviations
        wf['fRING'] = wf['fRING'] - (wf['PNR_DEV_PARAMETER'] * wf['NU5'])
        wf['fDAMP'] = wf['fDAMP'] + (wf['PNR_DEV_PARAMETER'] * wf['NU6'])

        if IMRPhenomXPNRUseTunedCoprec:
            # Transition to EZH effective ringdown frequency
            pnr_window = wf22.get('pnr_window', 1.0)
            fRINGEffShiftDividedByEmm = wf22.get('fRINGEffShiftDividedByEmm', 0.0)
            wf['fRING'] = wf['fRING'] - (1.0 - pnr_window) * emm * fRINGEffShiftDividedByEmm

    # If (l,m)=(3,2), initialize mixing coefficients
    if modeTag == 32:
        wf['MixingOn'] = 1
        wf = IMRPhenomXHM_Initialize_MixingCoeffs(wf, wf22)

    # Phase shift and time shift
    
    wf['timeshift'] = 0.0
    wf['phaseshift'] = 0.0

    # Current time-alignment of the hybrids
    psi4tostrain =  IMRPhenomXPsi4ToStrain(wf22['eta'], wf22['STotR'], wf22['dchi'])
    wf['DeltaT'] = -2.0 * jnp.pi * (500.0 + psi4tostrain)

    wf['fPhaseRDflat'] = 0.0
    wf['fAmpRDfalloff'] = 0.0

    return wf
