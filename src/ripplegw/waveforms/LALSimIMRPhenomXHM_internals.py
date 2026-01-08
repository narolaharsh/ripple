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


from .LALSimIMRPhenomXHM_ringdown import (
        IMRPhenomXHM_RD_Phase_Ansatz,
        IMRPhenomXHM_RD_Phase_AnsatzInt,
        IMRPhenomXHM_RD_Phase_DerAnsatz,
        IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift,
        IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift,
        IMRPhenomXHM_RD_Phase_22_alpha2,
        IMRPhenomXHM_RD_Phase_22_alphaL
    )

from .LALSimIMRPhenomXHM_ringdown import (
        IMRPhenomXHM_RD_Amp_Ansatz,
        IMRPhenomXHM_RD_Amp_DAnsatz,
        IMRPhenomXHM_RD_Amp_NDAnsatz,
        IMRPhenomXHM_RD_Amp_Coefficients,
        IMRPhenomXHM_Amplitude_fcutRD,
        SpheroidalToSpherical
    )





from .LALSimIMRPhenomX_internals import (IMRPhenomXGetPhaseCoefficients,  
                                         IMRPhenomX_dPhase_22, 
                                         IMRPhenomX_Phase_22,
                                         IMRPhenomX_TimeShift_22,
                                         IMRPhenomX_Phase_22_ConnectionCoefficients)


from .LALSimIMRPhenomXHM_inspiral import (IMRPhenomXHM_Insp_Phase_LambdaPN,
                                          IMRPhenomXHM_Inspiral_Phase_AnsatzInt,
                                          IMRPhenomXHM_Inspiral_Phase_Ansatz,
                                          )

from .LALSimIMRPhenomXHM_inspiral import (
        IMRPhenomXHM_Inspiral_PNAmp_Ansatz,
        IMRPhenomXHM_Inspiral_Amp_Ansatz,
        IMRPhenomXHM_Inspiral_Amp_NDAnsatz,
        IMRPhenomXHM_Inspiral_Amp_rho1,
        IMRPhenomXHM_Inspiral_Amp_rho2,
        IMRPhenomXHM_Inspiral_Amp_rho3,
        IMRPhenomXHM_GetPNAmplitudeCoefficients,
        IMRPhenomXHM_Inspiral_Amplitude_Veto,
        IMRPhenomXHM_Get_Inspiral_Amp_Coefficients,
        IMRPhenomXHM_Amplitude_fcutInsp
    )


from .LALSimIMRPhenomXHM_intermediate import (IMRPhenomXHM_Inter_Phase_AnsatzInt,
                                              IMRPhenomXHM_Inter_Phase_Ansatz)



from .LALSimIMRPhenomXHM_intermediate import (
        IMRPhenomXHM_Intermediate_Amp_Coefficients,
        IMRPhenomXHM_Intermediate_Amp_delta0,
        IMRPhenomXHM_Intermediate_Amp_delta1,
        IMRPhenomXHM_Intermediate_Amp_delta2,
        IMRPhenomXHM_Intermediate_Amp_delta3,
        IMRPhenomXHM_Intermediate_Amp_delta4,
        IMRPhenomXHM_Intermediate_Amp_delta5,
        WavyPoints
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


def IMRPhenomXHM_FillAmpFitsArray() -> dict:
    """
    Fill arrays with function references for amplitude coefficient/collocation point fits.

    This function creates a mapping of fit functions for amplitude coefficients
    and collocation points across three regions (inspiral, intermediate, ringdown)
    and four modes (21, 33, 32, 44).

    Returns:
        dict: Dictionary containing three arrays:
            - 'InspiralAmpFits': List of 12 functions for inspiral amplitude fits
            - 'IntermediateAmpFits': List of 24 functions for intermediate amplitude fits
            - 'RingdownAmpFits': List of 26 functions for ringdown amplitude fits


    """
    pAmp = {}

    # Initialize arrays with None placeholders
    # In the full implementation, these would be function references
    pAmp['InspiralAmpFits'] = [None] * 12
    pAmp['IntermediateAmpFits'] = [None] * 24
    pAmp['RingdownAmpFits'] = [None] * 26

    # ******Inspiral Fits for collocation points******
    # Each mode (21, 33, 32, 44) has 3 collocation points at different frequencies

    # Mode 21
    pAmp['InspiralAmpFits'][0] = IMRPhenomXHM_Insp_Amp_21_iv1  # fcutInsp
    pAmp['InspiralAmpFits'][1] = IMRPhenomXHM_Insp_Amp_21_iv2  # fcutInsp*0.75
    pAmp['InspiralAmpFits'][2] = IMRPhenomXHM_Insp_Amp_21_iv3  # fcutInsp*0.5

    # Mode 33
    pAmp['InspiralAmpFits'][3] = IMRPhenomXHM_Insp_Amp_33_iv1  # fcutInsp
    pAmp['InspiralAmpFits'][4] = IMRPhenomXHM_Insp_Amp_33_iv2  # fcutInsp*0.75
    pAmp['InspiralAmpFits'][5] = IMRPhenomXHM_Insp_Amp_33_iv3  # fcutInsp*0.5

    # Mode 32
    pAmp['InspiralAmpFits'][6] = IMRPhenomXHM_Insp_Amp_32_iv1  # fcutInsp
    pAmp['InspiralAmpFits'][7] = IMRPhenomXHM_Insp_Amp_32_iv2  # fcutInsp*0.75
    pAmp['InspiralAmpFits'][8] = IMRPhenomXHM_Insp_Amp_32_iv3  # fcutInsp*0.5

    # Mode 44
    pAmp['InspiralAmpFits'][9] = IMRPhenomXHM_Insp_Amp_44_iv1   # fcutInsp
    pAmp['InspiralAmpFits'][10] = IMRPhenomXHM_Insp_Amp_44_iv2  # fcutInsp*0.75
    pAmp['InspiralAmpFits'][11] = IMRPhenomXHM_Insp_Amp_44_iv3  # fcutInsp*0.5

    # *****Intermediate Fits for EMR collocation points, 2 Intermediate regions*****

    # Mode 21
    pAmp['IntermediateAmpFits'][0] = IMRPhenomXHM_Inter_Amp_21_int1  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp['IntermediateAmpFits'][1] = IMRPhenomXHM_Inter_Amp_21_int2  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 33
    pAmp['IntermediateAmpFits'][2] = IMRPhenomXHM_Inter_Amp_33_int1  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp['IntermediateAmpFits'][3] = IMRPhenomXHM_Inter_Amp_33_int2  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 32
    pAmp['IntermediateAmpFits'][4] = IMRPhenomXHM_Inter_Amp_32_int1  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp['IntermediateAmpFits'][5] = IMRPhenomXHM_Inter_Amp_32_int2  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Mode 44
    pAmp['IntermediateAmpFits'][6] = IMRPhenomXHM_Inter_Amp_44_int1  # fcutInsp + (fcutRD-fcutInsp)/3
    pAmp['IntermediateAmpFits'][7] = IMRPhenomXHM_Inter_Amp_44_int2  # fcutInsp + 2(fcutRD-fcutInsp)/3

    # Additional intermediate fits for EMR
    # Mode 21
    pAmp['IntermediateAmpFits'][8] = IMRPhenomXHM_Inter_Amp_21_int0   # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp['IntermediateAmpFits'][9] = IMRPhenomXHM_Inter_Amp_21_dint0  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 33
    pAmp['IntermediateAmpFits'][10] = IMRPhenomXHM_Inter_Amp_33_int0   # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp['IntermediateAmpFits'][11] = IMRPhenomXHM_Inter_Amp_33_dint0  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 32
    pAmp['IntermediateAmpFits'][12] = IMRPhenomXHM_Inter_Amp_32_int0   # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp['IntermediateAmpFits'][13] = IMRPhenomXHM_Inter_Amp_32_dint0  # fcutInsp + (fInt1 - fcutInsp)/3

    # Mode 44
    pAmp['IntermediateAmpFits'][14] = IMRPhenomXHM_Inter_Amp_44_int0   # fcutInsp + (fInt1 - fcutInsp)/3
    pAmp['IntermediateAmpFits'][15] = IMRPhenomXHM_Inter_Amp_44_dint0  # fcutInsp + (fInt1 - fcutInsp)/3

    # Additional intermediate fits
    # Mode 21
    pAmp['IntermediateAmpFits'][16] = IMRPhenomXHM_Inter_Amp_21_int3
    pAmp['IntermediateAmpFits'][17] = IMRPhenomXHM_Inter_Amp_21_int4

    # Mode 33
    pAmp['IntermediateAmpFits'][18] = IMRPhenomXHM_Inter_Amp_33_int3
    pAmp['IntermediateAmpFits'][19] = IMRPhenomXHM_Inter_Amp_33_int4

    # Mode 32
    pAmp['IntermediateAmpFits'][20] = IMRPhenomXHM_Inter_Amp_32_int3
    pAmp['IntermediateAmpFits'][21] = IMRPhenomXHM_Inter_Amp_32_int4

    # Mode 44
    pAmp['IntermediateAmpFits'][22] = IMRPhenomXHM_Inter_Amp_44_int3
    pAmp['IntermediateAmpFits'][23] = IMRPhenomXHM_Inter_Amp_44_int4

    # ****Ringdown Fits for coefficients****

    # Mode 21
    pAmp['RingdownAmpFits'][0] = IMRPhenomXHM_RD_Amp_21_alambda
    pAmp['RingdownAmpFits'][1] = IMRPhenomXHM_RD_Amp_21_lambda
    pAmp['RingdownAmpFits'][2] = IMRPhenomXHM_RD_Amp_21_sigma

    # Mode 33
    pAmp['RingdownAmpFits'][3] = IMRPhenomXHM_RD_Amp_33_alambda
    pAmp['RingdownAmpFits'][4] = IMRPhenomXHM_RD_Amp_33_lambda
    pAmp['RingdownAmpFits'][5] = IMRPhenomXHM_RD_Amp_33_sigma  # currently constant

    # Mode 32
    pAmp['RingdownAmpFits'][6] = IMRPhenomXHM_RD_Amp_32_alambda
    pAmp['RingdownAmpFits'][7] = IMRPhenomXHM_RD_Amp_32_lambda
    pAmp['RingdownAmpFits'][8] = IMRPhenomXHM_RD_Amp_32_sigma  # currently constant

    # Mode 44
    pAmp['RingdownAmpFits'][9] = IMRPhenomXHM_RD_Amp_44_alambda
    pAmp['RingdownAmpFits'][10] = IMRPhenomXHM_RD_Amp_44_lambda
    pAmp['RingdownAmpFits'][11] = IMRPhenomXHM_RD_Amp_44_sigma  # currently constant

    # ****Ringdown Fits for Collocation Points****

    # Mode 21
    pAmp['RingdownAmpFits'][12] = IMRPhenomXHM_RD_Amp_21_rdcp1
    pAmp['RingdownAmpFits'][13] = IMRPhenomXHM_RD_Amp_21_rdcp2
    pAmp['RingdownAmpFits'][14] = IMRPhenomXHM_RD_Amp_21_rdcp3

    # Mode 33
    pAmp['RingdownAmpFits'][15] = IMRPhenomXHM_RD_Amp_33_rdcp1
    pAmp['RingdownAmpFits'][16] = IMRPhenomXHM_RD_Amp_33_rdcp2
    pAmp['RingdownAmpFits'][17] = IMRPhenomXHM_RD_Amp_33_rdcp3

    # Mode 32
    pAmp['RingdownAmpFits'][18] = IMRPhenomXHM_RD_Amp_32_rdcp1
    pAmp['RingdownAmpFits'][19] = IMRPhenomXHM_RD_Amp_32_rdcp2
    pAmp['RingdownAmpFits'][20] = IMRPhenomXHM_RD_Amp_32_rdcp3

    # Mode 44
    pAmp['RingdownAmpFits'][21] = IMRPhenomXHM_RD_Amp_44_rdcp1
    pAmp['RingdownAmpFits'][22] = IMRPhenomXHM_RD_Amp_44_rdcp2
    pAmp['RingdownAmpFits'][23] = IMRPhenomXHM_RD_Amp_44_rdcp3

    # Mode 32 auxiliary fits
    pAmp['RingdownAmpFits'][24] = IMRPhenomXHM_RD_Amp_32_rdaux1
    pAmp['RingdownAmpFits'][25] = IMRPhenomXHM_RD_Amp_32_rdaux2

    return pAmp


def IMRPhenomXHM_FillPhaseFitsArray() -> dict:
    """
    Fill arrays with function references for phase coefficient/collocation point fits.

    This function creates a mapping of fit functions for phase coefficients
    and collocation points across three regions (inspiral, intermediate, ringdown)
    and four modes (21, 33, 32, 44).

    Returns:
        dict: Dictionary containing three arrays:
            - 'InspiralPhaseFits': List of 4 functions for inspiral phase lambda fits
            - 'IntermediatePhaseFits': List of 24 functions for intermediate phase fits
            - 'RingdownPhaseFits': List of 5 functions for ringdown phase fits (32 mode spheroidal)


    """
    pPhase = {}

    # Initialize arrays with None placeholders
    # In the full implementation, these would be function references
    pPhase['InspiralPhaseFits'] = [None] * 4
    pPhase['IntermediatePhaseFits'] = [None] * 24
    pPhase['RingdownPhaseFits'] = [None] * 5

    # ******Inspiral Phase Fits for lambda coefficients******

    # Mode 21
    pPhase['InspiralPhaseFits'][0] = IMRPhenomXHM_Insp_Phase_21_lambda

    # Mode 33
    pPhase['InspiralPhaseFits'][1] = IMRPhenomXHM_Insp_Phase_33_lambda

    # Mode 32
    pPhase['InspiralPhaseFits'][2] = IMRPhenomXHM_Insp_Phase_32_lambda

    # Mode 44
    pPhase['InspiralPhaseFits'][3] = IMRPhenomXHM_Insp_Phase_44_lambda

    # ******Intermediate Phase Fits for collocation points******

    # Mode 21 - 6 collocation points (p1 through p6)
    pPhase['IntermediatePhaseFits'][0] = IMRPhenomXHM_Inter_Phase_21_p1
    pPhase['IntermediatePhaseFits'][1] = IMRPhenomXHM_Inter_Phase_21_p2
    pPhase['IntermediatePhaseFits'][2] = IMRPhenomXHM_Inter_Phase_21_p3
    pPhase['IntermediatePhaseFits'][3] = IMRPhenomXHM_Inter_Phase_21_p4
    pPhase['IntermediatePhaseFits'][4] = IMRPhenomXHM_Inter_Phase_21_p5
    pPhase['IntermediatePhaseFits'][5] = IMRPhenomXHM_Inter_Phase_21_p6

    # Mode 33 - 6 collocation points (p1 through p6)
    pPhase['IntermediatePhaseFits'][6] = IMRPhenomXHM_Inter_Phase_33_p1
    pPhase['IntermediatePhaseFits'][7] = IMRPhenomXHM_Inter_Phase_33_p2
    pPhase['IntermediatePhaseFits'][8] = IMRPhenomXHM_Inter_Phase_33_p3
    pPhase['IntermediatePhaseFits'][9] = IMRPhenomXHM_Inter_Phase_33_p4
    pPhase['IntermediatePhaseFits'][10] = IMRPhenomXHM_Inter_Phase_33_p5
    pPhase['IntermediatePhaseFits'][11] = IMRPhenomXHM_Inter_Phase_33_p6

    # Mode 32 - 6 collocation points (p1 through p6)
    pPhase['IntermediatePhaseFits'][12] = IMRPhenomXHM_Inter_Phase_32_p1
    pPhase['IntermediatePhaseFits'][13] = IMRPhenomXHM_Inter_Phase_32_p2
    pPhase['IntermediatePhaseFits'][14] = IMRPhenomXHM_Inter_Phase_32_p3
    pPhase['IntermediatePhaseFits'][15] = IMRPhenomXHM_Inter_Phase_32_p4
    pPhase['IntermediatePhaseFits'][16] = IMRPhenomXHM_Inter_Phase_32_p5
    pPhase['IntermediatePhaseFits'][17] = IMRPhenomXHM_Inter_Phase_32_p6

    # Mode 44 - 6 collocation points (p1 through p6)
    pPhase['IntermediatePhaseFits'][18] = IMRPhenomXHM_Inter_Phase_44_p1
    pPhase['IntermediatePhaseFits'][19] = IMRPhenomXHM_Inter_Phase_44_p2
    pPhase['IntermediatePhaseFits'][20] = IMRPhenomXHM_Inter_Phase_44_p3
    pPhase['IntermediatePhaseFits'][21] = IMRPhenomXHM_Inter_Phase_44_p4
    pPhase['IntermediatePhaseFits'][22] = IMRPhenomXHM_Inter_Phase_44_p5
    pPhase['IntermediatePhaseFits'][23] = IMRPhenomXHM_Inter_Phase_44_p6

    # ******Ringdown Phase Fits (32 Spheroidal)******

    # Mode 32 - 5 collocation points for spheroidal ringdown phase
    pPhase['RingdownPhaseFits'][0] = IMRPhenomXHM_RD_Phase_32_p1
    pPhase['RingdownPhaseFits'][1] = IMRPhenomXHM_RD_Phase_32_p2
    pPhase['RingdownPhaseFits'][2] = IMRPhenomXHM_RD_Phase_32_p3
    pPhase['RingdownPhaseFits'][3] = IMRPhenomXHM_RD_Phase_32_p4
    pPhase['RingdownPhaseFits'][4] = IMRPhenomXHM_RD_Phase_32_p5

    return pPhase


def compute_powers_of_f(f: float) -> dict:
    """
    Compute useful powers of frequency for phase/amplitude calculations.

    Args:
        f: Frequency value

    Returns:
        Dictionary containing various powers of f
    """
    return {
        'itself': f,
        'one_sixth': f ** (1.0 / 6.0),
        'one_third': f ** (1.0 / 3.0),
        'two_thirds': f ** (2.0 / 3.0),
        'four_thirds': f ** (4.0 / 3.0),
        'five_thirds': f ** (5.0 / 3.0),
        'seven_sixths': f ** (7.0 / 6.0),
        'seven_thirds': f ** (7.0 / 3.0),
        'eight_thirds': f ** (8.0 / 3.0),
        'sqrt': jnp.sqrt(f),
        'two': f ** 2.0,
        'three': f ** 3.0,
        'four': f ** 4.0,
        'five': f ** 5.0,
        'm_one_sixth': f ** (-1.0 / 6.0),
        'm_one_third': f ** (-1.0 / 3.0),
        'm_two_thirds': f ** (-2.0 / 3.0),
        'm_four_thirds': f ** (-4.0 / 3.0),
        'm_five_thirds': f ** (-5.0 / 3.0),
        'm_seven_sixths': f ** (-7.0 / 6.0),
        'm_sqrt': f ** (-0.5),
        'm_one': 1.0 / f,
        'm_two': f ** (-2.0),
        'm_three': f ** (-3.0),
        'm_four': f ** (-4.0),
        'm_five': f ** (-5.0),
        'log': jnp.log(f),
    }


def GetSpheroidalCoefficients(pPhase: dict, pPhase22: dict, pWFHM: dict, pWF22: dict) -> dict:
    """
    Compute spheroidal ringdown phase coefficients for higher modes.

    This function reconstructs the spheroidal ringdown phase derivative using collocation
    points and solves for the ansatz coefficients. It then adjusts the time and phase
    shifts relative to the (2,2) mode.

    Args:
        pPhase: Phase coefficient dictionary for the higher mode
        pPhase22: Phase coefficient dictionary for the (2,2) mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pPhase dictionary with spheroidal coefficients
    """
    
    

    nCollocationPts_RD_Phase = pWFHM['nCollocPtsRDPhase']

    # Initialize collocation arrays
    CollocValuesPhaseRingdown = jnp.zeros(nCollocationPts_RD_Phase)
    CollocFreqsPhaseRingdown = jnp.zeros(nCollocationPts_RD_Phase)

    # Initialize GSL-equivalent arrays for JAX linear algebra
    b = jnp.zeros(nCollocationPts_RD_Phase)
    A = jnp.zeros((nCollocationPts_RD_Phase, nCollocationPts_RD_Phase))

    # Get collocation point frequencies
    pPhase = IMRPhenomXHM_Ringdown_CollocPtsFreqs(pPhase, pWFHM, pWF22)

    # Fill in the collocation points for the phase
    if pWFHM['IMRPhenomXHMRingdownPhaseVersion'] == 122019:
        for i in range(nCollocationPts_RD_Phase):
            CollocValuesPhaseRingdown = CollocValuesPhaseRingdown.at[i].set(
                pPhase['RingdownPhaseFits'][i](pWF22, pWFHM['IMRPhenomXHMRingdownPhaseFitsVersion'])
            )
            CollocFreqsPhaseRingdown = CollocFreqsPhaseRingdown.at[i].set(
                pPhase['CollocationPointsFreqsPhaseRD'][i]
            )
            b = b.at[i].set(CollocValuesPhaseRingdown[i])

            ff = CollocFreqsPhaseRingdown[i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            lorentzian = pWFHM['fDAMP'] / (pWFHM['fDAMP']**2 + (ff - pWFHM['fRING'])**2)
            fpowers = jnp.array([1.0, lorentzian, ffm2, ffm2 * ffm2])

            for j in range(nCollocationPts_RD_Phase):
                A = A.at[i, j].set(fpowers[j])
    else:
        for i in range(nCollocationPts_RD_Phase):
            CollocValuesPhaseRingdown = CollocValuesPhaseRingdown.at[i].set(
                pPhase['RingdownPhaseFits'][i](pWF22, pWFHM['IMRPhenomXHMRingdownPhaseFitsVersion'])
            )
            CollocFreqsPhaseRingdown = CollocFreqsPhaseRingdown.at[i].set(
                pPhase['CollocationPointsFreqsPhaseRD'][i]
            )
            b = b.at[i].set(CollocValuesPhaseRingdown[i])

            ff = CollocFreqsPhaseRingdown[i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            lorentzian = pWFHM['fDAMP'] / (pWFHM['fDAMP'] * pWFHM['fDAMP'] +
                                           (ff - pWFHM['fRING']) * (ff - pWFHM['fRING']))
            fpowers = jnp.array([1.0, ffm1, ffm2, ffm2 * ffm2, lorentzian])

            for j in range(nCollocationPts_RD_Phase):
                A = A.at[i, j].set(fpowers[j])

    # Solve the linear system using JAX (instead of GSL LU decomposition)
    x = jnp.linalg.solve(A, b)

    # Extract coefficients based on version
    if pWFHM['IMRPhenomXHMRingdownPhaseVersion'] == 122019:
        # ansatz: alpha0 + (alpha2)/(f^2) + (alpha4)/(f^4) + alphaL*(fdamplm)/((fdamplm)^2 + (f - fRDlm)^2)
        pPhase['alpha0_S'] = x[0]
        pPhase['alphaL_S'] = x[1]
        pPhase['alpha2_S'] = x[2]
        pPhase['alpha4_S'] = x[3]
    else:
        # Store all coefficients
        if 'RDCoefficient' not in pPhase:
            pPhase['RDCoefficient'] = jnp.zeros(nCollocationPts_RD_Phase + 3)
        for i in range(nCollocationPts_RD_Phase):
            pPhase['RDCoefficient'] = pPhase['RDCoefficient'].at[i].set(x[i])

    # Adjust the relative time and phase shift of the final phiS wrt IMRPhenomX
    frefRD = pWF22['fRING'] + pWF22['fDAMP']
    powers_of_FREF = compute_powers_of_f(frefRD)

    # Get time shift from fit
    tshift = IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift(pWF22, pWFHM['IMRPhenomXHMRingdownPhaseFitsVersion'])

    # Compute connection coefficients for the 22 mode if not already computed
    if 'c0' not in pPhase22:
        pPhase22 = IMRPhenomXGetPhaseCoefficients(pWF22)

    # Compute time shift for 22 mode
    pWFHM['timeshift'] = IMRPhenomX_TimeShift_22(pPhase22, pWF22)

    pPhase['phi0_S'] = 0.0

    # Impose that dphiS(fref) - dphi22(fref) has the value given by our fit
    if pWFHM['IMRPhenomXHMRingdownPhaseVersion'] == 122019:
        dphi22ref = (1.0 / pWF22['eta']) * IMRPhenomX_dPhase_22(frefRD, powers_of_FREF, pPhase22, pWF22) + pWFHM['timeshift']
        pPhase['alpha0_S'] = (pPhase['alpha0_S'] + dphi22ref + tshift -
                             IMRPhenomXHM_RD_Phase_Ansatz(frefRD, powers_of_FREF, pWFHM, pPhase))
    else:
        dphi22ref = (1.0 / pWF22['eta']) * IMRPhenomX_dPhase_22(frefRD, powers_of_FREF, pPhase22, pWF22) + pWFHM['timeshift']
        dphi32ref = IMRPhenomXHM_RD_Phase_Ansatz(frefRD, powers_of_FREF, pWFHM, pPhase)
        pPhase['RDCoefficient'] = pPhase['RDCoefficient'].at[0].add(dphi22ref + tshift - dphi32ref)

        # Compute additional coefficients for flattening at high frequencies
        exponent = 5  # a + b / f^5
        ff = pWFHM['fRING'] + 2 * pWFHM['fDAMP']
        powers_of_ff = compute_powers_of_f(ff)

        pPhase['RDCoefficient'] = pPhase['RDCoefficient'].at[nCollocationPts_RD_Phase + 1].set(
            -IMRPhenomXHM_RD_Phase_DerAnsatz(ff, powers_of_ff, pWFHM, pPhase) * ff**(exponent + 1) / exponent
        )
        pPhase['RDCoefficient'] = pPhase['RDCoefficient'].at[nCollocationPts_RD_Phase].set(
            IMRPhenomXHM_RD_Phase_Ansatz(ff, powers_of_ff, pWFHM, pPhase) -
            pPhase['RDCoefficient'][nCollocationPts_RD_Phase + 1] / ff**exponent
        )
        pPhase['RDCoefficient'] = pPhase['RDCoefficient'].at[nCollocationPts_RD_Phase + 2].set(
            IMRPhenomXHM_RD_Phase_AnsatzInt(ff, powers_of_ff, pWFHM, pPhase) -
            (pPhase['RDCoefficient'][5] * ff - 0.25 * pPhase['RDCoefficient'][6] * powers_of_ff['m_four'])
        )
        pWFHM['fPhaseRDflat'] = ff

    # Phase-shift of spheroidal ansatz
    frefRD = pWF22['fRING']
    powers_of_FREF = compute_powers_of_f(frefRD)

    # Compute phi22(fref)
    powers_of_MfRef = compute_powers_of_f(pWF22['MfRef'])
    pWFHM['phiref22'] = (-1.0 / pWF22['eta'] * IMRPhenomX_Phase_22(pWF22['MfRef'], powers_of_MfRef, pPhase22, pWF22) -
                         pWFHM['timeshift'] * pWF22['MfRef'] - pWFHM['phaseshift'] +
                         2.0 * pWF22['phi0'] + jnp.pi / 4.0)

    phi22ref = (1.0 / pWF22['eta'] * IMRPhenomX_Phase_22(frefRD, powers_of_FREF, pPhase22, pWF22) +
                pWFHM['timeshift'] * frefRD + pWFHM['phaseshift'] + pWFHM['phiref22'])

    # Get phase shift from fit
    phishift = IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift(pWF22, pWFHM['IMRPhenomXHMRingdownPhaseFitsVersion'])

    # Adjust the relative phase of our reconstruction
    pPhase['phi0_S'] = (phi22ref - IMRPhenomXHM_RD_Phase_AnsatzInt(frefRD, powers_of_FREF, pWFHM, pPhase) +
                        phishift)

    return pPhase


def IMRPhenomXHM_GetAmplitudeCoefficients(
    pAmp: dict,
    pPhase: dict,
    pAmp22: dict,
    pPhase22: dict,
    pWFHM: dict,
    pWF22: dict
) -> dict:
    """
    Calculate amplitude coefficients for IMRPhenomXHM higher modes.

    This function computes amplitude coefficients across three regions (inspiral,
    intermediate, and ringdown) for a specific higher mode. It handles mode-specific
    normalizations, Post-Newtonian coefficients, and applies various vetoes for
    regions outside calibration.

    Args:
        pAmp: Amplitude coefficient dictionary for the higher mode
        pPhase: Phase coefficient dictionary for the higher mode
        pAmp22: Amplitude coefficient dictionary for the (2,2) mode
        pPhase22: Phase coefficient dictionary for the (2,2) mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pAmp dictionary with all amplitude coefficients
    """

    # Initialize amplitude normalization
    pAmp['ampNorm'] = pWF22['ampNorm']
    pAmp['PNdominant'] = pWF22['ampNorm'] * jnp.power(2.0 / pWFHM['emm'], -7.0/6.0)

    # Mode-specific PN dominant term
    modeTag = pWFHM['modeTag']

    if modeTag == 21:
        if pWF22['q'] == 1:
            pAmp['PNdominantlmpower'] = 2
            pAmp['PNdominantlm'] = (jnp.sqrt(2.0) / 3.0 * 1.5 * pWF22['dchi'] * 0.5 *
                                    jnp.power(2 * jnp.pi / pWFHM['emm'], 2.0/3.0))
        else:
            pAmp['PNdominantlmpower'] = 1
            pAmp['PNdominantlm'] = (jnp.sqrt(2.0) / 3.0 * pWF22['delta'] *
                                    jnp.power(2 * jnp.pi / pWFHM['emm'], 1.0/3.0))
    elif modeTag == 33:
        if pWF22['q'] == 1:
            pAmp['PNdominantlmpower'] = 4
            pAmp['PNdominantlm'] = (0.75 * jnp.sqrt(5.0/7.0) * pWF22['dchi'] * 0.5 *
                                    (65.0/24.0 - 28.0/3.0 * pWF22['eta']) *
                                    jnp.power(2 * jnp.pi / pWFHM['emm'], 4.0/3.0))
        else:
            pAmp['PNdominantlmpower'] = 1
            pAmp['PNdominantlm'] = (0.75 * jnp.sqrt(5.0/7.0) * pWF22['delta'] *
                                    jnp.power(2 * jnp.pi / pWFHM['emm'], 1.0/3.0))
    elif modeTag == 32:
        pAmp['PNdominantlmpower'] = 2
        pAmp['PNdominantlm'] = 0.75 * jnp.sqrt(5.0/7.0) * jnp.power(2 * jnp.pi / pWFHM['emm'], 1.0/3.0)
    elif modeTag == 44:
        pAmp['PNdominantlmpower'] = 2
        pAmp['PNdominantlm'] = (4.0/9.0 * jnp.sqrt(10.0/7.0) * (1 - 3 * pWF22['eta']) *
                                jnp.power(2 * jnp.pi / pWFHM['emm'], 1.0/3.0))
    else:
        raise ValueError(f"Mode {modeTag} not supported. Available modes: 21, 33, 32, 44")

    pAmp['PNdominantlm'] = jnp.abs(pAmp['PNdominantlm'])
    pWFHM['fAmpRDfalloff'] = 0.0
    pAmp['nCoefficientsInter'] = 0

    # Branch based on release version
    if pWFHM['IMRPhenomXHMReleaseVersion'] != 122019:
        # Newer version (122022)
        pAmp['InspRescaleFactor'] = 0
        pAmp['RDRescaleFactor'] = 0
        pAmp['InterRescaleFactor'] = 0

        # Transform IMRPhenomXHMIntermediateAmpVersion to int array
        num = pWFHM['IMRPhenomXHMIntermediateAmpVersion']
        pAmp['VersionCollocPtsInter'] = jnp.zeros(pWFHM['nCollocPtsInterAmp'], dtype=jnp.int32)
        for i in range(pWFHM['nCollocPtsInterAmp']):
            pAmp['VersionCollocPtsInter'] = pAmp['VersionCollocPtsInter'].at[pWFHM['nCollocPtsInterAmp'] - i - 1].set(num % 10)
            num = num // 10

        pAmp['nCoefficientsInter'] = jnp.sum(pAmp['VersionCollocPtsInter'])

        if pAmp['nCoefficientsInter'] > pWFHM['nCollocPtsInterAmp'] + 2:
            raise ValueError(
                f"Inconsistent number of collocation points ({pWFHM['nCollocPtsInterAmp'] + 2}) "
                f"and free parameters ({pAmp['nCoefficientsInter']})"
            )

        pAmp['nCoefficientsRDAux'] = 0
        if pWFHM['MixingOn']:
            pAmp['nCollocPtsRDAux'] = 2
            pAmp['nCoefficientsRDAux'] = 4
            pAmp['fRDAux'] = pWFHM['fRING'] - pWFHM['fDAMP']

        # Get cutting frequencies
        pAmp['fAmpMatchIN'] = IMRPhenomXHM_Amplitude_fcutInsp(pWFHM, pWF22)
        pAmp['fAmpMatchIM'] = IMRPhenomXHM_Amplitude_fcutRD(pWFHM, pWF22)
        pWFHM['fAmpRDfalloff'] = pWFHM['fRING'] + 2 * pWFHM['fDAMP']

        # Get PN amplitude coefficients
        pAmp = IMRPhenomXHM_GetPNAmplitudeCoefficients(pAmp, pWFHM, pWF22)

        # Get coefficients for each region
        pAmp = IMRPhenomXHM_Get_Inspiral_Amp_Coefficients(pAmp, pWFHM, pWF22)
        pAmp = IMRPhenomXHM_RD_Amp_Coefficients(pWF22, pWFHM, pAmp)
        pAmp = IMRPhenomXHM_Intermediate_Amp_Coefficients(pAmp, pWFHM, pWF22, pPhase, pAmp22, pPhase22)

        # Set rescale factors
        pAmp['InspRescaleFactor'] = 0
        pAmp['RDRescaleFactor'] = 0
        pAmp['InterRescaleFactor'] = 0

    else:
        # Original version (122019)
        pAmp['InspRescaleFactor'] = 1
        pAmp['InterRescaleFactor'] = -1
        pAmp['RDRescaleFactor'] = 1

        # Options for extrapolation outside calibration region
        if ((modeTag == 44 or modeTag == 33) and pWF22['q'] > 7.0 and pWF22['chi1L'] > 0.95):
            pAmp['useInspAnsatzRingdown'] = 1
        else:
            pAmp['useInspAnsatzRingdown'] = 0

        pAmp['WavyInsp'] = 0
        pAmp['WavyInt'] = 0
        if modeTag == 21:
            pAmp['WavyInsp'] = 1
            pAmp['WavyInt'] = 1
        if modeTag == 32:
            pAmp['WavyInsp'] = 1

        # Get cutting frequencies
        pAmp['fAmpMatchIN'] = IMRPhenomXHM_Amplitude_fcutInsp(pWFHM, pWF22)
        pAmp['fAmpMatchIM'] = IMRPhenomXHM_Amplitude_fcutRD(pWFHM, pWF22)

        # Compute intermediate collocation point frequencies
        df = pAmp['fAmpMatchIM'] - pAmp['fAmpMatchIN']
        if 'CollocationPointsFreqsAmplitudeInter' not in pAmp:
            pAmp['CollocationPointsFreqsAmplitudeInter'] = jnp.zeros(2)
        pAmp['CollocationPointsFreqsAmplitudeInter'] = pAmp['CollocationPointsFreqsAmplitudeInter'].at[0].set(
            pAmp['fAmpMatchIN'] + df/3.0
        )
        pAmp['CollocationPointsFreqsAmplitudeInter'] = pAmp['CollocationPointsFreqsAmplitudeInter'].at[1].set(
            pAmp['fAmpMatchIN'] + df*2.0/3.0
        )

        nCollocPtsInspAmp = pWFHM['nCollocPtsInspAmp']
        nCollocPtsInterAmp = pWFHM['nCollocPtsInterAmp']
        modeint = pWFHM['modeInt']

        # === INSPIRAL REGION ===

        # Get PN amplitude coefficients
        pAmp = IMRPhenomXHM_GetPNAmplitudeCoefficients(pAmp, pWFHM, pWF22)

        # Initialize collocation point frequencies
        if 'CollocationPointsFreqsAmplitudeInsp' not in pAmp:
            pAmp['CollocationPointsFreqsAmplitudeInsp'] = jnp.zeros(3)
        pAmp['CollocationPointsFreqsAmplitudeInsp'] = pAmp['CollocationPointsFreqsAmplitudeInsp'].at[0].set(
            1.0 * pAmp['fAmpMatchIN']
        )
        pAmp['CollocationPointsFreqsAmplitudeInsp'] = pAmp['CollocationPointsFreqsAmplitudeInsp'].at[1].set(
            0.75 * pAmp['fAmpMatchIN']
        )
        pAmp['CollocationPointsFreqsAmplitudeInsp'] = pAmp['CollocationPointsFreqsAmplitudeInsp'].at[2].set(
            0.5 * pAmp['fAmpMatchIN']
        )

        fcutInsp = pAmp['fAmpMatchIN']
        f1 = pAmp['CollocationPointsFreqsAmplitudeInsp'][0]
        f2 = pAmp['CollocationPointsFreqsAmplitudeInsp'][1]
        f3 = pAmp['CollocationPointsFreqsAmplitudeInsp'][2]

        # Compute useful powers
        powers_of_fcutInsp = compute_powers_of_f(fcutInsp)
        powers_of_f1 = compute_powers_of_f(f1)
        powers_of_f2 = compute_powers_of_f(f2)
        powers_of_f3 = compute_powers_of_f(f3)

        # Compute PN ansatz values
        PNf1 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f1, pWFHM, pAmp)
        PNf2 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f2, pWFHM, pAmp)
        PNf3 = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_f3, pWFHM, pAmp)

        if 'PNAmplitudeInsp' not in pAmp:
            pAmp['PNAmplitudeInsp'] = jnp.zeros(3)
        pAmp['PNAmplitudeInsp'] = pAmp['PNAmplitudeInsp'].at[0].set(PNf1)
        pAmp['PNAmplitudeInsp'] = pAmp['PNAmplitudeInsp'].at[1].set(PNf2)
        pAmp['PNAmplitudeInsp'] = pAmp['PNAmplitudeInsp'].at[2].set(PNf3)

        # Get collocation point values from fits
        if 'CollocationPointsValuesAmplitudeInsp' not in pAmp:
            pAmp['CollocationPointsValuesAmplitudeInsp'] = jnp.zeros(nCollocPtsInspAmp)
        for i in range(nCollocPtsInspAmp):
            pAmp['CollocationPointsValuesAmplitudeInsp'] = pAmp['CollocationPointsValuesAmplitudeInsp'].at[i].set(
                jnp.abs(pAmp['InspiralAmpFits'][modeint*nCollocPtsInspAmp + i](pWF22, pWFHM['IMRPhenomXHMInspiralAmpFitsVersion']))
            )

        # Pseudo-PN collocation points
        iv1 = pAmp['CollocationPointsValuesAmplitudeInsp'][0] - PNf1
        iv2 = pAmp['CollocationPointsValuesAmplitudeInsp'][1] - PNf2
        iv3 = pAmp['CollocationPointsValuesAmplitudeInsp'][2] - PNf3

        # Apply inspiral veto
        if pWFHM['InspiralAmpVeto'] == 1:
            iv1, iv2, iv3, powers_of_f1, powers_of_f2, powers_of_f3 = IMRPhenomXHM_Inspiral_Amplitude_Veto(
                iv1, iv2, iv3, powers_of_f1, powers_of_f2, powers_of_f3, pAmp, pWFHM
            )

        # Mode-specific vetoes
        if modeTag == 32 and pWF22['q'] > 2.5 and pWF22['chi1L'] < -0.9 and pWF22['chi2L'] < -0.9:
            pWFHM['IMRPhenomXHMInspiralAmpVersion'] = 0

        if modeTag == 32 and pWF22['q'] > 2.5 and pWF22['chi1L'] < -0.6 and pWF22['chi2L'] > 0.0 and iv1 != 0:
            pWFHM['IMRPhenomXHMInspiralAmpVersion'] = pWFHM['IMRPhenomXHMInspiralAmpVersion'] - 1
            iv1 = 0.0

        if modeTag == 33 and (1.2 > pWF22['q'] > 1.0 and pWF22['chi1L'] < -0.1 and pWF22['chi2L'] > 0.0 and iv1 != 0):
            pWFHM['IMRPhenomXHMInspiralAmpVersion'] = pWFHM['IMRPhenomXHMInspiralAmpVersion'] - 1
            iv1 = 0.0

        # Check for wavy collocation points
        if pWFHM['IMRPhenomXHMInspiralAmpVersion'] == 3 and pAmp['WavyInsp'] == 1:
            if WavyPoints(
                pAmp['CollocationPointsValuesAmplitudeInsp'][0] * powers_of_f1['m_seven_sixths'],
                pAmp['CollocationPointsValuesAmplitudeInsp'][1] * powers_of_f2['m_seven_sixths'],
                pAmp['CollocationPointsValuesAmplitudeInsp'][2] * powers_of_f3['m_seven_sixths']
            ) == 1:
                iv2 = 0
                pWFHM['IMRPhenomXHMInspiralAmpVersion'] = pWFHM['IMRPhenomXHMInspiralAmpVersion'] - 1

        # Rename collocation points
        if iv2 == 0:
            iv2 = iv3
            iv3 = 0.0
            powers_of_f2 = powers_of_f3

        if iv1 == 0:
            iv1 = iv2
            powers_of_f1 = powers_of_f2
            powers_of_fcutInsp = powers_of_f1
            iv2 = iv3
            iv3 = 0.0
            powers_of_f2 = powers_of_f3

        if pWFHM['IMRPhenomXHMInspiralAmpVersion'] == 0:
            powers_of_fcutInsp = compute_powers_of_f(pWFHM['fMECOlm'])

        # Update inspiral cutting frequency
        pAmp['fAmpMatchIN'] = powers_of_fcutInsp['itself']

        # Get pseudo-PN coefficients
        pAmp['rho1'] = IMRPhenomXHM_Inspiral_Amp_rho1(iv1, iv2, iv3, powers_of_fcutInsp, powers_of_f1, powers_of_f2, powers_of_f3, pWFHM)
        pAmp['rho2'] = IMRPhenomXHM_Inspiral_Amp_rho2(iv1, iv2, iv3, powers_of_fcutInsp, powers_of_f1, powers_of_f2, powers_of_f3, pWFHM)
        pAmp['rho3'] = IMRPhenomXHM_Inspiral_Amp_rho3(iv1, iv2, iv3, powers_of_fcutInsp, powers_of_f1, powers_of_f2, powers_of_f3, pWFHM)

        # Store useful powers for later use
        pAmp['fcutInsp_seven_thirds'] = powers_of_fcutInsp['seven_thirds']
        pAmp['fcutInsp_eight_thirds'] = powers_of_fcutInsp['eight_thirds']
        pAmp['fcutInsp_three'] = powers_of_fcutInsp['three']

        # === RINGDOWN REGION ===

        # Get ringdown coefficients from fits
        if 'RDCoefficient' not in pAmp:
            pAmp['RDCoefficient'] = jnp.zeros(4)

        pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[0].set(
            jnp.abs(pAmp['RingdownAmpFits'][modeint*3](pWF22, pWFHM['IMRPhenomXHMRingdownAmpFitsVersion']))
        )
        pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[1].set(
            pAmp['RingdownAmpFits'][modeint*3 + 1](pWF22, pWFHM['IMRPhenomXHMRingdownAmpFitsVersion'])
        )
        pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[2].set(
            pAmp['RingdownAmpFits'][modeint*3 + 2](pWF22, pWFHM['IMRPhenomXHMRingdownAmpFitsVersion'])
        )
        pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[3].set(1.0/12.0)

        # Apply PNR tuning for (3,3) mode
        if modeTag == 33:
            pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[0].add(pWFHM['PNR_DEV_PARAMETER'] * pWFHM['MU3'])
            pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[2].add(pWFHM['PNR_DEV_PARAMETER'] * pWFHM['MU4'])

        # Handle extreme spin cases
        if pAmp['useInspAnsatzRingdown'] == 1:
            powers_of_fRD = compute_powers_of_f(pAmp['fAmpMatchIM'])
            insp_amp = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_fcutInsp, pWFHM, pAmp)
            rd_amp = IMRPhenomXHM_RD_Amp_Ansatz(powers_of_fRD, pWFHM, pAmp) / pAmp['RDCoefficient'][0]
            pAmp['RDCoefficient'] = pAmp['RDCoefficient'].at[0].set(
                0.9 * jnp.abs(insp_amp / rd_amp)
            )

        # === INTERMEDIATE REGION ===

        # Get collocation point values from fits
        if 'CollocationPointsValuesAmplitudeInter' not in pAmp:
            pAmp['CollocationPointsValuesAmplitudeInter'] = jnp.zeros(nCollocPtsInterAmp)
        for i in range(nCollocPtsInterAmp):
            pAmp['CollocationPointsValuesAmplitudeInter'] = pAmp['CollocationPointsValuesAmplitudeInter'].at[i].set(
                jnp.abs(pAmp['IntermediateAmpFits'][modeint*nCollocPtsInterAmp + i](pWF22, pWFHM['IMRPhenomXHMIntermediateAmpFitsVersion']))
            )

        # Set up frequencies
        F1 = powers_of_fcutInsp['itself']
        F2 = pAmp['CollocationPointsFreqsAmplitudeInter'][0]
        F3 = pAmp['CollocationPointsFreqsAmplitudeInter'][1]
        F4 = pAmp['fAmpMatchIM']

        # Compute powers
        powers_of_F1 = compute_powers_of_f(F1)
        powers_of_F4 = compute_powers_of_f(F4)

        # Compute boundary values
        inspF1 = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_F1, pWFHM, pAmp)

        if pWFHM['MixingOn'] == 1:
            rdF4 = jnp.abs(SpheroidalToSpherical(powers_of_F4, pAmp22, pPhase22, pAmp, pPhase, pWFHM, pWF22))
        else:
            rdF4 = IMRPhenomXHM_RD_Amp_Ansatz(powers_of_F4, pWFHM, pAmp)

        # Compute boundary derivatives
        d1 = IMRPhenomXHM_Inspiral_Amp_NDAnsatz(powers_of_F1, pWFHM, pAmp)

        if pWFHM['MixingOn'] == 1:
            d4 = IMRPhenomXHM_RD_Amp_NDAnsatz(powers_of_F4, pAmp, pPhase, pWFHM, pAmp22, pPhase22, pWF22)
        else:
            d4 = IMRPhenomXHM_RD_Amp_DAnsatz(powers_of_F4, pWFHM, pAmp)

        # Set rescale factors
        pAmp['InspRescaleFactor'] = 0
        pAmp['InterRescaleFactor'] = 0
        pAmp['RDRescaleFactor'] = 0

        # Transform derivatives
        d1 = ((7.0/6.0) * jnp.power(F1, 1.0/6.0) / inspF1) - (jnp.power(F1, 7.0/6.0) * d1 / (inspF1 * inspF1))
        d4 = ((7.0/6.0) * jnp.power(F4, 1.0/6.0) / rdF4) - (jnp.power(F4, 7.0/6.0) * d4 / (rdF4 * rdF4))

        # Set up collocation point values
        pWFHM['IMRPhenomXHMIntermediateAmpVersion'] = 105  # Default: 5th order polynomial

        V1 = powers_of_F1['m_seven_sixths'] * inspF1
        V2 = pAmp['CollocationPointsValuesAmplitudeInter'][0]
        V3 = pAmp['CollocationPointsValuesAmplitudeInter'][1]
        V4 = powers_of_F4['m_seven_sixths'] * rdF4

        # Apply veto for extreme cases
        if pAmp['useInspAnsatzRingdown'] == 1:
            V2 = 1.0
            V3 = 1.0
            pWFHM['IMRPhenomXHMIntermediateAmpVersion'] = 101  # Linear reconstruction

        # Invert for polynomial reconstruction
        V1 = 1.0 / V1
        V2 = 1.0 / V2
        V3 = 1.0 / V3
        V4 = 1.0 / V4

        # Apply NR tuning for (3,3) mode
        if modeTag == 33:
            V2 = V2 + (pWFHM['PNR_DEV_PARAMETER'] * pWFHM['MU1'])
            V3 = V3 + (pWFHM['PNR_DEV_PARAMETER'] * pWFHM['MU2'])

        # Handle EMR cases (two intermediate regions)
        if pWFHM['AmpEMR'] == 1:
            # First intermediate region
            F0 = F1 + (F2 - F1) / 3.0
            pAmp['fAmpMatchInt12'] = F0

            V0 = pAmp['IntermediateAmpFits'][modeint*nCollocPtsInterAmp + 8](pWF22, pWFHM['IMRPhenomXHMIntermediateAmpFitsVersion'])
            d0 = pAmp['IntermediateAmpFits'][modeint*nCollocPtsInterAmp + 9](pWF22, pWFHM['IMRPhenomXHMIntermediateAmpFitsVersion'])

            F0_seven_sixths = jnp.power(F0, 7.0/6.0)
            d0 = ((7.0/6.0) / (V0 * F0)) - (d0 / (V0 * V0 * F0_seven_sixths))
            V0 = 1.0 / V0

            # Get coefficients for first intermediate region
            if 'alpha0' not in pAmp:
                pAmp['alpha0'] = 0.0
                pAmp['alpha1'] = 0.0
                pAmp['alpha2'] = 0.0
                pAmp['alpha3'] = 0.0
                pAmp['alpha4'] = 0.0

            pAmp['alpha0'] = IMRPhenomXHM_Intermediate_Amp_delta0(d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104)
            pAmp['alpha1'] = IMRPhenomXHM_Intermediate_Amp_delta1(d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104)
            pAmp['alpha2'] = IMRPhenomXHM_Intermediate_Amp_delta2(d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104)
            pAmp['alpha3'] = IMRPhenomXHM_Intermediate_Amp_delta3(d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104)
            pAmp['alpha4'] = IMRPhenomXHM_Intermediate_Amp_delta4(d1, d0, V1, V2, V3, V0, F1, F2, F3, F0, 104)

            # Update values for second intermediate region
            d1 = d0
            V1 = V0
            F1 = F0

        # Get coefficients for main intermediate region
        version = pWFHM['IMRPhenomXHMIntermediateAmpVersion']

        if 'delta0' not in pAmp:
            pAmp['delta0'] = 0.0
            pAmp['delta1'] = 0.0
            pAmp['delta2'] = 0.0
            pAmp['delta3'] = 0.0
            pAmp['delta4'] = 0.0
            pAmp['delta5'] = 0.0

        pAmp['delta0'] = IMRPhenomXHM_Intermediate_Amp_delta0(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)
        pAmp['delta1'] = IMRPhenomXHM_Intermediate_Amp_delta1(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)
        pAmp['delta2'] = IMRPhenomXHM_Intermediate_Amp_delta2(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)
        pAmp['delta3'] = IMRPhenomXHM_Intermediate_Amp_delta3(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)
        pAmp['delta4'] = IMRPhenomXHM_Intermediate_Amp_delta4(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)

        if version == 105:
            pAmp['delta5'] = IMRPhenomXHM_Intermediate_Amp_delta5(d1, d4, V1, V2, V3, V4, F1, F2, F3, F4, version)

        # Final rescale factors
        pAmp['InspRescaleFactor'] = 0
        pAmp['RDRescaleFactor'] = 0
        pAmp['InterRescaleFactor'] = 0

    return pAmp





def IMRPhenomXHM_GetPhaseCoefficients(
    pAmp: dict,
    pPhase: dict,
    pAmp22: dict,
    pPhase22: dict,
    pWFHM: dict,
    pWF22: dict,
    lalParams: dict = None
) -> dict:
    """
    Compute phase coefficients for IMRPhenomXHM higher modes.

    This function computes all phenomenological phase coefficients for a given higher mode,
    including inspiral (rescaled from 22 mode), intermediate (via collocation), and
    ringdown regions. It handles both spherical and spheroidal mode mixing cases.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients structure for the higher mode
    pPhase : dict
        Phase coefficients structure for the higher mode (will be updated)
    pAmp22 : dict
        Amplitude coefficients for the (2,2) mode
    pPhase22 : dict
        Phase coefficients for the (2,2) mode
    pWFHM : dict
        Waveform structure for the higher mode
    pWF22 : dict
        Waveform structure for the (2,2) mode
    lalParams : dict, optional
        Additional LAL parameters (unused)

    Returns
    -------
    dict
        Updated pPhase dictionary with all computed coefficients
    """


    ell = pWFHM['ell']
    emm = pWFHM['emm']

    # Pre-initialize all phenomenological coefficients
    # Inspiral - will be computed by rescaling IMRPhenomX inspiral coefficients
    pPhase['phi'] = jnp.zeros(16)
    pPhase['phiL'] = jnp.zeros(16)

    # Intermediate - determined by solving linear system at collocation points
    pPhase['c0'] = 0.0
    pPhase['c1'] = 0.0
    pPhase['c2'] = 0.0
    pPhase['c3'] = 0.0
    pPhase['c4'] = 0.0
    pPhase['cL'] = 0.0

    # Ringdown spherical - obtained by rescaling the 22 ringdown parameters
    pPhase['alpha0'] = 0.0
    pPhase['alpha2'] = 0.0
    pPhase['alphaL'] = 0.0

    # Set number of collocation points used (depends on the mode)
    nCollocationPts_inter = pWFHM['nCollocPtsInterPhase']
    eta_m1 = 1.0 / pWF22['eta']
    modeint = pWFHM['modeInt']
    N_MAX_COEFFICIENTS_PHASE_INTER = 6
    # Initialize frequencies of collocation points in intermediate region
    pPhase = IMRPhenomXHM_Intermediate_CollocPtsFreqs(pPhase, pWFHM, pWF22)

    # For each collocation point, call fit giving the value of the phase derivative
    for i in range(N_MAX_COEFFICIENTS_PHASE_INTER):
        pPhase['CollocationPointsValuesPhaseInter'][i] = (
            pPhase['IntermediatePhaseFits'][modeint * N_MAX_COEFFICIENTS_PHASE_INTER + i](
                pWF22, pWFHM['IMRPhenomXHMIntermediatePhaseFitsVersion']
            )
        )
        # Time-shift waveform so that modes peak around t=0
        pPhase['CollocationPointsValuesPhaseInter'][i] += pWFHM['DeltaT']

    fcutRD = pPhase['fPhaseMatchIM']
    fcutInsp = pPhase['fPhaseMatchIN']

    # ============ Inspiral: rescale PhenomX and apply PN corrections ============

    # Collect all the PhenomX inspiral coefficients
    phenXnonLog = jnp.array([
        pPhase22['phi_minus2'], pPhase22['phi_minus1'], pPhase22['phi0'],
        pPhase22['phi1'], pPhase22['phi2'], pPhase22['phi3'],
        pPhase22['phi4'], pPhase22['phi5'], pPhase22['phi6'],
        pPhase22['phi7'], pPhase22['phi8'], pPhase22['phi9'],
        0., 0., 0., 0.
    ])

    phenXLog = jnp.array([
        0., 0., 0., 0., 0., 0., 0.,
        pPhase22['phi5L'], pPhase22['phi6L'], 0.,
        pPhase22['phi8L'], pPhase22['phi9L'],
        0., 0., 0., 0.
    ])

    pseudoPN = jnp.array([
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        pPhase22['sigma1'], pPhase22['sigma2'], pPhase22['sigma3'],
        pPhase22['sigma4'], pPhase22['sigma5'], 0.
    ])

    # Rescale coefficients by applying phi_lm(f) ~ m/2 * phi_22(2/m * f)
    m_over_2 = emm * 0.5
    two_over_m = 1.0 / m_over_2

    fact = pPhase22['phiNorm'] / pWF22['eta']

    phenXnonLog = phenXnonLog * fact
    phenXLog = phenXLog * fact
    pseudoPN = pseudoPN * fact

    # Scaling logarithmic terms introduces extra contributions in non-log terms
    for i in range(16):
        power_idx = (10 - i) / 3.0
        pPhase['phi'] = pPhase['phi'].at[i].set(
            (phenXnonLog[i] + pseudoPN[i] - phenXLog[i] * jnp.log(m_over_2)) *
            jnp.power(m_over_2, power_idx)
        )
        pPhase['phiL'] = pPhase['phiL'].at[i].set(
            phenXLog[i] * jnp.power(m_over_2, power_idx)
        )

    # Add PN amplitude correction to orbital phase
    if pWF22['eta'] > 0.01:
        # Use complex PN amplitudes for non-extreme mass ratios

        pPhase['LambdaPN'] = IMRPhenomXHM_Insp_Phase_LambdaPN(pWF22['eta'], pWFHM['modeTag'])
    else:
        # Use phenomenological fits for extreme mass ratios
        pPhase['LambdaPN'] = pPhase['InspiralPhaseFits'][pWFHM['modeInt']](
            pWF22, pWFHM['IMRPhenomXHMInspiralPhaseVersion']
        )

    pPhase['phi'] = pPhase['phi'].at[10].add(pPhase['LambdaPN'])

    # ============ Intermediate-Ringdown Region ============

    # Choose collocation points according to spin/mass ratio
    cpoints_indices = jnp.array([0, 1, 0, 0, 5], dtype=jnp.int32)

    if (pWF22['eta'] < pWFHM['etaEMR']) or (emm == ell and pWF22['STotR'] >= 0.8) or \
       (pWFHM['modeTag'] == 33 and pWF22['STotR'] < 0):
        cpoints_indices = cpoints_indices.at[2].set(3)
        cpoints_indices = cpoints_indices.at[3].set(4)
    elif pWF22['STotR'] >= 0.8 and pWFHM['modeTag'] == 21:
        cpoints_indices = cpoints_indices.at[2].set(2)
        cpoints_indices = cpoints_indices.at[3].set(4)
    else:
        cpoints_indices = cpoints_indices.at[2].set(2)
        cpoints_indices = cpoints_indices.at[3].set(3)

    # Handle mode mixing cases
    if pWFHM['MixingOn'] == 0:
        # Mode-mixing off: spherical harmonics
        # Compute ringdown coefficients by rescaling
        if pWFHM['ell'] == pWFHM['emm']:
            wlm = 2.0
        else:
            wlm = pWFHM['emm'] / 3.0

        pPhase['alpha2'] = (
            1.0 / (pWFHM['fRING'] ** 2) * wlm *
            IMRPhenomXHM_RD_Phase_22_alpha2(pWF22, pWFHM['IMRPhenomXHMRingdownPhaseVersion'])
        )

        pPhase['alphaL'] = (
            eta_m1 * IMRPhenomXHM_RD_Phase_22_alphaL(pWF22, pWFHM['IMRPhenomXHMRingdownPhaseVersion'])
        )

        # Compute spherical-harmonic phase and derivative at matching frequency
        powers_of_f = compute_powers_of_f(fcutRD)

        pPhase['phi0RD'] = IMRPhenomXHM_RD_Phase_AnsatzInt(fcutRD, powers_of_f, pWFHM, pPhase)
        pPhase['dphi0RD'] = IMRPhenomXHM_RD_Phase_Ansatz(fcutRD, powers_of_f, pWFHM, pPhase)

        # Set up linear system for intermediate region
        A = jnp.zeros((nCollocationPts_inter, nCollocationPts_inter))
        b = jnp.zeros(nCollocationPts_inter)

        for i in range(nCollocationPts_inter):
            ind = cpoints_indices[i]
            b = b.at[i].set(pPhase['CollocationPointsValuesPhaseInter'][ind])
            ff = pPhase['CollocationPointsFreqsPhaseInter'][ind]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            fpowers = jnp.array([
                1.0,
                pWFHM['fDAMP'] / (pWFHM['fDAMP']**2 + (ff - pWFHM['fRING'])**2),
                ffm1,
                ffm2,
                ffm2 * ffm2,
                ffm1 * ffm2
            ])
            for j in range(nCollocationPts_inter):
                A = A.at[i, j].set(fpowers[j])

    elif pWFHM['MixingOn'] == 1:
        # Mode-mixing on (32 mode with spheroidal harmonics)
        # Compute derivatives using finite differences
        fstep = 0.0000001
        SphericalWF = jnp.zeros(3, dtype=jnp.complex128)

        for i in range(3):
            FF = fcutRD + (i - 1) * fstep
            powers_of_FF = compute_powers_of_f(FF)
            SphericalWF = SphericalWF.at[i].set(
                SpheroidalToSpherical(powers_of_FF, pAmp22, pPhase22, pAmp, pPhase, pWFHM, pWF22)
            )

        phase_args = jnp.array([
            jnp.mod(jnp.angle(SphericalWF[0]), 2.0 * jnp.pi),
            jnp.mod(jnp.angle(SphericalWF[1]), 2.0 * jnp.pi),
            jnp.mod(jnp.angle(SphericalWF[2]), 2.0 * jnp.pi)
        ])

        # Ensure all points belong to same branch
        phase_args = jnp.where(phase_args > 0, phase_args - 2.0 * jnp.pi, phase_args)

        pPhase['phi0RD'] = phase_args[1]
        fstep_m1 = 1.0 / fstep
        pPhase['dphi0RD'] = 0.5 * fstep_m1 * (phase_args[2] - phase_args[0])
        d2phi0RD = fstep_m1 * fstep_m1 * (phase_args[2] - 2.0 * phase_args[1] + phase_args[0])

        # Update collocation points with derivatives
        if pWF22['eta'] > pWFHM['etaEMR']:
            pPhase['CollocationPointsFreqsPhaseInter'] = (
                pPhase['CollocationPointsFreqsPhaseInter'].at[nCollocationPts_inter - 2].set(fcutRD)
            )
            pPhase['CollocationPointsValuesPhaseInter'] = (
                pPhase['CollocationPointsValuesPhaseInter'].at[nCollocationPts_inter - 2].set(pPhase['dphi0RD'])
            )

        pPhase['CollocationPointsFreqsPhaseInter'] = (
            pPhase['CollocationPointsFreqsPhaseInter'].at[nCollocationPts_inter - 1].set(fcutRD)
        )
        pPhase['CollocationPointsValuesPhaseInter'] = (
            pPhase['CollocationPointsValuesPhaseInter'].at[nCollocationPts_inter - 1].set(d2phi0RD)
        )

        # Set up linear system
        A = jnp.zeros((nCollocationPts_inter, nCollocationPts_inter))
        b = jnp.zeros(nCollocationPts_inter)

        for i in range(nCollocationPts_inter - 1):
            b = b.at[i].set(pPhase['CollocationPointsValuesPhaseInter'][i])
            ff = pPhase['CollocationPointsFreqsPhaseInter'][i]
            ffm1 = 1.0 / ff
            ffm2 = ffm1 * ffm1
            fpowers = jnp.array([
                1.0,
                pWFHM['fDAMP'] / (pWFHM['fDAMP']**2 + (ff - pWFHM['fRING'])**2),
                ffm1, ffm2, ffm2 * ffm2, ffm1 * ffm2
            ])
            for j in range(nCollocationPts_inter):
                A = A.at[i, j].set(fpowers[j])

        # Last point: second derivative constraint
        cpoint_ind = nCollocationPts_inter - 1
        b = b.at[cpoint_ind].set(pPhase['CollocationPointsValuesPhaseInter'][cpoint_ind])
        ff = pPhase['CollocationPointsFreqsPhaseInter'][cpoint_ind]
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        ffm3 = ffm2 * ffm1
        ffm4 = ffm2 * ffm2
        ffm5 = ffm3 * ffm2

        fpowers = jnp.array([
            0.0,
            -2.0 * pWFHM['fDAMP'] * (ff - pWFHM['fRING']) /
            ((pWFHM['fDAMP']**2 + (ff - pWFHM['fRING'])**2)**2),
            -ffm2, -2.0 * ffm3, -4.0 * ffm5, -3.0 * ffm4
        ])
        for j in range(nCollocationPts_inter):
            A = A.at[cpoint_ind, j].set(fpowers[j])

    # Solve linear system A x = b
    x = jnp.linalg.solve(A, b)

    pPhase['c0'] = x[0]
    pPhase['cL'] = x[1]
    pPhase['c1'] = x[2]
    pPhase['c2'] = x[3]
    pPhase['c4'] = x[4]

    # Add PNR deviations for 33 mode
    if pWFHM['modeTag'] == 33:
        pPhase['c0'] += pWFHM['PNR_DEV_PARAMETER'] * pWFHM['NU0']
        pPhase['cL'] += pWFHM['PNR_DEV_PARAMETER'] * pWFHM['NU4']
        pPhase['c1'] += pWFHM['PNR_DEV_PARAMETER'] * pWFHM['ZETA2']
        pPhase['c4'] += pWFHM['PNR_DEV_PARAMETER'] * pWFHM['ZETA1']

    # 32 mode uses one extra collocation point
    if pWFHM['modeTag'] == 32:
        pPhase['c3'] = x[5]

        # Glue intermediate and ringdown for extreme mass ratios
        if pWF22['eta'] < pWFHM['etaEMR']:
            powers_of_f = compute_powers_of_f(fcutRD)
            pPhase['c0'] += (
                pPhase['dphi0RD'] -
                IMRPhenomXHM_Inter_Phase_Ansatz(fcutRD, powers_of_f, pWFHM, pPhase)
            )

    # Glue inspiral and intermediate regions
    powers_of_f = compute_powers_of_f(fcutInsp)


    



    pPhase['C1INSP'] = (
        IMRPhenomXHM_Inter_Phase_Ansatz(fcutInsp, powers_of_f, pWFHM, pPhase) -
        IMRPhenomXHM_Inspiral_Phase_Ansatz(fcutInsp, powers_of_f, pPhase)
    )


    pPhase['CINSP'] = (
        -pPhase['C1INSP'] * fcutInsp +
        IMRPhenomXHM_Inter_Phase_AnsatzInt(fcutInsp, powers_of_f, pWFHM, pPhase) -
        IMRPhenomXHM_Inspiral_Phase_AnsatzInt(fcutInsp, powers_of_f, pPhase)
    )

    # Glue ringdown and intermediate regions
    powers_of_f = compute_powers_of_f(fcutRD)
    pPhase['C1RD'] = (
        IMRPhenomXHM_Inter_Phase_Ansatz(fcutRD, powers_of_f, pWFHM, pPhase) -
        pPhase['dphi0RD']
    )
    pPhase['CRD'] = (
        -pPhase['C1RD'] * fcutRD +
        IMRPhenomXHM_Inter_Phase_AnsatzInt(fcutRD, powers_of_f, pWFHM, pPhase) -
        pPhase['phi0RD']
    )

    # Align each mode so low-frequency relative phase matches PN
    if pWF22['eta'] > pWFHM['etaEMR']:
        falign = 0.6 * m_over_2 * pWF22['fMECO']
    else:
        falign = m_over_2 * pWF22['fMECO']

    powers_of_falign = compute_powers_of_f(falign)
    powers_of_f = compute_powers_of_f(two_over_m * falign)

    # Compute phase normalization for spherical modes
    if pWFHM['MixingOn'] == 0:
        powers_of_MfRef = compute_powers_of_f(pWF22['MfRef'])
        pPhase22 = IMRPhenomX_Phase_22_ConnectionCoefficients(pWF22, pPhase22)
        pWFHM['timeshift'] = IMRPhenomX_TimeShift_22(pPhase22, pWF22)
        pWFHM['phiref22'] = (
            -1.0 / pWF22['eta'] * IMRPhenomX_Phase_22(pWF22['MfRef'], powers_of_MfRef, pPhase22, pWF22) -
            pWFHM['timeshift'] * pWF22['MfRef'] - pWFHM['phaseshift'] +
            2.0 * pWF22['phi0'] + jnp.pi / 4.0
        )

    # Phase normalization from Eq. (4.13)
    deltaphiLM = (
        m_over_2 * (
            1.0 / pWF22['eta'] * IMRPhenomX_Phase_22(two_over_m * falign, powers_of_f, pPhase22, pWF22) +
            pWFHM['phaseshift'] + pWFHM['phiref22']
        ) +
        pWFHM['timeshift'] * falign -
        3.0 * jnp.pi / 4.0 * (1.0 - m_over_2) -
        (IMRPhenomXHM_Inspiral_Phase_AnsatzInt(falign, powers_of_falign, pPhase) +
         pPhase['C1INSP'] * falign + pPhase['CINSP'])
    )
    pPhase['deltaphiLM'] = jnp.mod(deltaphiLM, 2.0 * jnp.pi)

    # For 21 mode, account for sign flip in PN amplitude
    if pWFHM['modeTag'] == 21:
        ampsign = IMRPhenomXHM_PN21AmpSign(0.008, pWF22)
        if ampsign > 0:
            pPhase['deltaphiLM'] += jnp.pi

    return pPhase



def IMRPhenomXHM_Ringdown_CollocPtsFreqs(pPhase: dict, pWFHM: dict, pWF22: dict) -> dict:
    """
    Initialize frequencies of collocation points for spheroidal ringdown reconstruction.

    This function initializes the frequencies stored in the pPhase struct, which gets
    updated with new values as the code processes each mode.

    Args:
        pPhase: Phase coefficient dictionary for the higher mode
        pWFHM: Waveform structure dictionary for the higher mode
        pWF22: Waveform structure dictionary for the (2,2) mode

    Returns:
        dict: Updated pPhase dictionary with collocation point frequencies
    """
    # Extract ringdown and damping frequencies for the mode
    fringlm = pWFHM['fRING']
    fdamplm = pWFHM['fDAMP']
    fring22 = pWF22['fRING']

    # Initialize collocation points array if not present
    if 'CollocationPointsFreqsPhaseRD' not in pPhase:
        # Maximum size needed based on version 122022 (5 points)
        pPhase['CollocationPointsFreqsPhaseRD'] = jnp.zeros(5)

    # Switch based on version
    version = pWFHM['IMRPhenomXHMRingdownPhaseFreqsVersion']

    def case_122019():
        """Version 122019: 4 collocation points"""
        freqs = jnp.zeros(5)
        freqs = freqs.at[0].set(fring22)
        freqs = freqs.at[2].set(fringlm - 0.5 * fdamplm)
        freqs = freqs.at[1].set(fringlm - 1.5 * fdamplm)
        freqs = freqs.at[3].set(fringlm + 0.5 * fdamplm)
        return freqs

    def case_122022():
        """Version 122022: 5 collocation points"""
        fdamp22 = pWF22['fDAMP']
        freqs = jnp.zeros(5)
        freqs = freqs.at[0].set(fring22 - fdamp22)
        freqs = freqs.at[1].set(fring22)
        freqs = freqs.at[2].set((fring22 + fringlm) * 0.5)
        freqs = freqs.at[3].set(fringlm)
        freqs = freqs.at[4].set(fringlm + fdamplm)
        return freqs

    def default_case():
        """Default case: return NaN (error condition)"""
        # In JAX we can't raise errors in jit-compiled code
        # Return array of NaNs to signal error
        return jnp.full(5, jnp.nan)

    # Use lax.switch for conditional branching
    freqs = jax.lax.switch(
        jnp.where(version == 122019, 0,
        jnp.where(version == 122022, 1, 2)),
        [case_122019, case_122022, default_case]
    )

    pPhase['CollocationPointsFreqsPhaseRD'] = freqs

    return pPhase


def RescaleFactor(powers_of_Mf: dict, pAmp: dict, rescalefactor: int) -> float:
    """
    Compute the rescaling factor for amplitude normalization.

    This function returns different rescaling factors depending on the rescalefactor
    parameter. It is used to convert between different amplitude normalizations:
    - 0: Strain (no rescaling)
    - 1: 22 mode factor
    - 2: lm mode factor (mode-dependent)

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pAmp : dict
        Amplitude coefficients structure containing:
        - ampNorm: Amplitude normalization
        - PNdominant: PN dominant term
        - PNdominantlm: Mode-specific PN dominant term
        - PNdominantlmpower: Power for the mode-specific term (1, 2, 3, or 4)
    rescalefactor : int
        Rescale factor version (0, 1, or 2)

    Returns
    -------
    float
        The rescaling factor

    Raises
    ------
    ValueError
        If rescalefactor is not 0, 1, or 2
    """
    if rescalefactor == 0:  # Strain
        factor = 1.0

    elif rescalefactor == 1:  # 22factor
        factor = pAmp['ampNorm'] * powers_of_Mf['m_seven_sixths']

    elif rescalefactor == 2:  # lmfactor
        PNdominantlmpower = pAmp['PNdominantlmpower']
        base_factor = pAmp['PNdominant'] * powers_of_Mf['m_seven_sixths'] * pAmp['PNdominantlm']

        if PNdominantlmpower == 1:
            factor = base_factor * powers_of_Mf['one_third']
        elif PNdominantlmpower == 2:
            factor = base_factor * powers_of_Mf['two_thirds']
        elif PNdominantlmpower == 3:
            factor = base_factor * powers_of_Mf['itself']
        elif PNdominantlmpower == 4:
            factor = base_factor * powers_of_Mf['four_thirds']
        else:
            raise ValueError(
                f"Error in RescaleFactor: PNdominantlmpower {PNdominantlmpower} is not valid. "
                f"Valid values are 1, 2, 3, or 4."
            )

    else:
        raise ValueError(
            f"Error in RescaleFactor: version {rescalefactor} is not valid. "
            f"Recommended version is 1."
        )

    return factor


def IMRPhenomXHM_Intermediate_CollocPtsFreqs(pPhase, pWFHM, pWF22):
    #TODO
    return pPhase


def SpheroidalToSpherical():
    return #TODO

def IMRPhenomXHM_PN21AmpSign():
    return 