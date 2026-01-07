
from .LALSimIMRPhenomX_internals import IMRPhenomXSetWaveformVariables
import jax.numpy as jnp
from .LALSimIMRPhenomX_precession import IMRPhenomXGetAndSetPrecessionVariables
from .LALSimIMRPhenomX_precession import IMRPhenomX_Return_phi_zeta_costhetaL_MSA

def XLALSimIMRPhenomXPHM(m1_SI, m2_SI,
                         chi1x, chi1y, chi1z,
                         chi2x, chi2y, chi2z,
                         distance, inclination, phiRef,
                         f_min, f_max, deltaF, fRef_In, lalParams):
    
    #Check what f_max_prime is doing. 
    
    pWF = IMRPhenomXSetWaveformVariables(m1_SI, m2_SI, chi1z, chi2z, 
                                         deltaF, fRef_In, phiRef, f_min, f_max, 
                                         distance, inclination, lalParams)
    
    samples_frequencies = jnp.arange(f_min, f_max, deltaF) # Double check 284-285

    pPrec = IMRPhenomXGetAndSetPrecessionVariables(pWF,
                                                   m1_SI,
                                                   m2_SI,
                                                   chi1x,
                                                   chi1y,
                                                   chi1z,
                                                   chi2x,
                                                   chi2y,
                                                   chi2z,
                                                   lalParams,
                                                   debug_flag=False
                                                   )
    
    
    hplus, hcross = IMRPhenomXPHM_hplushcross(samples_frequencies, pWF, pPrec, lalParams)


    ## FIXME Resizing routine ###

    return hplus, hcross


def IMRPhenomXPHM_hplushcross(frequency_array, pWF, pPrec, lalParams):
    # Line 741

    # skip 751 - 754


    L_MAX = 4

    for ell in jnp.arange(2, L_MAX+1, 1):
        print(ell)
        for emmprime in jnp.arange(1, ell+1, 1):
            cond = (pWF['q']==1) & (pWF['chi1L']==pWF['chi2L']) & (emmprime%2 !=0)
            if cond:
                "Skip twisting up"
                continue

            if (ell==3) and (emmprime==2):
                htildelm = IMRPhenomXHMMultiBandOneModeMixing(pWF, ell, emmprime, lalParams)

            else:
                htildelm = IMRPhenomXHMMultiBandOneMode(pWF, ell, emmprime, lalParams)

            # To be implemented
            #/* If the 22 and 32 modes are active, we recycle the 22 mode for the mixing in the 32 and it is passed to IMRPhenomXHMMultiBandOneModeMixing.
            #The 22 mode is always computed first than the 32, we store the 22 mode in the variable htilde22. */


            ########             TWISTING UP         #############
            #Transform modes from the precessing L-frame to inertial J-frame.

            ########        USING MBAND FOR ANGLES     #############

            coarseFreqs = XLALSimIMRPhenomXPHMMultibandingGrid(ell, emmprime, pWF, lalParams)
            v = jnp.cbrt(jnp.pi * coarseFreqs * 2  / emmprime)
            vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v, pWF, pPrec)

            alpha_offset_mprime, epsilon_offset_mprime = Get_alpha_epsilon_offset(emmprime, pPrec)

            valpha = vangles[0] - alpha_offset_mprime
            vepsilon = vangles[1] - epsilon_offset_mprime
            cos_beta = vangles[2]

            cBetah, sBetah = IMRPhenomXWignerdCoefficients_cosbeta(cos_beta)
            vbeta = jnp.acos(cBetah)

            # j < len(coarseFreqs)-1 and finecount < istop
            # For loop for interpolatio #TODO

            #Now we have the complex exponentials of the three Euler angles alpha, beta, epsilon evaluated in the fine frequency grid.
            #Next step is do the twisting up with these.

            #/************** TWISTING UP in the fine grid *****************/

            IMRPhenomXPHMTwistUp(Mf, hlmcoprec, pWF, pPrec, ell, emmprime)

    
    
    hp = None
    hc = None

    return hp, hc


def IMRPhenomXHMMultiBandOneModeMixing(pWF, ell, emmprime, lalParams):
    htildelm = None
    return htildelm

def IMRPhenomXHMMultiBandOneMode(pWF, ell, emmprime, lalParams):
    htildelm = None
    return htildelm


def XLALSimIMRPhenomXPHMMultibandingGrid(ell, emmprime, pWF, lalParams):
    return None

def Get_alpha_epsilon_offset(emmprime, pPrec):
    return None, None