
from .LALSimIMRPhenomX_internals import IMRPhenomXSetWaveformVariables
import jax.numpy as jnp
from .LALSimIMRPhenomX_precession import IMRPhenomXGetAndSetPrecessionVariables

def XLALSimIMRPhenomXPHM(m1_SI, 
                         m2_SI,
                         chi1x,
                         chi1y,
                         chi1z,
                         chi2x,
                         chi2y,
                         chi2z,
                         distance,
                         inclination,
                         phiRef,
                         f_min,
                         f_max,
                         deltaF,
                         fRef_In,
                         lalParams):
    
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
                                                   lalParams
                                                   )
    
    
    hplus, hcross = IMRPhenomXPHM_hplushcross(samples_frequencies, pWF, pPrec, lalParams)


    ## FIXME Resizing routine ###

    return hplus, hcross


def IMRPhenomXPHM_hplushcross(frequency_array, pWF, pPrec):

    return hp, hc