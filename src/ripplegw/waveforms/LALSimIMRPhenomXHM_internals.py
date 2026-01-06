import jax.numpy as jnp
from .LALSimIMRPhenomTHM_fits import (evaluate_QNMfit_fring21, 
                                      evaluate_QNMfit_fring33,
                                      evaluate_QNMfit_fring32,
                                      evaluate_QNMfit_fring44
                                      )
import jax

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


