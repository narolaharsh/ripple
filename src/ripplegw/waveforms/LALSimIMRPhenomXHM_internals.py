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

    Note:
        The fdamp functions are currently placeholders (None) as they are not
        yet implemented in the Python translation. They should be added when
        evaluate_QNMfit_fdamp21/33/32/44 functions become available.
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
