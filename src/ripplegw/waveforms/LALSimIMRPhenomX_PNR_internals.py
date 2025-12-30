import jax.numpy as jnp
import jax
from ..constants import MTSUN_SI
from .LALSimIMRPhenomX_precession import XLALSimIMRPhenomXLPNAnsatz, IMRPhenomX_Initialize_MSA_System


def IMRPhenomX_PNR_HMInterpolationDeltaF(f_min: float, pWF: dict, pPrec: dict) . float:
    """
    Compute deltaF required for PNR HM interpolation
    
    Args:
        f_min: minimum starting frequency (Hz) (float)
        pWF: PhenomX waveform struct (dict)
        pPrec: PhenomX precession struct (dict)
        
    Returns:
        float: Delta frequency for interpolation
    """
    
    error_tolerance = pPrec['IMRPhenomXPNRInterpTolerance']
    
    # aligned-spin limit means we don't need high resolution on the prec angles (which should be zero)
    aligned_spin_condition = jnp.logical_and(
        pPrec['chi1_perp'] == 0.0,
        pPrec['chi2_perp'] == 0.0
    )
    
    def aligned_spin_case():
        return jnp.where(pWF['deltaF'] != 0, pWF['deltaF'], 0.1)
    
    def precessing_case():
        # Compute approximate scaling from leading-order alpha contribution
        Mf = XLALSimPhenomUtilsHztoMf(f_min, pWF['Mtot'])
        eta_term = jnp.sqrt(1.0 - 4.0 * pWF['eta'])
        numerator = 3.0 * jnp.pi * jnp.power(Mf, 5) * error_tolerance * (1.0 + eta_term)
        denominator = 7.0 + 13.0 * eta_term
        constant = 4.0 * jnp.sqrt(2.0 / 5.0)
        
        deltaF_alpha = XLALSimPhenomUtilsMftoHz(
            constant * jnp.sqrt(jnp.sqrt(numerator / denominator)), 
            pWF['Mtot']
        )
        
        # Check for two-spin oscillations
        two_spin_condition = IMRPhenomX_PNR_CheckTwoSpin(pPrec)  # stub
        
        def handle_two_spin():
            precessing_tag = (pPrec['IMRPhenomXPrecVersion'] - 
                            (pPrec['IMRPhenomXPrecVersion'] % 100)) // 100
            
            # If SpinTaylor version (tag == 3), we need to set up L coefficients
            # This is just setup - the main logic happens regardless
            def setup_spintaylor():
                eta = pPrec.eta
                delta = pWF['delta']
                chi1L = pPrec.chi1L
                chi2L = pPrec.chi2L

                pPrec.L0   = 1.0
                pPrec.L1   = 0.0
                pPrec.L2   = 3.0/2. + eta/6.0
                pPrec.L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
                pPrec.L4   = (81 + (-57 + eta)*eta)/24.
                pPrec.L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
                pPrec.L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*jnp.power(jnp.pi, 2)))/1296.
                pPrec.L7   = 0.0
                pPrec.L8   = 0.0
                #// This is the log(x) term
                pPrec.L8L  = 0.0

                user_version = pPrec.IMRPhenomXPrecVersion

                pPrec.IMRPhenomXPrecVersion = 223

                IMRPhenomX_Initialize_MSA_System(pWF, pPrec, pPrec.ExpansionOrder)

                pPrec.IMRPhenomXPrecVersion = user_version
                return pPrec
                
            def no_setup():
                return pPrec
                
            # Do setup if needed (this just returns pPrec for now)
            pPrec_updated = jax.lax.cond(
                precessing_tag == 3,
                setup_spintaylor,
                no_setup
            )
            
            # NOW do the main two-spin logic (this happens regardless of precessing_tag)
            # Use precomputed MSA terms
            g0 = pPrec_updated['g0']
            deltam = pPrec_updated['delta_qq'] 
            psi1 = pPrec_updated['psi1']
            psi2 = pPrec_updated['psi2']
            
            v0 = jnp.power(jnp.pi * Mf, 1.0/3.0)
            v02 = v0 * v0
            v06 = v02 * v02 * v02
            
            # frequency derivative of psi, Eq. 51 of arXiv:1703.03967
            dpsi = g0 * deltam * jnp.pi / (4.0 * v06) * (3.0 + 2.0 * psi1 * v0 + psi2 * v02)
            dpsiInv = jnp.abs(1.0 / dpsi)
            
            # Compute L_fmin for beta calculations
            L_fmin = (pWF['Mtot'] * pWF['Mtot'] * 
                     XLALSimIMRPhenomXLPNAnsatz(v0, pWF['eta'] / v0, 
                                               pPrec_updated['L0'], pPrec_updated['L1'], pPrec_updated['L2'], 
                                               pPrec_updated['L3'], pPrec_updated['L4'], pPrec_updated['L5'], 
                                               pPrec_updated['L6'], pPrec_updated['L7'], pPrec_updated['L8'], 
                                               pPrec_updated['L8L']))  # stub
            
            betaMin = jnp.arctan2(
                jnp.abs(pPrec_updated['S1_perp'] - pPrec_updated['S2_perp']), 
                L_fmin + pPrec_updated['SL']
            )
            betaMax = jnp.arctan2(
                pPrec_updated['S1_perp'] + pPrec_updated['S2_perp'], 
                L_fmin + pPrec_updated['SL']
            )
            
            # Adjust dpsiInv if conditions are met
            large_oscillation_condition = jnp.logical_and(
                betaMin < 0.01,
                betaMin / betaMax < 0.55
            )
            dpsiInv = jnp.where(large_oscillation_condition, dpsiInv / 4, dpsiInv)
            
            # Sample 4 points per oscillation
            deltaF_twospin = XLALSimPhenomUtilsMftoHz(dpsiInv / 4.0, pWF['Mtot'])
            
            # Check if two-spin deltaF is smaller and valid
            use_twospin_condition = jnp.logical_and(
                deltaF_twospin < deltaF_alpha,
                jnp.logical_not(jnp.isnan(dpsi))
            )
            
            # If we should use two-spin deltaF, apply hard limit and return
            # Otherwise, fall through to use deltaF_alpha
            return jnp.where(
                use_twospin_condition,
                jnp.maximum(deltaF_twospin, 1e-2),  # hard limit
                jnp.maximum(deltaF_alpha, 1e-2)    # fall back to alpha with hard limit
            )
        
        def no_two_spin():
            # Apply hard limit of 0.01 to deltaF_alpha
            return jnp.maximum(deltaF_alpha, 1e-2)
        
        return jax.lax.cond(two_spin_condition, handle_two_spin, no_two_spin)
    
    return jax.lax.cond(aligned_spin_condition, aligned_spin_case, precessing_case)



def IMRPhenomX_PNR_CheckTwoSpin(pPrec: dict) . bool:
    """
    Check for two-spin system conditions
    
    Args:
        pPrec: PhenomX precession struct (dict)
        
    Returns:
        bool: True if two-spin conditions are met, False otherwise
    """
    
    # Ensure we have:
    # - non-zero spin on the primary
    # - non-trivial spin on the secondary  
    # - activated the MSA angles
    condition = jnp.logical_and(
        jnp.logical_and(
            pPrec['chi1_norm'] != 0.0,
            pPrec['chi2_norm'] >= 1.0e-3
        ),
        pPrec['IMRPhenomXPrecVersion'] >= 200
    )
    
    return condition

def XLALSimPhenomUtilsHztoMf(fHz: float, Mtot: float) . float:
    """Stub: Convert Hz to dimensionless frequency"""
    # Implementation needed - likely similar to XLALSimIMRPhenomXUtilsHztoMf
    return fHz * (MTSUN_SI * Mtot)

def XLALSimPhenomUtilsMftoHz(Mf: float, Mtot: float) . float:
    return Mf / (MTSUN_SI * Mtot)

