
import jax
import jax.numpy as jnp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_delta0,
    get_delta1,
    get_delta2,
    get_delta3,
    get_delta4,
    get_transition_frequencies,
)

from .IMRPhenomD_QNMdata import fM_CUT
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ..typing import Array
from ripplegw import Mc_eta_to_ms
from .spherical_harmonics import (compute_sminus2_l2, compute_sminus2_l3, compute_sminus2_l4)


def test_link():
    print('link OK')
    return


'''
Steps to construct the XPHM waveform from relative binning project

### L-frame strain of each mode
    lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode

### C  prefactors
    lalsim.SimIMRPhenomXPMSAAngles
    

    
### Order of  C prefactor computations:
    1. Euler angles + twist_factor = C prefactors

    2. Euler angles ingredients:
        A. lalsim.SimIMRPhenomXPMSAAngles

    3. Twist factor ingredients
        A. Transfer functions
            a. Wigner coefficients
        B. Wigner coefficients
        C. y2lm


### multiply twisting up factors with the L-frame mode and sum all modes

'''




def compute_zeta(params: Array):

    zeta = None
    return zeta




def compute_m2ylm(l: int, m: int, theta_jn: float):
    """
    Computes the -2 spin weighted spherical harmonic evaluated at theta = theta_jn, phi = 0.
    l: l index
    m: m index
    theta_jn: theta_jn angle
    """

    m2ylm = jnp.where(l==2, compute_sminus2_l2(theta_jn, m), 
                      jnp.where(l==3, compute_sminus2_l3(theta_jn, m), 
                                jnp.where(l==4, compute_sminus2_l4(theta_jn, m), jnp.nan)
                                )
                    )

    return m2ylm

def compute_transfer_function(l: float, m: float, mprime: float, alpha: float, beta: float, theta_jn: float):

    pos_wigner_coefficient = compute_wigner_coefficient(l, m, mprime, beta)
    neg_wigner_coefficient = compute_wigner_coefficient(l, -m, mprime, beta)
    negative_power = (-1)**(l+m)



    term_a = jnp.exp(-1j*m*alpha) * pos_wigner_coefficient[0] * compute_m2ylm(l, m, theta_jn)

    term_b = negative_power * jnp.exp(-1j*m*alpha) * pos_wigner_coefficient[1] * compute_m2ylm(l, m, theta_jn)

    term_c = jnp.exp(1j*m*alpha) * neg_wigner_coefficient[0] * compute_m2ylm(l, -m, theta_jn)

    term_d = negative_power * jnp.exp(1j*m*alpha) * neg_wigner_coefficient[1] * compute_m2ylm(l, -m, theta_jn)

    return term_a, term_b, term_c, term_d


def compute_wigner_coefficient():
    return None


def compute_twist_factor_plus_cross(l: float, mprime: float, theta_jn: float, alpha: Array, beta: Array, gamma: Array):

    m_array = jnp.arange(1, l+1, 1)
    plus_summand = 0
    cross_summand = 0

    for m in m_array:
        transfer = compute_transfer_function(l, m, mprime, alpha, beta, theta_jn)
        term_1 = transfer[1] + transfer[3]
        term_2 = ((-1)**l)*jnp.conj(transfer[0] + transfer[2])
        plus_summand += term_1 + term_2
        cross_summand += term_1 - term_2

    
    wigner_coefficient = compute_wigner_coefficient(l, 0, mprime, beta)

    term_1 = ((-1)**l) * wigner_coefficient[1] * compute_m2ylm(l, 0, theta_jn)
    term_2 = ((-1)**l) * wigner_coefficient[0] * compute_m2ylm(l, 0, theta_jn)


    plus_summand += term_1 + term_2
    cross_summand += term_1 - term_2

    return 0.5*jnp.exp(1*mprime*gamma)*plus_summand, 1j*0.5*jnp.exp(1*mprime*gamma)*cross_summand


def compute_c_prefactors(f: Array, params: Array, X: float):

    c_plus_j, c_cross_j =  compute_twist_factor_plus_cross()

    zeta = compute_zeta(params)

    c_plus = jnp.cos(2*zeta)*c_plus_j + jnp.sin(2*zeta)*c_cross_j
    
    c_cross = jnp.cos(2*zeta)*c_cross_j - jnp.sin(2*zeta)*c_plus_j


    return c_plus, c_cross




def generate_hlframe_waveform(f: Array, params: Array, f_ref: float, l: float, m: float):

    hlframe = None
    return hlframe


def gen_IMRPhenomXPHM_hphc(f: Array, params: Array, f_ref: float):
    hp = 0
    hc = 0
    return hp, hc

