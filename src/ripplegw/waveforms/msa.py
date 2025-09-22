import jax
import jax.numpy as jnp
import math
from ..typing import Array



"""
Remaining function in this module:

gsl_sf_elljac_e

This does not exist in numpy or jax.scipy.special.
"""

def XLALSimIMRPhenomXPMSAAngles(freq: Array, params: Array, f_ref: float, mprime: int):
    ## XLALSimIMRPhenomXPMSAAngles

    '''
    lalsim.SimIMRPhenomXPMSAAngles(f_seq,
                                params['mass_1'] * lal.MSUN_SI,
                                params['mass_2'] * lal.MSUN_SI,
                                params['chi_1x'], params['chi_1y'], params['chi_1z'],
                                params['chi_2x'], params['chi_2y'], params['chi_2z'],
                                0,   ### this is inclination which is set to zero. It needs to be set to params['inclination'] if you want to use PhenomPNR waveforms
                                waveform_arugments['reference_frequency'], mprime,
                                lalDict_MSA);

    '''

    default_prec_version = 221
    piGM = 1
    pWF = None
    pPrec = None
    v = freq * piGM * (2/mprime)   ### v is the frequency. Not velocity. To be double checked
    v = v**(1/3)
    vangles = jax.vmap(lambda v_: IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v_, pWF, pPrec))(v)



    alpha_of_f = vangles.x - pPrec.alpha_offset
    gamma_of_f = -1 * (vangles.y - pPrec.epsilon_offset)
    cosbeta_of_f = vangles.z

    return alpha_of_f, gamma_of_f, cosbeta_of_f


def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v: Array, pWF, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a8a1f3cefac1e80942b2848ca0133bcbc

    v: Velocity
    pWF: IMRPhenomX waveform struct  
    pPrec: IMRPhenomX precession struct
    """

    vout = {0.,0.,0.}

    L_norm = pWF.eta / v
    J_norm = IMRPhenomX_JNorm_MSA(L_norm, pPrec)

    L_norm3PN = 0.0

    if pPrec.IMRPhenomXPrecVersion == 'blah':
        L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)
    else:
        print('Do somethig else')
    

    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN,pPrec)


    vRoots = IMRPhenomX_Return_Roots_MSA(L_norm,J_norm,pPrec)


    pPrec.S32 = vRoots.x
    pPrec.Smi2 = vRoots.y
    pPrec.Spl2 = vRoots.z

    ### define following four from the three above
    pPrec.Spl2mSmi2 
    pPrec.Spl2pSmi2 
    pPrec.Spl
    pPrec.Smi

    ### New function
    SNorm = IMRPhenomX_Return_SNorm_MSA(v,pPrec)
    pPrec.S_norm      = SNorm
    pPrec.S_norm_2    = SNorm * SNorm

    vMSA = {0.,0.,0.}
    if jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5:
        vMSA = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
 


    phiz_MSA     = vMSA.x
    zeta_MSA     = vMSA.y

    phiz         = IMRPhenomX_Return_phiz_MSA(v,J_norm,pPrec)
    zeta         = IMRPhenomX_Return_zeta_MSA(v,pPrec)
    cos_theta_L        = IMRPhenomX_costhetaLJ(L_norm3PN,J_norm3PN,SNorm)
    
    vout_x = phiz + phiz_MSA
    vout_y = zeta + zeta_MSA
    vout_z = cos_theta_L


    return vout_x, vout_y, vout_z


def IMRPhenomX_JNorm_MSA(LNorm: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#aca1fc9b93acf010a788cdc02cfb92631
    """
    JNorm2 = (LNorm*LNorm + (2.0 * LNorm * pPrec.c1_over_eta) + pPrec.SAv2)
    return JNorm2


def IMRPhenomX_Return_Roots_MSA(LNorm: float, JNorm: float, pPrec):

    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a3d6fc6f6edff39a14e884521a3cd2997
    """
    vout = None

    tmp1 = 0.0
    tmp2 = 0.0
    tmp3 = 0.0
    tmp4 = 0.0
    tmp5 = 0.0
    tmp6 = 0.0

    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)

    B  = vBCD.x
    C  = vBCD.y
    D  = vBCD.z

    S1Norm2 = pPrec.S1_norm_2
    S2Norm2 = pPrec.S2_norm_2
    
    S0Norm2 = pPrec.S_0_norm_2
    
    B2 = B  * B
    B3 = B2 * B
    BC = B  * C
    
    p  = C - B2 / 3
    qc = (2.0/27.0)*B3 - BC/3.0 + D
    
    sqrtarg = jnp.sqrt(-p/3.0)
    acosarg = 1.5 * qc/p/sqrtarg

    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta     = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)
 
    dotS1Ln = pPrec.dotS1Ln
    dotS2Ln = pPrec.dotS2Ln
    

    ############# If block condition ##########################


    condition = ((theta != theta) | (sqrtarg != sqrtarg) | (dotS1Ln == 1) | (dotS2Ln == 1) | (dotS1Ln == -1) | (dotS2Ln == -1) | (S1Norm2 == 0) | (S2Norm2 == 0))

    ############# else block for the condition above ##########################

    tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * 2.0 * jnp.pi / 3.0) - B / 3.0
    tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2*jnp.pi / 3.0) - B / 3.0
    tmp3 = 2.0 * sqrtarg * jnp.cos_theta - B / 3.0 
    tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
    tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)
    cond_tmp6_tmp3 = (tmp4 - tmp3) > 0.0 & (tmp5 - tmp3) < 0.0
    cond_tmp6_tmp1 = (tmp4 - tmp1) > 0.0 & (tmp5 - tmp1) < 0.0
    tmp6 = jnp.where(cond_tmp6_tmp3, tmp3, jnp.where(cond_tmp6_tmp1, tmp1, tmp2))

    tmp4 = jnp.abs(tmp4)
    tmp6 = jnp.abs(tmp6)

    ############# All possibilities evalusted. Now assign the values ##########################

    S32  = jnp.where(condition, 0.0, tmp5)
    Smi2 = jnp.where(condition, S0Norm2, tmp6)
    Spl2 = jnp.where(condition, Smi2 + 1e-9, tmp4)

    return S32, Smi2, Spl2



def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm: float, JNorm: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a1d96ad5bf7cd20cc0426c0fb6c5135ca
    """
    JNorm2  = JNorm * JNorm
    LNorm2  = LNorm * LNorm
    S1Norm2 = pPrec.S1_norm_2
    S2Norm2 = pPrec.S2_norm_2


    q       = pPrec.qq
    eta     = pPrec.eta
    
    J2mL2   = (JNorm2 - LNorm2)
    J2mL2Sq = J2mL2 * J2mL2
    
    delta   = pPrec.delta_qq
    deltaSq = delta*delta

    Seff    = pPrec.Seff

    vout_x = (LNorm2 + S1Norm2)*q + 2.0*LNorm*Seff - 2.0*JNorm2 - S1Norm2 - S2Norm2 + (LNorm2 + S2Norm2)/q
    vout_y = J2mL2Sq - 2.0*LNorm*Seff*J2mL2 - 2.0*((1.0 - q)/q)*LNorm2*(S1Norm2 - q*S2Norm2) + 4.0*eta*LNorm2*Seff*Seff - 2.0*delta*(S1Norm2 - S2Norm2)*Seff*LNorm + 2.0*((1.0 - q)/q)*(q*S1Norm2 - S2Norm2)*JNorm2
    vout_z = ((1.0 - q)/q)*(S2Norm2 - q*S1Norm2)*J2mL2Sq + deltaSq*(S1Norm2 - S2Norm2)*(S1Norm2 - S2Norm2)*LNorm2/eta + 2.0*delta*LNorm*Seff*(S1Norm2 - S2Norm2)*J2mL2

    return vout_x, vout_y, vout_z


def IMRPhenomX_L_norm_3PN_of_v(v: float, v2: float, L_norm: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#aa4bb5fdf7feb7f787c02ce5d4e6b4b88
    """
    return L_norm * (1. + v2*(pPrec.constants_L[0] + v*pPrec.constants_L[1] + v2*(pPrec.constants_L[2] + v*pPrec.constants_L[3] + v2*(pPrec.constants_L[4]))))


def IMRPhenomX_Return_SNorm_MSA(v: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a5620d4146607de6696bf42bd346adcbf

    remaining functions: gsl_sf_elljac_e
    """

    v2 = v*v
 
    #If spin norms ~ cancel then we do not need to evaluate the Jacobi elliptic function. Check tolerance?

    if jnp.abs(pPrec.Smi2 - pPrec.Spl2) < 1.0e-5:

        sn = 0.0

    else:

        m   = (pPrec.Smi2 - pPrec.Spl2) / (pPrec.S32 - pPrec.Spl2)
 
        psi = IMRPhenomX_psiofv(v, v2, pPrec.psi0, pPrec.psi1, pPrec.psi2, pPrec)
 
        # Evaluate the Jacobi ellptic functions
        sn, cn, dn = gsl_sf_elljac_e(psi, m)
 

    SNorm2 = pPrec.Spl2 + (pPrec.Smi2 - pPrec.Spl2)*sn*sn
 
    return jnp.sqrt(SNorm2)

def IMRPhenomX_psiofv(v: float, v2: float, psi0: float, psi1: float, psi2: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a03b4dac8d2e93b6ab44bc1539981f5ce
    """
    return ( psi0 - 0.75*pPrec.g0 * pPrec.delta_qq * (1.0 + psi1*v + psi2*v2) / (v2*v) )

def IMRPhenomX_Return_MSA_Corrections_MSA(v: float, LNorm: float, JNorm: float, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a1aa2393a086fd52c37808975ed9086dc
    """

    return None


def IMRPhenomX_costhetaLJ(L_norm: float, J_norm: float, S_norm: float):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#aa7c77bf203a2f25482647de7ab325548
    """

    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / (L_norm * J_norm)
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)
    
    return costhetaLJ



def IMRPhenomX_Return_zeta_MSA(v: Array, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#a2f8434d98da248e9b6d4095bf13c0aa2
    """

    invv = 1.0/v
    invv2 = invv**2
    invv3 = invv2 * invv
    v2 = v*v
    logv = jnp.log(v) ### FIXME log or ln. Check base in lalsuite

    zeta_out = pPrec.eta * (
        pPrec.Omegazeta0_coeff*invv3 
        + pPrec.Omegazeta1_coeff*invv2 
        + pPrec.Omegazeta2_coeff*invv 
        + pPrec.Omegazeta3_coeff*logv 
        + pPrec.Omegazeta4_coeff*v 
        + pPrec.Omegazeta5_coeff*v2) 
    + pPrec.zeta_0


    zeta_out = jnp.where(jnp.isnan(zeta_out), 0.0, zeta_out) #### If zeta_out is nan, the function will return 0. 


    return zeta_out



def IMRPhenomX_Return_phiz_MSA(v: Array, JNorm, pPrec):
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#acc03f055e60854037e30ade700d56b5d
    """

    invv    = 1.0 / v
    invv2   = invv * invv

    LNewt   = pPrec.eta / v

    c1      = pPrec.c1
    c12     = c1 * c1

    SAv2    = pPrec.SAv2
    SAv     = pPrec.SAv
    invSAv  = pPrec.invSAv
    invSAv2 = pPrec.invSAv2


    log1 = jnp.log(jnp.abs(c1 + JNorm * pPrec.eta + pPrec.eta * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))


    phiz_0_coeff = (JNorm * pPrec.inveta4) * (0.5*c12 - c1*pPrec.eta2*invv/6.0 - SAv2*pPrec.eta2/3.0 - pPrec.eta4*invv2/3.0) - ((c1 * 0.5 * pPrec.inveta) * (c12 * pPrec.inveta4 - SAv2 * pPrec.inveta2) * log1)
    phiz_1_coeff = - 0.5 * JNorm * pPrec.inveta2 * (c1 + pPrec.eta * LNewt) + 0.5*pPrec.inveta3 * (c12 - pPrec.eta2*SAv2) * log1
    phiz_2_coeff = -JNorm + SAv*log2 - c1*log1*pPrec.inveta
    phiz_3_coeff = JNorm*v - pPrec.eta*log1 + c1*log2*pPrec.invSAv
    phiz_4_coeff = (0.5*JNorm*invSAv2*v)*(c1 + v*SAv2) - (0.5*invSAv2*invSAv)*(c12 - pPrec.eta2*SAv2)*log2
    phiz_5_coeff = -JNorm*v*( (0.5*c12*invSAv2*invSAv2) - (c1*v*invSAv2/6.0) - v*v/3.0 - pPrec.eta2*invSAv2/3.0) + (0.5*c1*invSAv2*invSAv2*invSAv)*(c12 - pPrec.eta2*SAv2)*log2


    phiz_out   = (phiz_0_coeff * pPrec.Omegaz0_coeff 
                  + phiz_1_coeff * pPrec.Omegaz1_coeff 
                  + phiz_2_coeff * pPrec.Omegaz2_coeff 
                  + phiz_3_coeff * pPrec.Omegaz3_coeff 
                  + phiz_4_coeff * pPrec.Omegaz4_coeff 
                  + phiz_5_coeff * pPrec.Omegaz5_coeff 
                  + pPrec.phiz_0)
    

    phiz_out = jnp.where(jnp.isnan(phiz_out), 0.0, phiz_out) #### If zeta_out is nan, the function will return 0. 

    return (phiz_out)
    
    


class IMRPhenomXPrecessionStruct:
    def __init__(self, eta, Omegazeta0_coeff, Omegazeta1_coeff, Omegazeta2_coeff,
                 Omegazeta3_coeff, Omegazeta4_coeff, Omegazeta5_coeff, zeta_0):
        self.eta = eta
        self.Omegazeta0_coeff = Omegazeta0_coeff
        self.Omegazeta1_coeff = Omegazeta1_coeff
        self.Omegazeta2_coeff = Omegazeta2_coeff
        self.Omegazeta3_coeff = Omegazeta3_coeff
        self.Omegazeta4_coeff = Omegazeta4_coeff
        self.Omegazeta5_coeff = Omegazeta5_coeff
        self.zeta_0 = zeta_0

    

