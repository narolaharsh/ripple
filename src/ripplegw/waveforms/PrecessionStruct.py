import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C


class IMRPhenomXGetAndSetPrecessionVariables:

    def __init__(self, pWF, m1_SI, m2_SI, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, lalParams, debug_flag):
        """
        lalsuite: https:#lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#af089ef2586c52b12016c0d791b176121

        Functions to Perform Frame Transformations and Populate Structs
        - Calculates frame transformation
        - PN Euler angles
        - Frame transformations
        - Orbital angular momenta etc

        
        pWF: Some useful waveform parameters

        m1_SI: Mass of object 1 (heavier one) in SI units
        m2_SI: Mass of object 2 (lighter one) in SI units


        chi1x: Object 1 spin. x-component.
        chi1y: Object 1 spin. y-component.
        chi1z: Object 1 spin. z-component.

        chi2x: Object 2 spin. x-component.
        chi2y: Object 2 spin. y-component.
        chi2z: Object 2 spin. z-component.

        lalParams: FIXME
        debug_flag: FIXME
        """

        # Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
        self.pWF = pWF

        self.IMRPhenomXPrecVersion = lalParams['IMRPhenomXPrecVersion']

        ## default to NNLO angles if in-plane spins are negligible and one of the SpinTaylor options has been selected. The solutions would be dominated by numerical noise.
        self.IMRPhenomXPrecVersion = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        chi_in_plane = jnp.sqrt(chi1x*chi1x+chi1y*chi1y+chi2x*chi2x+chi2y*chi2y)
        self.IMRPhenomXPrecVersion = jnp.where((chi_in_plane<1e-6) & (self.IMRPhenomXPrecVersion == 330), 102, self.IMRPhenomXPrecVersion)
        
        cond1 = chi_in_plane<1e-7
        cond2 = (self.IMRPhenomXPrecVersion==320) | (self.IMRPhenomXPrecVersion==321) | (self.IMRPhenomXPrecVersion==310) | (self.IMRPhenomXPrecVersion==311) 
        self.IMRPhenomXPrecVersion = jnp.where((cond1) & (cond2), 102, self.IMRPhenomXPrecVersion)


        self.ExpansionOrder = lalParams['ExpansionOrder']
        self.PNRUseTunedAngles = lalParams['PNRUseTunedAngles']
        self.IMRPhenomXPNRInterpTolerance = lalParams['IMRPhenomXPNRInterpTolerance']
        self.AntisymmetricWaveform = lalParams['AntisymmetricWaveform']
        self.PolarizationSymmetry = 1.0



        ########## Define a number of convenient local parameters #############
        m1        = m1_SI / pWF['Mtot_SI']   #/* Normalized mass of larger companion:   m1_SI / Mtot_SI */
        m2        = m2_SI / pWF['Mtot_SI']   #/* Normalized mass of smaller companion:  m2_SI / Mtot_SI */
        M         = (m1 + m2)              #/* Total mass in solar units */
        
        # Useful powers of mass
        m1_2      = m1 * m1
        m1_3      = m1 * m1_2
        m1_4      = m1 * m1_3
        m1_5      = m1 * m1_4
        m1_6      = m1 * m1_5
        m1_7      = m1 * m1_6
        m1_8      = m1 * m1_7
        
        m2_2      = m2 * m2
        
        pWF['M'] = M
        pWF['m1_2'] = m1_2
        pWF['m2_2'] = m2_2

        q = m1/m2


        # Powers of eta
        eta       = pWF['eta']
        eta2      = eta*eta
        eta3      = eta*eta2
        eta4      = eta*eta3
        eta5      = eta*eta4
        eta6      = eta*eta5

        # \delta in terms of q > 1
        delta     = pWF['delta']
        delta2    = delta*delta
        delta3    = delta*delta2

        # Cache these powers, as we use them regularly
        self.eta            = eta
        self.eta2           = eta2
        self.eta3           = eta3
        self.eta4           = eta4

        self.inveta         = 1.0 / eta
        self.inveta2        = 1.0 / eta2
        self.inveta3        = 1.0 / eta3
        self.inveta4        = 1.0 / eta4
        self.sqrt_inveta    = 1.0 / jnp.sqrt(eta)

        chi_eff   = pWF['chiEff']

        self.twopiGM        = 2*jnp.pi * G * (m1_SI + m2_SI) / C / C / C
        self.piGM           = jnp.pi * (m1_SI + m2_SI) * (G / C) / (C * C)

        ####  Set spin variables in pPrec struct  ######
        self.chi1x          = chi1x
        self.chi1y          = chi1y
        self.chi1z          = chi1z
        self.chi1_norm      = jnp.sqrt(chi1x*chi1x + chi1y*chi1y + chi1z*chi1z)

        self.chi2x          = chi2x
        self.chi2y          = chi2y
        self.chi2z          = chi2z
        self.chi2_norm      = jnp.sqrt(chi2x*chi2x + chi2y*chi2y + chi2z*chi2z)

        ### /* Check that spins obey Kerr bound */ ####
        """
        FIXME
        ### I will come back to the Kerr bound later line 210 ############

        condition = (jnp.logical_not(self.PNRUseTunedAngles)) | (pWF['PNR_SINGLE_SPIN'] != 1)

        if condition:
            kerr_boud_1 = jnp.abs(self.chi1_norm) <= 1.0
            kerr_boud_2 = jnp.abs(self.chi2_norm) <= 1.0
            if kerr_boud_1 & kerr_boud_2:
                Continue running
            else:
                Quit
                print("Error in IMRPhenomXSetPrecessionVariables: |S1/m1^2| must be <= 1.\n")
        else:
            continue running
        """

        ###/* Calculate dimensionful spins */
        self.S1x        = chi1x * m1_2
        self.S1y        = chi1y * m1_2
        self.S1z        = chi1z * m1_2
        self.S1_norm    = jnp.abs(self.chi1_norm) * m1_2

        self.S2x        = chi2x * m2_2
        self.S2y        = chi2y * m2_2
        self.S2z        = chi2z * m2_2
        self.S2_norm    = jnp.abs(self.chi2_norm) * m2_2

        ###// Useful powers
        self.S1_norm_2  = self.S1_norm * self.S1_norm
        self.S2_norm_2  = self.S2_norm * self.S2_norm

        self.chi1_perp  = jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
        self.chi2_perp  = jnp.sqrt(chi2x*chi2x + chi2y*chi2y)

        ###/* Get spin projections */
        self.S1_perp    = (m1_2) * jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
        self.S2_perp    = (m2_2) * jnp.sqrt(chi2x*chi2x + chi2y*chi2y)

        ###/* Norm of in-plane vector sum: Norm[ S1perp + S2perp ] */
        self.STot_perp     = jnp.sqrt( (self.S1x+self.S2x)*(self.S1x+self.S2x) + (self.S1y+self.S2y)*(self.S1y+self.S2y) )

        ##/* This is called chiTot_perp to distinguish from Sperp used in contrusction of chi_p. For normalization, see Sec. IV D of arXiv:2004.06503 */
        self.chiTot_perp   = self.STot_perp * (M*M) / m1_2
        ##/* Store self.chiTot_perp to pWF so that it can be used in XCP modifications (PNRUseTunedCoprec) */
        pWF['chiTot_perp'] = self.chiTot_perp


        self.PNRUseTunedAngles, lalParams['PNRUseTunedAngles'], self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']  = jnp.where((chi_in_plane<1e-7) & (self.PNRUseTunedAngles == 1), jnp.array([False, False, False, False, False]), jnp.array([self.PNRUseTunedAngles, lalParams['PNRUseTunedAngles'], self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']]))
            

        ### Implementation up to line 257



        """
        self.eta = eta
        self.Omegazeta0_coeff = Omegazeta0_coeff
        self.Omegazeta1_coeff = Omegazeta1_coeff
        self.Omegazeta2_coeff = Omegazeta2_coeff
        self.Omegazeta3_coeff = Omegazeta3_coeff
        self.Omegazeta4_coeff = Omegazeta4_coeff
        self.Omegazeta5_coeff = Omegazeta5_coeff
        self.zeta_0 = zeta_0
        """
    




def assert_int(condition: bool, success: int = 0, failure: int = -1) -> int:
    """
    Returns `success` if condition is True, otherwise `failure`.

    Equivalent to a C macro that checks an assertion and returns -1 on failure.
    """
    return jnp.where(condition, success, failure)