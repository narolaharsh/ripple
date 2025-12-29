import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C, MTSUN_SI, GAMMA
import jax
from .spherical_harmonics import *
from .IMRPhenomXPHM_utils import *

class IMRPhenomXGetAndSetPrecessionVariables:

    def __init__(self, pWF: dict, m1_SI: float, m2_SI: float, chi1x: float, chi1y: float, chi1z: float, chi2x: float, chi2y: float, chi2z: float, lalParams: dict, debug_flag: bool):
        """
        lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#af089ef2586c52b12016c0d791b176121

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

        Note to self:
        1: NNLO
        2: MSA
        3: Spin-Taylor

        precversionTag  or precessing_tag means the first digit
        pflag means the full number
        """

        # Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
        self.pWF = pWF
        #self.pWF['LALparams'] = lalParams
        self.lalParams = lalParams
        self.MAX_TOL_ATAN = 1.0e-15
        self.check_convention_array = jnp.array([0, 5])
        #Pre-cache useful powers here
        self.sqrt2   = 1.4142135623730951
        self.sqrt5   = 2.23606797749978981
        self.sqrt6   = 2.44948974278317788
        self.sqrt7   = 2.64575131106459072
        self.sqrt10  = 3.16227766016838
        self.sqrt14  = 3.74165738677394133
        self.sqrt15  = 3.87298334620741702
        self.sqrt70  = 8.36660026534075563
        self.sqrt30  = 5.477225575051661
        self.sqrt2p5 = 1.58113883008419
        self.log16    = 2.772588722239781
        self.power_of_lalpi_2 = jnp.power(jnp.pi, 2)
        self.debug_prec = debug_flag

        # Sort out version-specific flags
        # Get IMRPhenomX precession version from LAL dictionary
        self.IMRPhenomXPrecVersion = self.lalParams['IMRPhenomXPrecVersion']
        self.IMRPhenomXPrecVersion = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        
        self.manual_prescription_tag = self.check_prescription_tag(self.IMRPhenomXPrecVersion)# 1 for NNLO, 2 for MSA, 3 for SpinTaylor
        # Line 109
        # default to NNLO angles if in-plane spins are negligible and one of the SpinTaylor options has been selected. The solutions would be dominated by numerical noise.
        chi_in_plane = jnp.sqrt(chi1x*chi1x+chi1y*chi1y+chi2x*chi2x+chi2y*chi2y)

        self.IMRPhenomXPrecVersion = jnp.where((chi_in_plane<1e-6) & (self.IMRPhenomXPrecVersion == 330), 102, self.IMRPhenomXPrecVersion)
        
        #Compress line 114
        self.IMRPhenomXPrecVersion = jnp.where((chi_in_plane<1e-7) & (self.manual_prescription_tag==3), 102, self.IMRPhenomXPrecVersion)


        self.ExpansionOrder = lalParams['ExpansionOrder']
        self.PNRUseTunedAngles = lalParams['PNRUseTunedAngles']
        self.IMRPhenomXPNRInterpTolerance = lalParams['IMRPhenomXPNRInterpTolerance']
        self.AntisymmetricWaveform = lalParams['AntisymmetricWaveform']
        self.PolarizationSymmetry = 1.0

        #### Skipping the multibanding bookkeeping around line 136-142

        self.m1_SI = m1_SI
        self.m2_SI = m2_SI
        # Define a number of convenient local parameters #############
        self.m1        = self.m1_SI / self.pWF['Mtot_SI']   #/* Normalized mass of larger companion:   m1_SI / Mtot_SI */
        self.m2        = self.m2_SI / self.pWF['Mtot_SI']   #/* Normalized mass of smaller companion:  m2_SI / Mtot_SI */
        self.M         = (self.m1 + self.m2)              #/* Total mass in solar units */
        
        # Useful powers of mass
        self.m1_2      = self.m1 * self.m1
        self.m1_3      = self.m1 * self.m1_2
        self.m1_4      = self.m1 * self.m1_3
        self.m1_5      = self.m1 * self.m1_4
        self.m1_6      = self.m1 * self.m1_5
        self.m1_7      = self.m1 * self.m1_6
        self.m1_8      = self.m1 * self.m1_7
        
        self.m2_2      = self.m2 * self.m2
        
        self.pWF['M'] = self.M
        self.pWF['m1_2'] = self.m1_2
        self.pWF['m2_2'] = self.m2_2

        self.q = self.m1/self.m2


        # Powers of eta
        self.eta       = self.pWF['eta']
        self.eta2      = self.eta*self.eta
        self.eta3      = self.eta*self.eta2
        self.eta4      = self.eta*self.eta3
        self.eta5      = self.eta*self.eta4
        self.eta6      = self.eta*self.eta5

        # \delta in terms of q > 1
        self.delta     = self.pWF['delta']
        self.delta2    = self.delta*self.delta
        self.delta3    = self.delta*self.delta2

        # Cache these powers, as we use them regularly

        self.inveta         = 1.0 / self.eta
        self.inveta2        = 1.0 / self.eta2
        self.inveta3        = 1.0 / self.eta3
        self.inveta4        = 1.0 / self.eta4
        self.sqrt_inveta    = 1.0 / jnp.sqrt(self.eta)

        self.chi_eff   = self.pWF['chiEff']

        self.twopiGM        = 2*jnp.pi * G * (self.m1_SI + self.m2_SI) / C / C / C
        self.piGM           = jnp.pi * (self.m1_SI + self.m2_SI) * (G / C) / (C * C)

        ####  Set spin variables in pPrec struct  ######
        self.chi1x          = chi1x
        self.chi1y          = chi1y
        self.chi1z          = chi1z
        self.chi1_norm      = jnp.sqrt(chi1x*chi1x + chi1y*chi1y + chi1z*chi1z)

        self.chi2x          = chi2x
        self.chi2y          = chi2y
        self.chi2z          = chi2z
        self.chi2_norm      = jnp.sqrt(chi2x*chi2x + chi2y*chi2y + chi2z*chi2z)

        ###Check that spins obey Kerr bound */ ####
        """
        FIXME
        ### I will come back to the Kerr bound later line 210 ############
        I do not know a jax friendly version for assert

        condition = (jnp.logical_not(self.PNRUseTunedAngles)) | (pWF['PNR_SINGLE_SPIN'] != 1)

        if condition:
            assert jnp.abs(self.chi1_norm) <= 1.0
            assert jnp.abs(self.chi2_norm) <= 1.0
        """

        ###/* Calculate dimensionful spins */
        self.S1x        = self.chi1x * self.m1_2
        self.S1y        = self.chi1y * self.m1_2
        self.S1z        = self.chi1z * self.m1_2
        self.S1_norm    = jnp.abs(self.chi1_norm) * self.m1_2

        self.S2x        = chi2x * self.m2_2
        self.S2y        = chi2y * self.m2_2
        self.S2z        = chi2z * self.m2_2
        self.S2_norm    = jnp.abs(self.chi2_norm) * self.m2_2

        ###// Useful powers
        self.S1_norm_2  = self.S1_norm * self.S1_norm
        self.S2_norm_2  = self.S2_norm * self.S2_norm

        self.chi1_perp  = jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
        self.chi2_perp  = jnp.sqrt(chi2x*chi2x + chi2y*chi2y)

        ###/* Get spin projections */
        self.S1_perp    = (self.m1_2) * jnp.sqrt(self.chi1x*self.chi1x + self.chi1y*self.chi1y)
        self.S2_perp    = (self.m2_2) * jnp.sqrt(self.chi2x*self.chi2x + self.chi2y*self.chi2y)

        ###/* Norm of in-plane vector sum: Norm[ S1perp + S2perp ] */
        self.STot_perp     = jnp.sqrt( (self.S1x+self.S2x)*(self.S1x+self.S2x) + (self.S1y+self.S2y)*(self.S1y+self.S2y) )

        ##/* This is called chiTot_perp to distinguish from Sperp used in contrusction of chi_p. For normalization, see Sec. IV D of arXiv:2004.06503 */
        self.chiTot_perp   = self.STot_perp * (self.M*self.M) / self.m1_2
        ##/* Store self.chiTot_perp to pWF so that it can be used in XCP modifications (PNRUseTunedCoprec) */
        self.pWF['chiTot_perp'] = self.chiTot_perp

        # Line 245-255
        #disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit */
        cond = (chi_in_plane<1e-7) & (self.PNRUseTunedAngles == 1) & (self.pWF['PNR_SINGLE_SPIN']!=1)

        self.PNRUseTunedAngles = jnp.where(cond, False, self.PNRUseTunedAngles)
        self.lalParams['PNRUseTunedAngles'] = jnp.where(cond, False, self.lalParams['PNRUseTunedAngles'])

        self.AntisymmetricWaveform = jnp.where(cond, False, self.AntisymmetricWaveform)
        self.lalParams['AntisymmetricWaveform'] = jnp.where(cond, False, self.lalParams['AntisymmetricWaveform'])

        self.lalParams['PNRUseTunedCoprec'] = jnp.where(cond, False, self.lalParams['PNRUseTunedCoprec'])

        #Calculate the effective precessing spin parameter (Schmidt et al, PRD 91, 024043, 2015): m1 > m2, so body 1 is the larger black hole
        self.A1             = 2.0 + (3.0 * self.m2) / (2.0 * self.m1)
        self.A2             = 2.0 + (3.0 * self.m1) / (2.0 * self.m2)
        self.ASp1           = self.A1 * self.S1_perp
        self.ASp2           = self.A2 * self.S2_perp

        #/* S_p = max(A1 S1_perp, A2 S2_perp) */
        num       = jnp.where(self.ASp2 > self.ASp1, self.ASp2, self.ASp1)
        den       = jnp.where(self.m2 > self.m1 , self.A2*self.m2_2, self.A1*self.m1_2)

        #/* chi_p = max(A1 * Sp1 , A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH */
        self.chip      = num / den
        self.chi1L     = self.chi1z
        self.chi2L     = self.chi2z


        self.chi_p          = self.chip
        #// (PNRUseTunedCoprec)
        self.pWF['chi_p']        = self.chi_p
        self.phi0_aligned   = self.pWF['phi0']

        #/* Effective (dimensionful) aligned spin */
        self.SL             = self.chi1L*self.m1_2 + self.chi2L*self.m2_2

        #/* Effective (dimensionful) in-plane spin */
        self.Sperp          = self.chi_p  * self.m1_2                 # /* m1 > m2 */

        #self.MSA_ERROR      = 0

        self.pWF22AS = None
        #start of SpinTaylor code


        #get first digit of precessing version: this tags the method employed to compute the Euler angles
        #1: NNLO 2: MSA 3: SpinTaylor (numerical)


        ## SpinTaylor code is from 294 to 484
        #start of SpinTaylor code
        '''
        1. Initialize PNarrays
        2. self.L_MAX_PNR = self.M_MAX

        ModeArray 

        LMAX_PNR = 2

        if mode_array is not none:
            if 44 is active:
                LMAX_PNR = 4
            elif 33 or 32 active:
                LMAX_PNR = 3
            GetandSetModes..
            self.LMAX_PNR = LMAX_PNR

        flow = pWF['fMin']
        if deltaF is zero: get it from wfstruct

        if PNRUseTunedAngles is flase:
            flow = update
        else:
            flow = update

        self.fmin_HM_inspiral = flow * 2.0 / pPrec->M_MAX;

        make backup of the original precVersion

        create a fake prec version 223

        IMRPhenomX_PNR_GetAndSetPNRVariables(pWF, pPrec);

        IMRPhenomX_PNR_precompute_alpha_coefficients(alphaParams, pWF, pPrec);

        IMRPhenomX_PNR_precompute_beta_coefficients(betaParams, pWF, pPrec);

        IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams);

        revert to original precVersion

        define some new floats based of if

        XLALSimIMRPhenomXUtilsMftoHz(Mf_low_cut * 0.65 * pPrec->M_MAX / 2.0, pWF->Mtot);

        if...else to adjust flow again


        IMRPhenomX_PNR_HMInterpolationDeltaF(flow, pWF, pPrec)

        IMRPhenomX_InspiralAngles_SpinTaylor(function with lots of arguments)

        Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(pPrec->fmin_integration,pWF->Mtot);

        if 330:
            do some rotations
        
        if failure:
            fall back on 223
        '''

        self.IMRPhenomXPrecVersion = jax.lax.cond(self.IMRPhenomXPrecVersion==self.manual_prescription_tag, self.spin_taylor_code, lambda _: 223, operand = None)

        # Compress line 486-490
        pflag = jnp.int32(self.IMRPhenomXPrecVersion)
        self.manual_prescription_tag = self.check_prescription_tag(self.IMRPhenomXPrecVersion) 

        # Skipping 491-494. Sanity check if pflag is in the allowd values [101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321, 330]
        # Skipping line  496-544. Looks like sanity check. FIXME: Source of error. 
        
        #/* Calculate parameter for two-spin to single-spin map used in PNR and XCP */
        #/* Initialize PNR variables */
        self.chi_singleSpin = 0.0
        self.costheta_singleSpin = 0.0
        self.costheta_final_singleSpin = 0.0
        self.chi_singleSpin_antisymmetric = 0.0
        self.theta_antisymmetric = 0.0
        self.PNR_HM_Mflow = 0.0
        self.PNR_HM_Mfhigh = 0.0

        self.PNR_q_window_lower = 0.0
        self.PNR_q_window_upper = 0.0
        self.PNR_chi_window_lower = 0.0
        self.PNR_chi_window_upper = 0.0
        #self.PNRInspiralScaling = 0

        PNR_variables = IMRPhenomX_PNR_GetAndSetPNRVariables(self.pWF, self.lalParams)

        self.alphaPNR = 0.0
        self.betaPNR = 0.0
        self.gammaPNR = 0.0

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Get and/or store CoPrec params into pWF and pPrec     /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        PNR_co_prec_params = IMRPhenomX_PNR_GetAndSetCoPrecParams(self.pWF, self.lalParams)

        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Logic used in compressing 585 to 609
        # If flag is in [220, 221, 222, 223, 224], try MSA. 
            # If it fails, 
            #   check if the flag is in [220, 223, 224]
                # If yes, updated flag to 102
                # If no, terminal failure
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Compress line 585 to 609 #FIXME: NH possible source of error
        MSA_ERROR = 0
        MSA_ERROR = self.compute_MSA_and_check_MSA_error(pflag)
        pflag = self.switch_to_NNLO(pflag, MSA_ERROR)
        
        # Skipping DEBUG line 610-616

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Compute and set final spin and RD frequency           /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
        
        precessing_remnant_params = IMRPhenomX_SetPrecessingRemnantParams(self.pWF, self.lalParams)
        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/


        # /* Useful powers of \chi_p */
        self.chip2    = self.chip * self.chip

        #/* Useful powers of spins aligned with L */
        self.chi1L2   = self.chi1L * self.chi1L
        self.chi2L2   = self.chi2L * self.chi2L

        #Cache the orbital angular momentum coefficients for future use.

        #References:
        #- Kidder, PRD, 52, 821-847, (1995), arXiv:gr-qc/9506022
        #- Blanchet, LRR, 17, 2, (2014), arXiv:1310.1528
        #- Bohe et al, 1212.5520v2
        #- Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118

        # Compress line 641 - 755
        self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L8L = self.twoPN_non_spinning_orbitan_angular_momentum(pflag)

        #Reference orbital angular momentum
        self.LRef = self.M * self.M * XLALSimIMRPhenomXLPNAnsatz(self.pWF['v_ref'], self.pWF['eta'] / self.pWF['v_ref'], self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L8L) 

        '''
            In the following code block we construct the convetions that relate the source frame and the LAL frame.

            A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602
        '''

        #Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
        self.J0x_Sf = (self.m1_2)*self.chi1x + (self.m2_2)*self.chi2x
        self.J0y_Sf = (self.m1_2)*self.chi1y + (self.m2_2)*self.chi2y
        self.J0z_Sf = (self.m1_2)*self.chi1z + (self.m2_2)*self.chi2z + self.LRef

        self.J0_Sf = jnp.array([self.J0x_Sf, self.J0y_Sf, self.J0z_Sf])


        self.J0     = jnp.sqrt(self.J0x_Sf*self.J0x_Sf + self.J0y_Sf*self.J0y_Sf + self.J0z_Sf*self.J0z_Sf)

        # Compress line 772 - 781
        #/* Get angle between J0 and LN (z-direction) */
        self.thetaJ_Sf = jax.lax.cond(self.J0<1e-10, lambda _: 0.0, lambda _:jnp.acos(self.J0z_Sf / self.J0), operand = None)

        # Line 783
        self.phiRef = self.pWF['phiRef_In']
        # Line 785
        phenom_xp_convention     = self.lalParams['PhenomXPConvention']

        tol_condition = (jnp.abs(self.J0x_Sf) < self.MAX_TOL_ATAN) & (jnp.abs(self.J0y_Sf) < self.MAX_TOL_ATAN)
        # Compress line 797-825
        #Get azimuthal angle of J0 in the source frame
        self.phiJ_Sf = self.get_phiJ_Sf(tol_condition, self.phiRef, phenom_xp_convention, self.J0_Sf)

        self.phi0_aligned = - self.phiJ_Sf

        #Compress line 828 - 846 #FIXME in function set_phi0 I am not sure what to do for cases 5, 6, 7. What is the old value?
        self.phi0 = self.set_phi0(phenom_xp_convention, self.phi0_aligned)

        '''
        Here we follow the same prescription as in IMRPhenomPv2:

        Now rotate from SF to J frame to compute alpha0, the azimuthal angle of LN, as well as
        thetaJ, the angle between J and N.

        The J frame is defined by imposing that J points in the z-direction and the line of sight N is in the xz-plane
        (with positive projection along x).

        The components of any vector in the (new) J-frame can be obtained by rotation from the (old) source frame (SF).
        This is done by multiplying by: RZ[-kappa].RY[-thetaJ].RZ[-phiJ]

        Note that kappa is determined by rotating N with RY[-thetaJ].RZ[-phiJ], which brings J to the z-axis, and
        taking the opposite of the azimuthal angle of the rotated N.
        '''

        #Determine kappa via rotations, as above */
        self.Nx_Sf = jnp.sin(self.pWF['inclination'])*jnp.cos((jnp.pi / 2.0) - self.phiRef)
        self.Ny_Sf = jnp.sin(self.pWF['inclination'])*jnp.sin((jnp.pi / 2.0) - self.phiRef)
        self.Nz_Sf = jnp.cos(self.pWF['inclination'])
        self.N_Sf = jnp.array([self.Nx_Sf, self.Ny_Sf, self.Nz_Sf])

        v_in = jnp.array([self.Nx_Sf, self.Ny_Sf, self.Nz_Sf])

        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf, v_in)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)

        #/* Note difference in overall - sign w.r.t PhenomPv2 code */
        self.kappa = XLALSimIMRPhenomXatan2tol(vout[1],vout[0], self.MAX_TOL_ATAN)

        #/* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
        tmp_x = 0.0
        tmp_y = 0.0
        tmp_z = 1.0
        v_in = jnp.array([tmp_x, tmp_y, tmp_z])
        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf,   v_in)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-self.kappa,     vout)

        # Compress line 887 - 930
        tol_condition = (jnp.abs(vout[0]) < self.MAX_TOL_ATAN) & (jnp.abs(vout[1]) < self.MAX_TOL_ATAN)
        self.alpha0 = self.set_alpha0(tol_condition, phenom_xp_convention, vout[0], vout[1], self.kappa)
        

        # Compress line 931-966
        self.thetaJN, self.Nz_Jf, self.Nx_Jf = jax.lax.cond(jnp.isin(phenom_xp_convention, jnp.array([0, 5])), self.thetaJN_Nz_Nx_0_5, self.thetaJN_Nz_Nx_1_6_7, v_in, self.N_Sf, self.J0_Sf, self.phiJ_Sf, self.thetaJ_Sf, self.kappa)


        '''
        Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

        The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
        i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

        However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
        NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

        The triads differ from each other by a rotation around N by an angle zeta. We therefore need to rotate 
        the polarizations by an angle 2 zeta.
        '''
        #Compressed line 983  to 991
        self.Xx_Sf = -jnp.cos(pWF['inclination']) * jnp.sin(self.phiRef)
        self.Xy_Sf = -jnp.cos(pWF['inclination']) * jnp.cos(self.phiRef)
        self.Xz_Sf = +jnp.sin(pWF['inclination'])


        v = jnp.array([self.Xx_Sf, self.Xy_Sf, self.Xz_Sf])
        vout = IMRPhenomX_rotate_z(-self.phiJ_Sf, v)
        vout = IMRPhenomX_rotate_y(-self.thetaJ_Sf, vout)
        vout = IMRPhenomX_rotate_z(-self.kappa, vout)


        '''

            The components tmp_i are now the components of X in the J frame.

            We now need the polar angle of this vector in the P, Q basis of Arun et al:

                P = (N x J) / |NxJ|

            Note, that we put N in the (pos x)z half plane of the J frame 

        '''
        #Compress line 1002-1034
        self.PArun_Jf, self.QArun_Jf = jax.lax.cond(jnp.isin(phenom_xp_convention, jnp.array([0, 5])), self.PQ_Arun_0_5, self.PQ_Arun_1_6_7, self.Nx_Jf, self.Nz_Jf)

        #As it is line 1035-1043
        #(X . P)
        self.XdotPArun = (vout[0] * self.PArun_Jf[0]) + (vout[1] * self.PArun_Jf[1]) + (vout[2] * self.PArun_Jf[2])

        #(X . Q)
        self.XdotQArun = (vout[0] * self.QArun_Jf[0]) + (vout[1] * self.QArun_Jf[1]) + (vout[2] * self.QArun_Jf[2])

        #Now get the angle zeta
        self.zeta_polarization = jnp.atan2(self.XdotQArun, self.XdotPArun)

        #/* ********** PN Euler Angle Coefficients ********** */
        #/*
        #    This uses the single spin PN Euler angles as per IMRPhenomPv2
        #*/  

        #/* ********** PN Euler Angle Coefficients ********** */
        # Compress line 1050-1143
        cond = jnp.isin(pflag, jnp.array([101, 102, 103, 104]))
        self.alpha1, self.alpha2, self.alpha3, self.alpha4L, self.alpha5, self.epsilon1, self.epsilon2, self.epsilon3, self.epsilon4L, self.epsilon5 = jax.lax.cond(cond, self.compute_alpha_epsilon_101_104, self.compute_alpha_epsilon_220_330, operand = None)

        #Skipping the #if DEBUG==1 lines 1144-1162
        
        # Compressed line 1163-1177
        self.epsilon0 = self.set_epsilon0(phenom_xp_convention, self.phiJ_Sf)

        ## Compression line 1178-1202
        cond = (phenom_xp_convention == 5) | (phenom_xp_convention==7)
        self.alpha_offset, self.epsilon_offset, self.alpha_offset_1, self.epsilon_offset_1, self.alpha_offset_3, self.epsilon_offset_3, self.alpha_offset_4, self.epsilon_offset_4 =  jax.lax.cond(cond, self.convention_five_or_seven_true, self.convention_five_or_seven_false, operand = self.alpha0)

        self.cexp_i_alpha   = 0.
        self.cexp_i_epsilon = 0.
        self.cexp_i_betah   = 0.

        # When L + SL < 0 and q>7, we disable multibanding NH: I will skip this function
        #self.IMRPhenomXPCheckMaxOpeningAngle()

        # Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
        self.MBandPrecVersion = jax.lax.cond(self.lalParams['PhenomXPHMThresholdMband']==0, lambda _: 0, lambda _: 1, operand = None)
        ## NH: I do not implement PhenomXPHMThresholdMband==1 option. The output of the above line will always be self.MBandPrecVersion = 0. 


        # At high mass ratios, we find there can be numerical instabilities in the model, although the waveforms continue to be well behaved.
        # We warn to user of the possibility of these instabilities.
        # printf(pWF->q)
        jax.lax.cond(pWF["q"] > 80, 
                     lambda _: jax.debug.print("Very high mass ratio, possibility of numerical instabilities. Waveforms remain well behaved."), 
                     lambda _: None, 
                     operand = None)


        self.Y2m2         = compute_sminus2_l2(theta = self.thetaJN, m = -2)
        self.Y2m1         = compute_sminus2_l2(theta = self.thetaJN, m = -1)
        self.Y20          = compute_sminus2_l2(theta = self.thetaJN, m = 0)
        self.Y21          = compute_sminus2_l2(theta = self.thetaJN, m = 1)
        self.Y22          = compute_sminus2_l2(theta = self.thetaJN, m = 2)

        self.Y3m3         = compute_sminus2_l3(theta = self.thetaJN, m = -3)
        self.Y3m2         = compute_sminus2_l3(theta = self.thetaJN, m = -2)
        self.Y3m1         = compute_sminus2_l3(theta = self.thetaJN, m = -1)
        self.Y30          = compute_sminus2_l3(theta = self.thetaJN, m = 0)
        self.Y31          = compute_sminus2_l3(theta = self.thetaJN, m = 1)
        self.Y32          = compute_sminus2_l3(theta = self.thetaJN, m = 2)
        self.Y33          = compute_sminus2_l3(theta = self.thetaJN, m = 3)

        self.Y4m4         = compute_sminus2_l4(theta = self.thetaJN, m = -4)
        self.Y4m3         = compute_sminus2_l4(theta = self.thetaJN, m = -3)
        self.Y4m2         = compute_sminus2_l4(theta = self.thetaJN, m = -2)
        self.Y4m1         = compute_sminus2_l4(theta = self.thetaJN, m = -1)
        self.Y40          = compute_sminus2_l4(theta = self.thetaJN, m = 0)
        self.Y41          = compute_sminus2_l4(theta = self.thetaJN, m = 1)
        self.Y42          = compute_sminus2_l4(theta = self.thetaJN, m = 2)
        self.Y43          = compute_sminus2_l4(theta = self.thetaJN, m = 3)
        self.Y44          = compute_sminus2_l4(theta = self.thetaJN, m = 4)

    def convention_five_or_seven_true(self, alpha0):
        return -alpha0, 0, -alpha0, 0, -alpha0, 0, -alpha0, 0
    
    def convention_five_or_seven_false(self, alpha0):
        # Get initial Get \alpha and \epsilon offsets at \omega = pi * M * f_{Ref} */
        alpha_offset, epsilon_offset = self.Get_alphaepsilon_atfref(2)
        return alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset, alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref(self, mprime):
        omega_ref = self.pWF['piM'] * self.pWF['fRef'] * 2 / mprime
        pflag = self.IMRPhenomXPrecVersion

        #/* Explicitly enumerate MSA flags */
        cond = (pflag == 220) | (pflag == 221) | (pflag == 222) | (pflag == 223) | (pflag == 224)

        alpha_offset, epsilon_offset = jax.lax.cond(cond, self.Get_alphaepsilon_atfref_pflag_true, self.Get_alphaepsilon_atfref_pflag_false, omega_ref)

        return alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref_pflag_true(self, omega_ref):

        v = jnp.cbrt(omega_ref)
        vangles  = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(self, v, self.pWF) # FIXME

        alpha_offset = vangles['x'] - self.alpha0
        epsilon_offset = vangles['x'] - self.epsilon0
        return alpha_offset, epsilon_offset
    
    def Get_alphaepsilon_atfref_pflag_false(self, omega_ref):
        logomega_ref    = jnp.log(omega_ref)
        omega_ref_cbrt  = jnp.cbrt(omega_ref)
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        alpha_offset = (self.alpha1  / omega_ref 
                        + self.alpha2  / omega_ref_cbrt2 
                        + self.alpha3  / omega_ref_cbrt 
                        + self.alpha4L * logomega_ref 
                        + self.alpha5  * omega_ref_cbrt - self.alpha0)

        epsilon_offset = (self.epsilon1  / omega_ref 
                          + self.epsilon2  / omega_ref_cbrt2 
                          + self.epsilon3  / omega_ref_cbrt 
                          + self.epsilon4L * logomega_ref 
                          + self.epsilon5  * omega_ref_cbrt - self.epsilon0)

        return alpha_offset, epsilon_offset
    
    def set_epsilon0(self, phenom_xp_convention, phiJ_Sf):

        epsilon0 = jax.lax.cond(
            jnp.isin(phenom_xp_convention, jnp.array([1, 6])),
            lambda _: phiJ_Sf - jnp.pi,
            lambda _: 0.0,
            operand=None,
        )
        
        return epsilon0
    

    def compute_alpha_epsilon_101_104(self, _):
        #This uses the single spin PN Euler angles as per IMRPhenomPv2
        #Post-Newtonian Euler Angles: alpha */
        chiL = (1.0 + self.q) * (self.chi_eff / self.q)
        chiL2 = chiL * chiL

        alpha1 = -35/192. + (5*self.delta)/(64.*self.m1)
        alpha2 = ((15*chiL*self.delta*self.m1)/128. - (35*chiL*self.m1_2)/128.) / self.eta
        alpha3 = (-5515/3072. + self.eta*(-515/384. - (15*self.delta2)/(256.*self.m1_2) 
                                          + (175*self.delta)/(256.*self.m1)) + (4555*self.delta)/(7168.*self.m1) 
                                          + ((15*self.chip2*self.delta*self.m1_3)/128. - (35*self.chip2*self.m1_4)/128.) / self.eta2)
        #This is the term proportional to log(w) */

        alpha4L = ((5*chiL*self.delta2)/16. - (5*chiL*self.delta*self.m1)/3. + (2545*chiL*self.m1_2)/1152.
                   + ((-2035*chiL*self.delta*self.m1)/21504. 
                      + (2995*chiL*self.m1_2)/9216.)/self.eta
                   + ((5*chiL*self.chip2*self.delta*self.m1_5)/128. 
                      - (35*chiL*self.chip2*self.m1_6)/384.) / self.eta3
                   - (35*jnp.pi)/48. + (5*self.delta*jnp.pi)/(16.*self.m1))
        
        alpha5 = (5*(-190512*self.delta3*self.eta6 + 2268*self.delta2*self.eta3*self.m1*(self.eta2*(323 + 784*self.eta) + 336*(25*chiL2 + self.chip2)*self.m1_4)
                + 7*self.m1_3*(8024297*self.eta4 + 857412*self.eta5 + 3080448*self.eta6
                               + 143640*self.chip2*self.eta2*self.m1_4 - 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                               + 6048*self.eta3*((2632*chiL2 + 115*self.chip2)*self.m1_4 - 672*chiL*self.m1_2*jnp.pi))
                + 3*self.delta*self.m1_2*(-5579177*self.eta4 + 80136*self.eta5 - 3845520*self.eta6
                                           + 146664*self.chip2*self.eta2*self.m1_4 + 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                                           - 42336*self.eta3*((726*chiL2 + 29*self.chip2)*self.m1_4 - 96*chiL*self.m1_2*jnp.pi)))) / (6.5028096e7*self.eta4*self.m1_3)


        epsilon1 = -35/192. + (5*self.delta)/(64.*self.m1)
        epsilon2 = ((15*chiL*self.delta*self.m1)/128. - (35*chiL*self.m1_2)/128.) / self.eta
        epsilon3 = -5515/3072. + self.eta*(-515/384. - (15*self.delta2)/(256.*self.m1_2) + (175*self.delta)/(256.*self.m1)) + (4555*self.delta)/(7168.*self.m1)
        epsilon4L = (5*chiL*self.delta2)/16. - (5*chiL*self.delta*self.m1)/3. + (2545*chiL*self.m1_2)/1152. \
                    + ((-2035*chiL*self.delta*self.m1)/21504. + (2995*chiL*self.m1_2)/9216.) / self.eta \
                    - (35*jnp.pi)/48. + (5*self.delta*jnp.pi)/(16.*self.m1)
        epsilon5 = (5*(-190512*self.delta3*self.eta3 + 2268*self.delta2*self.m1*(self.eta2*(323 + 784*self.eta) + 8400*chiL2*self.m1_4)
                        - 3*self.delta*self.m1_2*(self.eta*(5579177 + 504*self.eta*(-159 + 7630*self.eta)) + 254016*chiL*self.m1_2*(121*chiL*self.m1_2 - 16*jnp.pi))
                        + 7*self.m1_3*(self.eta*(8024297 + 36*self.eta*(23817 + 85568*self.eta)) + 338688*chiL*self.m1_2*(47*chiL*self.m1_2 - 12*jnp.pi)))) / (6.5028096e7*self.eta*self.m1_3)

        return alpha1, alpha2, alpha3, alpha4L, alpha5, epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5
    
    def compute_alpha_epsilon_220_330(self, _):
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  
    

    def PQ_Arun_0_5(self, Nx_Jf, Nz_Jf):
        #Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = 0.0
        PAruny_Jf = -1.0
        PArunz_Jf = 0.0

        #Q = (N x P) by construction
        QArunx_Jf = Nz_Jf
        QAruny_Jf = 0.0
        QArunz_Jf = -Nx_Jf

        return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array([QArunx_Jf, QAruny_Jf, QArunz_Jf])

    def PQ_Arun_1_6_7(self, Nx_Jf, Nz_Jf):
        # Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = Nz_Jf
        PAruny_Jf = 0.0
        PArunz_Jf = -Nx_Jf

        QArunx_Jf = 0.0
        QAruny_Jf = 1.0
        QArunz_Jf = 0.0

        return jnp.array([PArunx_Jf, PAruny_Jf, PArunz_Jf]), jnp.array([QArunx_Jf, QAruny_Jf, QArunz_Jf])
    

    def thetaJN_Nz_Nx_0_5(self, v_in, N_Sf, J0_Sf, phiJ_Sf, thetaJ_Sf, kappa):
        # Line 937-952 #
        #Now determine thetaJN by rotating N
        
        v = IMRPhenomX_rotate_z(phiJ_Sf,   v_in)
        v = IMRPhenomX_rotate_y(thetaJ_Sf, v)
        v = IMRPhenomX_rotate_z(kappa,     v)

        # We don't need the y-component but we will store it anyway

        Nz_Jf = v[2]
        # This is a unit vector, so no normalization
        thetaJN = jnp.acos(Nz_Jf)

        return thetaJN, v[2], v[0]

    def thetaJN_Nz_Nx_1_6_7(self, v_in, N_Sf, J0_Sf, phiJ_Sf, thetaJ_Sf, kappa):
        # Line 957-962

        J0dotN     = (J0_Sf[0] * N_Sf[0]) + (J0_Sf[1] * N_Sf[1]) + (J0_Sf[2] * N_Sf[2])
        thetaJN = jnp.acos( J0dotN / self.J0 )
        Nz_Jf     = jnp.cos(thetaJN)
        Nx_Jf     = jnp.sin(thetaJN)

        return thetaJN, Nz_Jf, Nx_Jf

    def set_phi0(self, phenom_xp_convention, phi0_aligned):
        phi0 = jnp.where(phenom_xp_convention == 0, phi0_aligned, 0.0)
        phi0 = jnp.where(phenom_xp_convention == 1, 0.0, phi0) 
        return phi0
    
    def set_alpha0(self, tol_condition, phenom_xp_convention, tmp_x, tmp_y, kappa):
        convention_condition = jnp.isin(phenom_xp_convention, self.check_convention_array)

        alpha0 = jax.lax.cond(tol_condition,
            lambda _: jax.lax.cond(convention_condition, lambda _2: jnp.pi, lambda _2: jnp.pi - kappa, operand = None),
            lambda _: jax.lax.cond(convention_condition, lambda _2: jnp.atan2(tmp_y, tmp_x), lambda _2: jnp.pi - kappa, operand = None),
            operand=None)
        
        return alpha0
    
    def get_phiJ_Sf(self, tol_condition, phiRef, phenom_xp_convention, J0_Sf):
        convention_condition = jnp.isin(phenom_xp_convention, self.check_convention_array)

        phiJ_Sf = jax.lax.cond(tol_condition, 
                     lambda _: jax.lax.cond(convention_condition, lambda xx: jnp.pi/2.0 - phiRef, lambda yy: 0.0, operand = None), 
                     lambda _: jnp.atan2(J0_Sf[1], J0_Sf[0]), 
                     operand = None)
        
        return phiJ_Sf

    def twoPN_non_spinning_orbitan_angular_momentum(self, pflag):

        # Branch functions
        def case_101(_):
            return self.flag_101_twoPN_non_spinning_orbitan_angular_momentum()

        def case_102_330(_):
            return self.flag_102_330_twoPN_non_spinning_orbitan_angular_momentum()

        def case_222_223(_):
            return self.flag_222_223_twoPN_non_spinning_orbitan_angular_momentum()

        def case_103(_):
            return self.flag_103_twoPN_non_spinning_orbitan_angular_momentum()

        def case_104(_):
            return self.flag_104_twoPN_non_spinning_orbitan_angular_momentum()

        branches = (
            case_101,
            case_102_330,
            case_222_223,
            case_103,
            case_104,
        )

        idx = jnp.where(pflag == 101, 0,
            jnp.where(jnp.isin(pflag, jnp.array([102, 220, 221, 224, 310, 311, 320, 321, 330])), 1,
            jnp.where(jnp.isin(pflag, jnp.array([222, 223])), 2,
            jnp.where(pflag == 103, 3,
            jnp.where(pflag == 104, 4, -1)))))

        return jax.lax.switch(idx, branches, operand=None)

    def flag_101_twoPN_non_spinning_orbitan_angular_momentum(self):
    
        L0   = 1.0
        L1   = 0.0
        L2   = ((3.0/2.0) + (self.eta/6.0))
        L3   = 0.0
        L4   = (81.0 + (-57.0 + self.eta)*self.eta)/24.
        L5   = 0.0
        L6   = 0.0
        L7   = 0.0
        L8   = 0.0
        L8L  = 0.0

        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_102_330_twoPN_non_spinning_orbitan_angular_momentum(self):
        # 3PN orbital angular momentum 
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = 0.0
        L8   = 0.0

        #This is the log(x) term
        L8L  = 0.0
        return L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_222_223_twoPN_non_spinning_orbitan_angular_momentum(self):
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (-7*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 5*(self.chi1L + self.chi2L)*self.eta)/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-1650*(self.chi1L + self.chi2L + self.chi1L*self.delta - self.chi2L*self.delta) + 1336*(self.chi1L + self.chi2L)*self.eta + 511*(self.chi1L - self.chi2L)*self.delta*self.eta + 28*(self.chi1L + self.chi2L)*self.eta2)/600.
        L6   = (10935 + self.eta*(-62001 + 1674*self.eta + 7*self.eta2 + 2214*self.power_of_lalpi_2))/1296.
        L7   = 0.0
        L8   = 0.0

        #This is the log(x) term
        L8L  = 0.0
        #break
        return L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_103_twoPN_non_spinning_orbitan_angular_momentum(self):

        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = (self.chi2L*(-324 + self.eta*(1119 - 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta)))
                            - self.chi1L*(324 + self.eta*(-1119 + 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta))))/32.
        L8   = 2835/128. - (self.eta*(-10677852 + 100*self.eta*(-640863 + self.eta*(774 + 11*self.eta))
                        + 26542080*GAMMA + 675*(3873 + 3608*self.eta)*self.power_of_lalpi_2))/622080. - (64*self.eta*self.log16)/3.

        L8L  = -(64.0/3.0) * self.eta


        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L
    
    def flag_104_twoPN_non_spinning_orbitan_angular_momentum(self):
        L0   = 1.0
        L1   = 0.0
        L2   = 3.0/2. + self.eta/6.0
        L3   = (5*(self.chi1L*(-2 - 2*self.delta + self.eta) + self.chi2L*(-2 + 2*self.delta + self.eta)))/6.
        L4   = (81 + (-57 + self.eta)*self.eta)/24.
        L5   = (-7*(self.chi1L*(72 + self.delta*(72 - 31*self.eta) + self.eta*(-121 + 2*self.eta)) + self.chi2L*(72 + self.eta*(-121 + 2*self.eta) + self.delta*(-72 + 31*self.eta))))/144.
        L6   = (10935 + self.eta*(-62001 + self.eta*(1674 + 7*self.eta) + 2214*self.power_of_lalpi_2))/1296.
        L7   = (self.chi2L*(-324 + self.eta*(1119 - 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta)))
                            - self.chi1L*(324 + self.eta*(-1119 + 2*self.eta*(172 + self.eta)) + self.delta*(324 + self.eta*(-633 + 14*self.eta))))/32.
        L8   = 2835/128. - (self.eta*(-10677852 + 100*self.eta*(-640863 + self.eta*(774 + 11*self.eta))
                        + 26542080*GAMMA + 675*(3873 + 3608*self.eta)*self.power_of_lalpi_2))/622080. - (64*self.eta*self.log16)/3.

        #This is the log(x) term at 4PN, x^4/2 * log(x)
        L8L  = -(64.0/3.0) * self.eta

        #Leading order in spin at all PN orders, note that the 1.5PN terms are already included. Here we have additional 2PN and 3.5PN corrections.
        L4  += (self.chi1L2*(1 + self.delta - 2*self.eta) + 4*self.chi1L*self.chi2L*self.eta - self.chi2L2*(-1 + self.delta + 2*self.eta))/2.
        L7  +=  (3*(self.chi1L + self.chi2L)*self.eta*(self.chi1L2*(1 + self.delta - 2*self.eta) + 4*self.chi1L*self.chi2L*self.eta - self.chi2L2*(-1 + self.delta + 2*self.eta)))/4.

        return  L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L


    def compute_MSA_and_check_MSA_error(self, pflag:int)->int:

        MSA_conditions = jnp.isin(pflag, jnp.array([220, 221, 222, 223, 224]))

        MSA_ERROR = 0

        MSA_ERROR = jax.lax.cond(MSA_conditions,
                                      IMRPhenomX_Initialize_MSA_System,
                                      lambda _: MSA_ERROR,
                                      operand = None)
        return MSA_ERROR
    
    def switch_to_NNLO(self, pflag:int, MSA_ERROR:int)->int:

        sub_condition = jnp.isin(pflag, jnp.array([220, 223, 224]))
        condition = (MSA_ERROR==1) & (sub_condition)
        
        return jax.lax.cond(condition,
                             lambda _: 102,
                             lambda _: pflag,
                             operand = None)

    
    def check_prescription_tag(self, n:int)->int:
        '''
        Retuns the first digit of a three digit number
        1 means NNLO
        2 means MSA
        3 means Spin Taylor
        '''
        return jnp.int32(n/100)
        

    def compute_flow_spin_taylor(self, PNRUseTunedAngles: int)->float:
        """
        Substitute function for line 324-340 in LALSimIMRPhenomX_precession.c script
        """
        
        def PNRTuned_true(_):
            return jnp.where(self.pWF['deltaF']==0.0, self.pWF['fMin'], jnp.floor_divide(self.pWF['fMin'], self.pWF['deltaF'])*self.pWF['deltaF'])
        
        def PNRTuned_false(_):
            M_MAX = 1.0 #FIXME this is definietly not the right value
            integration_buffer = jnp.where(self.pWF['deltaF']>0.0, 3*self.pWF['deltaF'], 0.5) 
            return (self.pWF['fMin'] - integration_buffer)*2 / M_MAX
        
        return jax.lax.cond(self.PNRUseTunedAngles, PNRTuned_true, PNRTuned_false, operand = None)


    def spin_taylor_code():
        '''

        Output of the spin taylor code is chi1_evolved and chi_2_evolved

        '''
        return 330

'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
End of IMRPhenomXGetAndSetPrecessionVaraibles
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
'''





#FIXME
# Wrapper of  XLALSimInspiralSpinTaylorPNEvolveOrbit : if integration is successful, stores arrays containing PN solution in  a PhenomXPInspiralArrays struct
def IMRPhenomX_InspiralAngles_SpinTaylor(chi1x: float, chi1y: float, chi1z: float, 
                                         chi2x: float, chi2y: float, chi2z: float,
                                         fmin: float, PrecVersion: int, pWF: dict, lalParams: dict):
    '''
    Output: PhenomXPInspiralArrays [out] Struct containing solutions returned by PNEvolveOrbit 
    Output: fmin_PN [out] Minimum frequency in PN solutions array
    '''


    fRef = pWF['fRef']
    m1_SI = pWF['m1_SI']
    m2_SI = pWF['m2_SI']


    s1x=chi1x 
    s1y=chi1y
    s1z=chi1z

    s2x=chi2x
    s2y=chi2y
    s2z=chi2z

    piGM = jnp.pi * (pWF['m1_SI'] + pWF['m2_SI']) * (G / C) / (C * C)


    quadparam1=pWF["quadparam1"]
    quadparam2=pWF["quadparam2"]
    lambda1=pWF["lambda1"]
    lambda2=pWF["lambda2"]

    PrecVersion_cond = (PrecVersion==311) | (PrecVersion==321)
    quadparam1 = jnp.where(PrecVersion_cond, 1, quadparam1)
    quadparam2 = jnp.where(PrecVersion_cond, 1, quadparam2)
    lambda1 = jnp.where(PrecVersion_cond, 0, lambda1)
    lambda2 = jnp.where(PrecVersion_cond, 0, lambda2)

    #Compress line 4634-4637
    phaseO = jnp.where(lalParams['phaseO']==-1, 7, lalParams['phaseO'])
    spinO = jnp.where(lalParams['spinO']==-1, 6, lalParams['spinO'])
    tideO = jnp.where(lalParams['tideO']==-1, 12, lalParams['tideO'])
    lscorr = 0.0
    
    #Skip 4638-4655

    lnhatx = 0.0
    lnhaty = 0.0
    lnhatz = 1.0

    e1x = 1.0
    e1y = 0.0
    e1z = 0.0
    

    """
    If PhenomXPSpinTaylorVersion is None: set it to "SpinTaylorT4"
    """

    approx = lalParams['approx_name']


    fMECO_Hz = XLALSimIMRPhenomXUtilsMftoHz(pWF['fMECO'], pWF['Mtot'])
    fmin_condition = (fmin > fMECO_Hz) & ((PrecVersion==320) | (PrecVersion==321))
    fmin = jnp.where(fmin_condition, fMECO_Hz, fmin)

    fCut = XLALSimIMRPhenomXUtilsMftoHz(pWF['fRING']+8 * pWF['fDAMP'], pWF['Mtot'])
    

    deltaT_coarse = .5 * lalParams['coarse_fac'] / fCut

    
    #Line 4681
    #if(coarse_fac  < 1) { XLAL_ERROR(XLAL_EDOM, "Coarse factor must be >= 1!\n");}

    #Line 4685-4686
    #fS = fmin
    #fE = fCut

    fref_zero_or_same_to_fmin = (fRef < 1e-10) | (jnp.abs(fRef-fmin) < 1e-10)

    #Compress line 4688-4780
    PhenomXPInspiralArrays = jax.lax.cond(fref_zero_or_same_to_fmin, integrate_forward, integrate_both_sides, fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI, s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
    #V_PN, Phi_PN, S1x_PN, S1y_PN, S1z_PN, S2x_PN, S2y_PN, S2z_PN, LNhatx_PN, LNhaty_PN, LNhatz_PN, E1x_PN, E1y_PN, E1z_PN

    #Line 4782
    #if lalParams['coarse_fac'] > 1: # ignoring this flag. I force it to be ==1.  

    ## copy coarse-grid data to fine-grid
    ## destroy coarse-grid

    #check that the first frequency node returned is indeed below the fmin requested, to avoid interpolation errors. If not return an error which will trigger the fallback to MSA

    fminPN=jnp.power(PhenomXPInspiralArrays[0][0],3.)/piGM

    spin_taylor_success_check = (fminPN<0.0) | (fminPN>fmin)
    status = jnp.where(spin_taylor_success_check, 0, 1)
    return PhenomXPInspiralArrays, status


#FIXME
def XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT: float, m1_SI: float, m2_SI: float, fStart: float, fEnd: float,
                                           s1x: float, s1y: float, s1z: float, s2x: float, s2y: float, s2z: float, 
                                           lnhatx: float, lnhaty: float, lnhatz: float, e1x: float, e1y: float, e1z: float,
                                           lambda1: float, lambda2: float, quadparam1: float, quadparam2: float, spinO: int,
                                           tideO: int, phaseO: float, lscorr: int, approx: str):
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral_spin_taylor__c.html#ga35cfdf3082e09cc97cda9e11ba4c2bff

    """
    spin0 >= 7 is not allowed.
    fStart < 0 is not allowed. 
    fEnd < 0 is not allowed.
    if fEnd<fStart && fEnd != 0.0, sgn = -1 else sign 1.
    
    """

    if approx=='SpinTaylorT4':
        from .LALSimInspiralSpinTaylor import XLALSimInspiralSpinTaylorPNEvolveOrbit as SpinTaylor4EvolveOrbit
        V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = SpinTaylor4EvolveOrbit(deltaT = deltaT, 
                                                                                                             m1_SI=m1_SI, 
                                                                                                             m2_SI=m2_SI, 
                                                                                                             fStart=fStart,
                                                                                                             fEnd = fEnd,
                                                                                                             s1x = s1x, s1y = s1y, s1z = s1z,
                                                                                                             s2x = s2x, s2y = s2y, s2z = s2z,
                                                                                                             lnhatx = lnhatx, lnhaty = lnhaty, lnhatz = lnhatz,
                                                                                                             e1x = e1x, e1y = e1y, e1z = e1z,
                                                                                                             lambda1 = lambda1, lambda2=lambda2,
                                                                                                             quadparam1=quadparam1, quadparam2=quadparam2,
                                                                                                             spinO=spinO, tideO=tideO, phaseO=phaseO,
                                                                                                             lscorr=lscorr)
    elif approx=='SpinTaylorT5':
        pass
    elif approx=='SpinTaylorT1':
        pass
    else:
        pass

    m1sec = m1_SI / MSUN * MTSUN_SI
    m2sec = m2_SI / MSUN * MTSUN_SI
    Msec = m1sec + m2sec
    Mcsec = Msec * pow( m1sec*m2sec/Msec/Msec, 0.6)


    #/* Estimate length of waveform using Newtonian t(f) formula */
    #/* Time from freq. = fStart to infinity */
    dtStart = (5.0/256.0) * pow(jnp.pi,-8.0/3.0) * pow(Mcsec * fStart,-5.0/3.0) / fStart
    #/* Time from freq. = fEnd to infinity. Set to zero if fEnd=0 */
    dtEnd = jnp.where(fEnd==0.0, 0, (5.0/256.0) * pow(jnp.pi,-8.0/3.0) * pow(Mcsec * fEnd,-5.0/3.0) / fEnd)
    lengths = dtStart - dtEnd


    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z


def integrate_forward(fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    '''
    If fRef is zero or is equal to fmin, we only need to integrate from fmin to fCut, i.e., forward. 
    This function is called to perform forward integration. 
    Line 4690-4697
    '''

    fS = fmin
    fE = fCut

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z


def integrate_both_sides(fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    '''
    If fRef > fmin, we first integrate from fRef to fmin and then fRef to fCut. 
    This function is called to integrate on both sides
    FIXME: We may want to get rid of jnp.append by making arrays of zeros and populating them. 
    Line 4701-4773
    '''

    fS =  fRef
    fE = fmin - 0.5

    # Backward integration
    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)


    fS = fRef
    fE = fCut
    #Skipping the sanity check of if...else. Just jump to forward integration. 
    V_forward, Phi_forward, S1x_forward, S1y_forward, S1z_forward, S2x_forward, S2y_forward, S2z_forward, LNhatx_forward, LNhaty_forward, LNhatz_forward, E1x_forward, E1y_forward, E1z_forward = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, 
                                                                                                                                            m1_SI, m2_SI, fS, fE, s1x, s1y, s1z, s2x, s2y,
                                                                                                                                            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1,lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
    V = jnp.append(V, V_forward)
    Phi = jnp.append(Phi, Phi_forward)
    S1x = jnp.append(S1x, S1x_forward)
    S1y = jnp.append(S1y, S1y_forward)
    S1z = jnp.append(S1z, S1z_forward)

    S2x = jnp.append(S2x, S2x_forward)
    S2y = jnp.append(S2y, S2y_forward)
    S2z = jnp.append(S2z, S2z_forward)

    LNhatx = jnp.append(LNhatx, LNhatx_forward)
    LNhaty = jnp.appnd(LNhaty, LNhaty_forward)
    LNhatz = jnp.append(LNhatz, LNhatz_forward)
    
    E1x = jnp.append(E1x, E1x_forward)
    E1y = jnp.append(E1y, E1y_forward)
    E1z = jnp.append(E1z, E1z_forward)


    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z

#FIXME
def IMRPhenomX_SetPrecessingRemnantParams(pWF, lalParams):
    return None


#FIXME
def IMRPhenomX_PNR_GetAndSetPNRVariables(*args):
    #https://github.com/GW-JAX-Team/ripple/blob/f46fe3610bce9b0c82927f1de966190cf5e0ae53/src/ripplegw/waveforms/imr_phenom_xphm/lal_sim_imr_phenom_x_pnr_internals.py#L25
    return None


#FIXME
def IMRPhenomX_PNR_GetAndSetCoPrecParams(*args):
    return None


def XLALSimIMRPhenomXUtilsMftoHz(Mf, Mtot_Msun):
    return Mf / (MTSUN_SI*Mtot_Msun)


#FIXME
def IMRPhenomX_Initialize_MSA_System(*args):
    return 1

#FIXME
def IMRPhenomX_GetandSetModes(ModeArray: list, IMRPhenomXPrecessionStruct: dict):
    return None




def XLALSimIMRPhenomXLPNAnsatz(v, LNorm, L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L):
    """
    Computes the PN orbital angular momentum expansion.
    v : Input velocity.
    LNorm : Orbital angular momentum normalization (e.g. η / sqrt(x)).
    L0–L8, L8L : PN coefficients.
    
    Returns
    L : Post-Newtonian angular momentum.
    """

    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)

    L = (
        L0
        + L1 * sqx
        + L2 * x
        + L3 * (x * sqx)
        + L4 * x2
        + L5 * (x2 * sqx)
        + L6 * x3
        + L7 * (x3 * sqx)
        + L8 * x4
        + L8L * x4 * jnp.log(x)
    )
    return LNorm * L



def IMRPhenomX_rotate_z(angle, v): 
    """
    Rotate a 3D vector v = (vx, vy, vz) about the z-axis by given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """
    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot = vx * cosa - vy * sina
    vy_rot = vx * sina + vy * cosa
    vz_rot = vz  # unchanged

    return jnp.array([vx_rot, vy_rot, vz_rot])


def IMRPhenomX_rotate_y(angle, v):
    """
    Rotate a 3D vector v = (vx, vy, vz) about the y-axis by a given angle.
    Args:
        angle: scalar angle in radians (JAX array or float)
        v: array-like of shape (3,) representing [vx, vy, vz]
    Returns:
        rotated vector as a JAX array of shape (3,)
    """

    cosa = jnp.cos(angle)
    sina = jnp.sin(angle)
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vx_rot =  vx * cosa + vz * sina
    vy_rot =  vy  # unchanged
    vz_rot = -vx * sina + vz * cosa

    return jnp.array([vx_rot, vy_rot, vz_rot])




def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(pPrec, v, pWF):
    # Wrapper to generate \f$\phi_z\f$, \f$\zeta\f$ and \f$\cos \theta_L\f$ at a given frequency

    vout = jnp.array([0, 0, 0])
    pPrec = None


    L_norm = pWF['eta']/v
    J_norm = IMRPhenomX_JNorm_MSA(L_norm, pPrec)

    L_norm3PN       = 0.0

    # Compressing line 2212 - 2220
    cond = (pPrec.IMRPhenomXPrecVersion == 222) | (pPrec.IMRPhenomXPrecVersion == 223)
    L_norm3PN = jax.lax.cond(cond, IMRPhenomX_L_norm_3PN_of_v, XLALSimIMRPhenomXLPNAnsatz, v, L_norm, pPrec)

    '''
    if (pPrec.IMRPhenomXPrecVersion == 222) | (pPrec.IMRPhenomXPrecVersion == 223):
        L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)

    else:
        L_norm3PN = XLALSimIMRPhenomXLPNAnsatz(v, L_norm, pPrec.L0, pPrec.L1, pPrec.L2, pPrec.L3, pPrec.L4, pPrec.L5, pPrec.L6, pPrec.L7, pPrec.L8, pPrec.L8L)
    '''
    

    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN, pPrec)
    vRoots    = IMRPhenomX_Return_Roots_MSA(L_norm, J_norm, pPrec)


    pPrec.S32  = vRoots.x
    pPrec.Smi2 = vRoots.y
    pPrec.Spl2 = vRoots.z

    pPrec.Spl2mSmi2   = pPrec.Spl2 - pPrec.Smi2
    pPrec.Spl2pSmi2   = pPrec.Spl2 + pPrec.Smi2
    pPrec.Spl         = jnp.sqrt(pPrec.Spl2)
    pPrec.Smi         = jnp.sqrt(pPrec.Smi2)

    SNorm = IMRPhenomX_Return_SNorm_MSA(v, pPrec)
    pPrec.S_norm      = SNorm
    pPrec.S_norm_2    = SNorm * SNorm

    vMSA = {0.,0.,0.}

    # Compressing line 2245-2249
    vMSA_correction = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    cond = (jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5)
    vMSA = jnp.where(cond, vMSA_correction, vMSA)
    '''
    if(jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5):
    
        #Get phiz_0_MSA and zeta_0_MSA
        vMSA = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    '''

    phiz_MSA     = vMSA.x
    zeta_MSA     = vMSA.y

    phiz         = IMRPhenomX_Return_phiz_MSA(v, J_norm, pPrec)
    zeta         = IMRPhenomX_Return_zeta_MSA(v, pPrec)
    cos_theta_L        = IMRPhenomX_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)

    vout[0] = phiz + phiz_MSA
    vout[1] = zeta + zeta_MSA
    vout[2] = cos_theta_L

    return vout


def IMRPhenomX_JNorm_MSA(LNorm:float, pPrec)->float:
    JNorm2 = (LNorm * LNorm + 2.0 * LNorm * pPrec.c1_over_eta + pPrec.SAv2)
    return jnp.sqrt(JNorm2)


def IMRPhenomX_L_norm_3PN_of_v(v: jax.Array, L_norm: float, pPrec)->float:
    v2 = v*v
    term_4 = pPrec.constants_L[4]
    term_3 = pPrec.constants_L[3]
    term_2 = pPrec.constants_L[2]
    term_1 = pPrec.constants_L[1]
    term_0 = pPrec.constants_L[0]
    L_norm3PN = L_norm*(1. + v2*(term_0 + v*term_1 + v2*(term_2 + v*term_3 + v2*(term_4))))

    return L_norm3PN


def IMRPhenomX_Return_Roots_MSA(LNorm, JNorm, pPrec):
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)  
    B, C, D = vBCD[0], vBCD[1], vBCD[2]

    B2 = B * B
    B3 = B2 * B
    BC = B * C

    p = C - B2 / 3.0
    qc = (2.0 / 27.0) * B3 - BC / 3.0 + D

    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / (p * sqrtarg)
    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)
    
    print(f'{p=}, {sqrtarg=}, {theta=}, {B=}, {B2=}, {C=}')
    
    vector_condition = jnp.logical_or(jnp.isnan(theta),
                                                   (jnp.isnan(sqrtarg)))
    scalar_condition = jnp.logical_or.reduce(jnp.array([(pPrec.dotS1Ln == 1.0),
                                                   (pPrec.dotS2Ln == 1.0),
                                                   (pPrec.dotS1Ln == -1.0),
                                                   (pPrec.dotS2Ln == -1.0),
                                                   (pPrec.S1_norm_2 == 0.0),
                                                   (pPrec.S2_norm_2 == 0.0)]))
    invalid_case = jnp.logical_or(vector_condition, scalar_condition)

    def roots_when_valid():
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 4.0 * jnp.pi / 3.0) - B / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * jnp.pi / 3.0) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        tmp6 = jnp.where(
            (tmp4 - tmp3 > 0.0) & (tmp5 - tmp3 < 0.0),
            tmp3,
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2)
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return jnp.array([S32, Smi2, Spl2])

    def roots_when_invalid():
        Smi2 = pPrec.S_0_norm**2 * jnp.ones_like(LNorm)
        Spl2 = Smi2 + 1e-9
        S32 = jnp.zeros_like(LNorm)
        return jnp.array([S32, Smi2, Spl2])

    roots_array = jnp.where(
        jnp.atleast_1d(invalid_case),
        roots_when_invalid(),
        roots_when_valid()
    )
    
    print(f'{roots_array=}')

    return roots_array


def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec):
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = pPrec.S1_norm_2
    S2Norm2 = pPrec.S2_norm_2
    q       = pPrec.qq
    eta     = pPrec.eta
    delta   = pPrec.delta_qq
    deltaSq = delta * delta
    Seff    = pPrec.Seff

    J2mL2   = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = ((LNorm2 + S1Norm2) * q +
               2.0 * LNorm * Seff -
               2.0 * JNorm2 -
               S1Norm2 - S2Norm2 +
               (LNorm2 + S2Norm2) / q)

    # C coefficient (Eq. B3)
    C_coeff = (J2mL2Sq -
               2.0 * LNorm * Seff * J2mL2 -
               2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) +
               4.0 * eta * LNorm2 * Seff * Seff -
               2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm +
               2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2)

    # D coefficient (Eq. B4)
    D_coeff = (((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq +
               deltaSq * (S1Norm2 - S2Norm2)**2 * LNorm2 / eta +
               2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2)

    return jnp.array([B_coeff, C_coeff, D_coeff])


def IMRPhenomX_Return_SNorm_MSA(v, pPrec):

    v2 = v * v

    cancel_condition = jnp.abs(pPrec.Smi2 - pPrec.Spl2) < 1e-5

    def sn_zero(_):
        sn = jnp.array(0.0)
        return sn

    def sn_jacobi(_):
        # Equation 25 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
        m = (pPrec.Smi2 - pPrec.Spl2) / (pPrec.S32 - pPrec.Spl2)

        psi = IMRPhenomX_psiofv(
            v, v2,
            pPrec.psi0, pPrec.psi1, pPrec.psi2,
            pPrec
        )

        # Jacobi elliptic functions
        sn, cn, dn = gsl_sf_elljac_e(psi, m) # FIXME
        return sn

    sn = jax.lax.cond(cancel_condition, sn_zero, sn_jacobi, operand=None)

    # Equation 23 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    SNorm2 = pPrec.Spl2 + (pPrec.Smi2 - pPrec.Spl2) * sn * sn

    return jnp.sqrt(SNorm2)



def gsl_sf_elljac_e(x, y):
    """
    TODO: Not yet implemented
    """
    return jnp.array(1.), jnp.array(2.), jnp.array(3.)

def IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, pPrec):
    # Equation 51 in arXiv:1703.03967
    return psi0 - 0.75 * pPrec.g0 * pPrec.delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)



def IMRPhenomX_Return_MSA_Corrections_MSA(
    v, 
    LNorm, 
    JNorm, 
    pPrec
    ):
    
    v2 = v * v

    # Sets c0, c2 and c4 in pPrec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec)
    # Sets d0, d2 and d4 in pPrec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec)  

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0
    
    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (pPrec.Spl2 / (JNorm * LNorm)))
    B_theta_L = 0.5 * pPrec.Spl2mSmi2 / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(v, v2, pPrec) + pPrec.psi0
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(v, pPrec) 

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    C2num = (c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) -
             c2 * d0 * (d2 + 2.0 * d4) +
             c4 * d0 * (two_d0 + d2))
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    C2 = C2num / C2den

    Cphi = C1 + C2
    Dphi = C1 - C2

    def compute_Cphi_term():
        
        return jnp.abs((
            (c4 * d0 * ((2 * d0 + d2) + sd) -
                c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - 
                sd))) / C2den) * (sqrt_nc / (nc - 1.0)) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))) / psi_dot
        
    def compute_Dphi_term():
            return jnp.abs((
                (-c4 * d0 * ((2 * d0 + d2) - sd) +
                 c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                 c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den
            ) * (sqrt_nd / (nd - 1.0)) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))) / psi_dot

    phiz_0_MSA_Cphi_term = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    #####  restart from here
    vMSA_y = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
                phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2))

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return jnp.array([vMSA_x, vMSA_y, 0.0])



def IMRPhenomX_Return_Psi_MSA(v, v2, pPrec):
    return -0.75 * pPrec.g0 * pPrec.delta_qq * (1.0 + pPrec.psi1 * v + pPrec.psi2 * v2) / (v2 * v)

def IMRPhenomX_Return_Psi_dot_MSA(v, pPrec):
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * pPrec.Seff) * jnp.sqrt(pPrec.inveta)
    psi_dot = 0.5 * A_coeff * jnp.sqrt(pPrec.Spl2 - pPrec.S32)

    return psi_dot


def IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec):
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    JNorm2 = JNorm * JNorm
    Seff = pPrec.Seff


    x = JNorm * (
        0.75 * (1.0 - Seff * v) * v2 * (
            pPrec.eta3
            + 4.0 * pPrec.eta3 * Seff * v
            - 2.0 * pPrec.eta * (
                JNorm2 - pPrec.Spl2 + 2.0 * (pPrec.S1_norm_2 - pPrec.S2_norm_2) * pPrec.delta_qq
            ) * v2
            - 4.0 * pPrec.eta * Seff * (JNorm2 - pPrec.Spl2) * v3
            + (JNorm2 - pPrec.Spl2) ** 2 * v4 * pPrec.inveta
        )
    )

    y = JNorm * (
        -1.5 * pPrec.eta * (pPrec.Spl2 - pPrec.Smi2)
        * (1.0 + 2.0 * Seff * v - (JNorm2 - pPrec.Spl2) * v2 * pPrec.inveta**2)
        * (1.0 - Seff * v) * v4
    )

    z = JNorm * (
        0.75 * pPrec.inveta * (pPrec.Spl2 - pPrec.Smi2) ** 2
        * (1.0 - Seff * v) * v6
    )

    return jnp.array([x, y, z])



def IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec):
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = - (JNorm2 - (LNorm + pPrec.Spl)) ** 2 * (JNorm2 - (LNorm - pPrec.Spl)) ** 2

    y = -2.0 * (pPrec.Spl2 - pPrec.Smi2) * (JNorm2 + LNorm2 - pPrec.Spl2)

    z = -(pPrec.Spl2 - pPrec.Smi2) ** 2

    return jnp.array([x, y, z])




def IMRPhenomX_Return_phiz_MSA(
    v: float, 
    JNorm: float, 
    pPrec
    ) -> float:
    
    invv = 1.0 / v
    invv2 = invv * invv
    LNewt = pPrec.eta / v

    c1 = pPrec.c1
    c12 = c1 * c1

    SAv2 = pPrec.SAv2
    SAv = pPrec.SAv
    invSAv = pPrec.invSAv
    invSAv2 = pPrec.invSAv2

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + JNorm * pPrec.eta + pPrec.eta * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))

    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (JNorm * pPrec.inveta**4) * (
        0.5 * c12 - (c1 * pPrec.eta2 * invv) / 6.0 - (SAv2 * pPrec.eta2) / 3.0 - (pPrec.eta4 * invv2) / 3.0
    ) - (0.5 * c1 * pPrec.inveta) * (
        c12 * pPrec.inveta**4 - SAv2 * pPrec.inveta**2
    ) * log1

    phiz_1_coeff = (
        -0.5 * JNorm * pPrec.inveta**2 * (c1 + pPrec.eta * LNewt)
        + 0.5 * pPrec.inveta**3 * (c12 - pPrec.eta2 * SAv2) * log1
    )

    phiz_2_coeff = -JNorm + SAv * log2 - c1 * log1 * pPrec.inveta

    phiz_3_coeff = JNorm * v - pPrec.eta * log1 + c1 * log2 * invSAv

    phiz_4_coeff = (
        0.5 * JNorm * invSAv2 * v * (c1 + v * SAv2)
        - 0.5 * invSAv2 * invSAv * (c12 - pPrec.eta2 * SAv2) * log2
    )

    phiz_5_coeff = (
        -JNorm * v * (
            0.5 * c12 * invSAv2 * invSAv2
            - c1 * v * invSAv2 / 6.0
            - v * v / 3.0
            - pPrec.eta2 * invSAv2 / 3.0
        )
        + 0.5 * c1 * invSAv2 * invSAv2 * invSAv * (c12 - pPrec.eta2 * SAv2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
 
    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0
 
    # Note that the <\Omega_z>^(n) are given by pPrec->Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * pPrec.Omegaz0_coeff
        + phiz_1_coeff * pPrec.Omegaz1_coeff
        + phiz_2_coeff * pPrec.Omegaz2_coeff
        + phiz_3_coeff * pPrec.Omegaz3_coeff
        + phiz_4_coeff * pPrec.Omegaz4_coeff
        + phiz_5_coeff * pPrec.Omegaz5_coeff
        + pPrec.phiz_0
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out
    
def IMRPhenomX_Return_zeta_MSA(
    v: float, 
    pPrec
    ) -> float:
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = pPrec.eta * (
        pPrec.Omegazeta0_coeff * invv3 +
        pPrec.Omegazeta1_coeff * invv2 +
        pPrec.Omegazeta2_coeff * invv +
        pPrec.Omegazeta3_coeff * logv +
        pPrec.Omegazeta4_coeff * v +
        pPrec.Omegazeta5_coeff * v2
    ) + pPrec.zeta_0

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def IMRPhenomX_costhetaLJ(
    L_norm: float, 
    J_norm: float, 
    S_norm: float
    ) -> float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / L_norm * J_norm

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ