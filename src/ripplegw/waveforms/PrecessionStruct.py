import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C, MTSUN_SI
import jax
from spherical_harmonics import *

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
        """

        # Here we assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
        self.pWF = pWF
        #self.pWF['LALparams'] = lalParams
        self.lalParams = lalParams
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
        self.debug_prec = debug_flag

        # Get IMRPhenomX precession version from LAL dictionary
        self.IMRPhenomXPrecVersion = lalParams['IMRPhenomXPrecVersion']
        self.IMRPhenomXPrecVersion = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        
        ## Condition line 109
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

        #### Skipping the multibanding bookkeeping around line 136

        # Define a number of convenient local parameters #############
        m1        = m1_SI / self.pWF['Mtot_SI']   #/* Normalized mass of larger companion:   m1_SI / Mtot_SI */
        m2        = m2_SI / self.pWF['Mtot_SI']   #/* Normalized mass of smaller companion:  m2_SI / Mtot_SI */
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
        
        self.pWF['M'] = M
        self.pWF['m1_2'] = m1_2
        self.pWF['m2_2'] = m2_2

        self.q = m1/m2


        # Powers of eta
        eta       = self.pWF['eta']
        eta2      = eta*eta
        eta3      = eta*eta2
        eta4      = eta*eta3
        eta5      = eta*eta4
        eta6      = eta*eta5

        # \delta in terms of q > 1
        delta     = self.pWF['delta']
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

        chi_eff   = self.pWF['chiEff']

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
        I do not know a jax friendly version for assert

        condition = (jnp.logical_not(self.PNRUseTunedAngles)) | (pWF['PNR_SINGLE_SPIN'] != 1)

        if condition:
            assert jnp.abs(self.chi1_norm) <= 1.0
            assert jnp.abs(self.chi2_norm) <= 1.0
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
        self.pWF['chiTot_perp'] = self.chiTot_perp

        #/* disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit */
        cond = (chi_in_plane<1e-7) & (self.PNRUseTunedAngles == 1) & (self.pWF['PNR_SINGLE_SPIN']!=1)
        self.PNRUseTunedAngles, lalParams['PNRUseTunedAngles'], self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']  = jnp.where(cond, jnp.array([False, False, False, False, False]), jnp.array([lalParams['PNRUseTunedAngles'], self.PNRUseTunedAngles, self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']]))
            


        # Calculate the effective precessing spin parameter (Schmidt et al, PRD 91, 024043, 2015): m1 > m2, so body 1 is the larger black hole
        self.A1             = 2.0 + (3.0 * m2) / (2.0 * m1)
        self.A2             = 2.0 + (3.0 * m1) / (2.0 * m2)
        self.ASp1           = self.A1 * self.S1_perp
        self.ASp2           = self.A2 * self.S2_perp

        #/* S_p = max(A1 S1_perp, A2 S2_perp) */
        num       = jnp.where(self.ASp2 > self.ASp1, self.ASp2, self.ASp1)
        den       = jnp.where(m2 > m1 , self.A2*m2_2, self.A1*m1_2)

        #/* chi_p = max(A1 * Sp1 , A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH */
        chip      = num / den
        chi1L     = chi1z
        chi2L     = chi2z


        self.chi_p          = chip
        #// (PNRUseTunedCoprec)
        self.pWF['chi_p']        = self.chi_p
        self.phi0_aligned   = self.pWF['phi0']

        #/* Effective (dimensionful) aligned spin */
        self.SL             = chi1L*m1_2 + chi2L*m2_2

        #/* Effective (dimensionful) in-plane spin */
        self.Sperp          = chip * m1_2                 # /* m1 > m2 */

        self.MSA_ERROR      = 0

        self.pWF22AS = None


        #// get first digit of precessing version: this tags the method employed to compute the Euler angles
        #// 1: NNLO 2: MSA 3: SpinTaylor (numerical)
        precversionTag = (self.IMRPhenomXPrecVersion-(self.IMRPhenomXPrecVersion%100))/100
        precversionTag = jnp.int32(precversionTag)
 
        #/* start of SpinTaylor code */ line 294

        #####################################
        precversionTag_3_true = None
        precversionTag_3_true = False
        self.precversionTag3()
        #end of SpinTaylor code
        # done up to line 484


        # Update the tag to 223 after 330 failed
        precversionTag = (self.IMRPhenomXPrecVersion-(self.IMRPhenomXPrecVersion%100))/100
        precversionTag = jnp.int(precversionTag)
        pflag = self.IMRPhenomXPrecVersion
        if pflag!= "101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321, 330":
            print("Invalid flag")
        
        '''
        switch 
        
        cases 101-104
        break 

        cases 220-224
        {sanity checks and break}


        case 310 - 330
        break


        default
        not recognised

        switch complete
        '''

        #line 545
        self.precessing_tag=precversionTag

        

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

        status = IMRPhenomX_PNR_GetAndSetPNRVariables()
        #status check

        self.alphaPNR = 0.0
        self.betaPNR = 0.0
        self.gammaPNR = 0.0

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Get and/or store CoPrec params into pWF and pPrec     /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        status = IMRPhenomX_PNR_GetAndSetCoPrecParams()
        #status check

        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/

        if pflag == "220, 221, 222, 223, 224":
            IMRPhenomX_Initialize_MSA_System()
            if self.MSA_ERROR ==1:
                if pflag=="220, 223, 224":
                    #XLAL_PRINT_WARNING("Warning: Initialization of MSA system failed. Defaulting to NNLO angles using 3PN aligned-spin approximation.")
                    pass
                else:
                    #XLAL_ERROR(XLAL_EDOM,"Error: IMRPhenomX_Initialize_MSA_System failed to initialize. Terminating.\n")
                    pass

        if DEBUG==1:
            pass

        #/*...#...#...#...#...#...#...#...#...#...#...#...#...#...#.../
        #/      Compute and set final spin and RD frequency           /
        #/...#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/
        
        IMRPhenomX_SetPrecessingRemnantParams(pWF,pPrec,lalParams)
        #/*..#...#...#...#...#...#...#...#...#...#...#...#...#...#...*/


        # /* Useful powers of \chi_p */
        chip2    = chip * chip

        #/* Useful powers of spins aligned with L */
        chi1L2   = chi1L * chi1L
        chi2L2   = chi2L * chi2L

        log16    = 2.772588722239781

        #/*  Cache the orbital angular momentum coefficients for future use.

        #References:
        #- Kidder, PRD, 52, 821-847, (1995), arXiv:gr-qc/9506022
        #- Blanchet, LRR, 17, 2, (2014), arXiv:1310.1528
        #- Bohe et al, 1212.5520v2
        #- Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118

        switch(pflag)

        case 101:
        self.L0   = 1.0
        self.L1   = 0.0
        self.L2   = ((3.0/2.0) + (eta/6.0))
        self.L3   = 0.0
        self.L4   = (81.0 + (-57.0 + eta)*eta)/24.
        self.L5   = 0.0
        self.L6   = 0.0
        self.L7   = 0.0
        self.L8   = 0.0
        self.L8L  = 0.0
        #break

        case 102, 220, 221, 224, 310, 311, 320, 321
        case 330

        self.L0   = 1.0
        self.L1   = 0.0
        self.L2   = 3.0/2. + eta/6.0
        self.L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
        self.L4   = (81 + (-57 + eta)*eta)/24.
        self.L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
        self.L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*powers_of_lalpi.two))/1296.
        self.L7   = 0.0
        self.L8   = 0.0

        #// This is the log(x) term
        self.L8L  = 0.0
        #break

        case 222

        case 223
        self.L0   = 1.0
        self.L1   = 0.0
        self.L2   = 3.0/2. + eta/6.0
        self.L3   = (-7*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 5*(chi1L + chi2L)*eta)/6.
        self.L4   = (81 + (-57 + eta)*eta)/24.
        self.L5   = (-1650*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 1336*(chi1L + chi2L)*eta + 511*(chi1L - chi2L)*delta*eta + 28*(chi1L + chi2L)*eta2)/600.
        self.L6   = (10935 + eta*(-62001 + 1674*eta + 7*eta2 + 2214*powers_of_lalpi.two))/1296.
        self.L7   = 0.0
        self.L8   = 0.0

        #// This is the log(x) term
        self.L8L  = 0.0
        #break

        case 103:
            # do something
        case 104:
            # do something
        
        default:
            print("Error: IMRPhenomXPrecVersion not recognized. Requires version 101, 102, 103, 104, 220, 221, 222, 223, 224, 310, 311, 320, 321 or 330.\n")

        ## End of pflag line 755

        #/* Reference orbital angular momentum */
        self.LRef = M * M * XLALSimIMRPhenomXLPNAnsatz(pWF['v_ref'], pWF['eta'] / pWF['v_ref'], self.L0, self.L1, self.L2, self.L3, self.L4, self.L5, self.L6, self.L7, self.L8, self.L8L) 


        #/*
        #In the following code block we construct the convetions that relate the source frame and the LAL frame.

        #A detailed discussion of the conventions can be found in Appendix C and D of arXiv:2004.06503 and https://dcc.ligo.org/LIGO-T1500602

        #/* Get source frame (*_Sf) J = L + S1 + S2. This is an instantaneous frame in which L is aligned with z */
        self.J0x_Sf = (m1_2)*chi1x + (m2_2)*chi2x
        self.J0y_Sf = (m1_2)*chi1y + (m2_2)*chi2y
        self.J0z_Sf = (m1_2)*chi1z + (m2_2)*chi2z + self.LRef

        self.J0     = jnp.sqrt(self.J0x_Sf*self.J0x_Sf + self.J0y_Sf*self.J0y_Sf + self.J0z_Sf*self.J0z_Sf)

        #/* Get angle between J0 and LN (z-direction) */
        if(self.J0 < 1e-10):

            XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n")
            self.thetaJ_Sf = 0.0

        else:
        
            self.thetaJ_Sf = acos(self.J0z_Sf / self.J0)
        

        phiRef = self.pWF['phiRef_In']

        convention     = self.lalParams['PhenomXPConvention']

        if convention:
            print('do something') 

        if DEBUG:
            print(convention)

        #/* Get azimuthal angle of J0 in the source frame */
        if (fabs(self.J0x_Sf) < MAX_TOL_ATAN & fabs(self.J0y_Sf) < MAX_TOL_ATAN)
            print('do something')
        
        else:
            self.phiJ_Sf = atan2(self.J0y_Sf, self.J0x_Sf) #/* azimuthal angle of J0 in the source frame */

        self.phi0_aligned = - self.phiJ_Sf


        switch(convention)
            {
                case 0:
                {
                pWF->phi0 = self.phi0_aligned
                break
                }
                case 1:
                {
                pWF->phi0 = 0
                break
                }
                case 5:
                case 6:
                case 7:
                {
                break
                }
            }

        

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

        #/* Determine kappa via rotations, as above */
        self.Nx_Sf = jnp.sin(self.pWF['inclination'])*cos((LAL_PI / 2.0) - phiRef)
        self.Ny_Sf = jnp.sin(self.pWF['inclination'])*sin((LAL_PI / 2.0) - phiRef)
        self.Nz_Sf = jnp.cos(self.pWF['inclination'])

        tmp_x = self.Nx_Sf
        tmp_y = self.Ny_Sf
        tmp_z = self.Nz_Sf

        IMRPhenomX_rotate_z(-self.phiJ_Sf,   &tmp_x, &tmp_y, &tmp_z)
        IMRPhenomX_rotate_y(-self.thetaJ_Sf, &tmp_x, &tmp_y, &tmp_z)

        #/* Note difference in overall - sign w.r.t PhenomPv2 code */
        self.kappa = XLALSimIMRPhenomXatan2tol(tmp_y,tmp_x, MAX_TOL_ATAN)

        #/* Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1} */
        tmp_x = 0.0
        tmp_y = 0.0
        tmp_z = 1.0
        IMRPhenomX_rotate_z(-self.phiJ_Sf,   tmp_x, tmp_y, tmp_z)
        IMRPhenomX_rotate_y(-self.thetaJ_Sf, tmp_x, tmp_y, tmp_z)
        IMRPhenomX_rotate_z(-self.kappa,     tmp_x, tmp_y, tmp_z)


        if jnp.abs(tmp_x) < MAX_TOL_ATAN & jnp.abs(tmp_y) < MAX_TOL_ATAN:
            print('do something')
            "Case again"
        else:
            "Case again"
            print('do something else')


        # Compress line 931-966
        self.thetaJN, self.Nz_Jf, self.Nx_Jf = jax.lax.cond(jnp.isin(convention, jnp.array([0, 5])), self.thetaJN_Nz_Nx_0_5, self.thetaJN_Nz_Nx_1_6_7, operand = None)


        '''
            Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.

            The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
            i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.

            However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
            NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).

            The triads differ from each other by a rotation around N by an angle \zeta. We therefore need to rotate 
            the polarizations by an angle 2 \zeta.
        '''
        #Compressed line 983  to 991
        self.Xx_Sf = -jnp.cos(pWF['inclination']) * jnp.sin(phiRef)
        self.Xy_Sf = -jnp.cos(pWF['inclination']) * jnp.cos(phiRef)
        self.Xz_Sf = +jnp.sin(pWF['inclination'])


        tmp_v = jnp.array([self.Xx_Sf, self.Xy_Sf, self.Xz_Sf])
        tmp_v = IMRPhenomX_rotate_z(-self.phiJ_Sf, tmp_v)
        tmp_v = IMRPhenomX_rotate_y(-self.thetaJ_Sf, tmp_v)
        tmp_v = IMRPhenomX_rotate_z(-self.kappa, tmp_v)


        '''

            The components tmp_i are now the components of X in the J frame.

            We now need the polar angle of this vector in the P, Q basis of Arun et al:

                P = (N x J) / |NxJ|

            Note, that we put N in the (pos x)z half plane of the J frame 

        '''
        #Compress line 1002-1034
        self.PArunx_Jf, self.PAruny_Jf, self.PArunz_Jf, self.QArunx_Jf, self.QAruny_Jf, self.QArunz_Jf = jax.lax.cond(jnp.isin(convention, jnp.array([0, 5])), self.PQ_Arun_0_5, self.PQ_Arun_1_6_7, operand = None)

        #As it is line 1035-1043
        #(X . P)
        self.XdotPArun = (tmp_x * self.PArunx_Jf) + (tmp_y * self.PAruny_Jf) + (tmp_z * self.PArunz_Jf)

        #(X . Q)
        self.XdotQArun = (tmp_x * self.QArunx_Jf) + (tmp_y * self.QAruny_Jf) + (tmp_z * self.QArunz_Jf)

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
        self.epsilon0 = self.set_epsilon0(convention)

        ## Compression line 1178-1202
        cond = (convention == 5) | (convention==7)
        self.alpha_offset, self.epsilon_offset, self.alpha_offset_1, self.epsilon_offset_1, self.alpha_offset_3, self.epsilon_offset_3, self.alpha_offset_4, self.epsilon_offset_4 =  jax.lax.cond(cond, self.convention_five_or_seven_true, self.convention_five_or_seven_false)

        self.cexp_i_alpha   = 0.
        self.cexp_i_epsilon = 0.
        self.cexp_i_betah   = 0.

        # When L + SL < 0 and q>7, we disable multibanding NH: I will skip this function
        #self.IMRPhenomXPCheckMaxOpeningAngle()

        # Activate multibanding for Euler angles it threshold !=0. Only for PhenomXPHM. */
        self.MBandPrecVersion = jax.lax.cond(self.lalParams['PhenomXPHMThresholdMband']==0, lambda _: 0, lambda _: 1)
        ## NH: I do not implement PhenomXPHMThresholdMband==1 option. The output of the above line will always be self.MBandPrecVersion = 0. 


        # At high mass ratios, we find there can be numerical instabilities in the model, although the waveforms continue to be well behaved.
        # We warn to user of the possibility of these instabilities.
        # printf(pWF->q)
        jax.lax.cond(pWF["q"] > 80, 
                     lambda _: jax.debug.print("Very high mass ratio, possibility of numerical instabilities. Waveforms remain well behaved."), 
                     lambda _: None, 
                     None)


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

        self.LALparams = lalParams
        ## End line 1307

    def convention_five_or_seven_true(self):
        return -self.alpha0, 0, -self.alpha0, 0, -self.alpha0, 0, -self.alpha0, 0
    
    def convention_five_or_seven_false(self):
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
        vangles  = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(self, v) # FIXME

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
    
    def set_epsilon0(self, convention):

        epsilon0 = jax.lax.cond(
            jnp.isin(convention, jnp.array([1, 6])),
            lambda _: self.phiJ_Sf - jnp.pi,
            lambda _: 0.0,
            operand=None,
        )
        
        return epsilon0
    

    def compute_alpha_epsilon_101_104(self):
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
                   - (35*self.LAL_PI)/48. + (5*self.delta*self.LAL_PI)/(16.*self.m1))
        
        alpha5 = (5*(-190512*self.delta3*self.eta6 + 2268*self.delta2*self.eta3*self.m1*(self.eta2*(323 + 784*self.eta) + 336*(25*chiL2 + self.chip2)*self.m1_4)
                + 7*self.m1_3*(8024297*self.eta4 + 857412*self.eta5 + 3080448*self.eta6
                               + 143640*self.chip2*self.eta2*self.m1_4 - 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                               + 6048*self.eta3*((2632*chiL2 + 115*self.chip2)*self.m1_4 - 672*chiL*self.m1_2*self.LAL_PI))
                + 3*self.delta*self.m1_2*(-5579177*self.eta4 + 80136*self.eta5 - 3845520*self.eta6
                                           + 146664*self.chip2*self.eta2*self.m1_4 + 127008*self.chip2*(-4*chiL2 + self.chip2)*self.m1_8
                                           - 42336*self.eta3*((726*chiL2 + 29*self.chip2)*self.m1_4 - 96*chiL*self.m1_2*self.LAL_PI)))) / (6.5028096e7*self.eta4*self.m1_3)


        epsilon1 = -35/192. + (5*self.delta)/(64.*self.m1)
        epsilon2 = ((15*chiL*self.delta*self.m1)/128. - (35*chiL*self.m1_2)/128.) / self.eta
        epsilon3 = -5515/3072. + self.eta*(-515/384. - (15*self.delta2)/(256.*self.m1_2) + (175*self.delta)/(256.*self.m1)) + (4555*self.delta)/(7168.*self.m1)
        epsilon4L = (5*chiL*self.delta2)/16. - (5*chiL*self.delta*self.m1)/3. + (2545*chiL*self.m1_2)/1152. \
                    + ((-2035*chiL*self.delta*self.m1)/21504. + (2995*chiL*self.m1_2)/9216.) / self.eta \
                    - (35*self.LAL_PI)/48. + (5*self.delta*self.LAL_PI)/(16.*self.m1)
        epsilon5 = (5*(-190512*self.delta3*self.eta3 + 2268*self.delta2*self.m1*(self.eta2*(323 + 784*self.eta) + 8400*chiL2*self.m1_4)
                        - 3*self.delta*self.m1_2*(self.eta*(5579177 + 504*self.eta*(-159 + 7630*self.eta)) + 254016*chiL*self.m1_2*(121*chiL*self.m1_2 - 16*self.LAL_PI))
                        + 7*self.m1_3*(self.eta*(8024297 + 36*self.eta*(23817 + 85568*self.eta)) + 338688*chiL*self.m1_2*(47*chiL*self.m1_2 - 12*self.LAL_PI)))) / (6.5028096e7*self.eta*self.m1_3)

        return alpha1, alpha2, alpha3, alpha4L, alpha5, epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5
    
    def compute_alpha_epsilon_220_330(self):
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  
    

    def PQ_Arun_0_5(self):
        #Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = 0.0
        PAruny_Jf = -1.0
        PArunz_Jf = 0.0

        #Q = (N x P) by construction
        QArunx_Jf = self.Nz_Jf
        QAruny_Jf = 0.0
        QArunz_Jf = -self.Nx_Jf

        return PArunx_Jf, PAruny_Jf, PArunz_Jf, QArunx_Jf, QAruny_Jf, QArunz_Jf

    def PQ_Arun_1_6_7(self):
        # Get polar angle of X vector in J frame in the P,Q basis of Arun et al
        PArunx_Jf = self.Nz_Jf
        PAruny_Jf = 0.0
        PArunz_Jf = -self.Nx_Jf

        QArunx_Jf = 0.0
        QAruny_Jf = 1.0
        QArunz_Jf = 0.0

        return PArunx_Jf, PAruny_Jf, PArunz_Jf, QArunx_Jf, QAruny_Jf, QArunz_Jf
    

    def thetaJN_Nz_Nx_0_5(self):
        # Line 937-952
        #Now determine thetaJN by rotating N
        tmp_v = jnp.array([self.Nx_Sf, self.Ny_Sf, self.Nz_Sf])
        tmp_v = IMRPhenomX_rotate_z(self.phiJ_Sf,   tmp_v)
        tmp_v = IMRPhenomX_rotate_y(self.thetaJ_Sf, tmp_v)
        tmp_v = IMRPhenomX_rotate_z(self.kappa,     tmp_v)

        # We don't need the y-component but we will store it anyway

        # This is a unit vector, so no normalization
        thetaJN = jnp.acos(self.Nz_Jf)

        return thetaJN, tmp_v[2], tmp_v[0]

    def thetaJN_Nz_Nx_1_6_7(self):
        # Line 957-962
        J0dotN     = (self.J0x_Sf * self.Nx_Sf) + (self.J0y_Sf * self.Ny_Sf) + (self.J0z_Sf * self.Nz_Sf)
        thetaJN = jnp.acos( J0dotN / self.J0 )
        Nz_Jf     = jnp.cos(thetaJN)
        Nx_Jf     = jnp.sin(thetaJN)
        return thetaJN, Nz_Jf, Nx_Jf

    def precversionTag3(self)->None:

        #check mode array to estimate frequency range over which splines will need to be evaluated
        self.ModeArray = self.lalParams['ModeArray']
        
        self.L_MAX_PNR = jnp.max(self.lalParams['ModeArray']) # line 305 to 316
        IMRPhenomX_GetandSetModes(self.ModeArray, None) #FIXME #None should be pPrec


        self.pWF['deltaMF'] = jnp.where(self.pWF['deltaF']==0.0, get_deltaF_from_wfstruct(self.pWF), self.pWF['deltaMF'])

        # // if PNR angles are disabled, step back accordingly to the waveform's frequency grid step
        flow = self.compute_flow() #substitute for line 323 to 404

        assert flow>0. # line 405

        self.PNarrays, self.fmin_integration = IMRPhenomX_InspiralAngles_SpinTaylor(self.chi1x, self.chi1y, self.chi1z, self.chi2x, self.chi2y, self.chi2z, flow, self.IMRPhenomXPrecVersion, self.pWF, self.lalParams)        

        self.Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(self.fmin_integration, self.pWF['Mtot'])


        if self.IMRPhenomXPrecVersion==330:
            # do something
            pass

        if "this fails":
            self.IMRPhenomXPrecVersion==223

        
        #// if PN numerical integration fails, default to MSA+fallback to NNLO
        #// end of SpinTaylor code
        return None
        

    def compute_flow(self)->float:
        """
        Substitute function for line 324-340 in LALSimIMRPhenomX_precession.c script
        """
        
        def PNRTuned_true(_):
            return jnp.where(self.pWF['deltaF']==0.0, self.pWF['fMin'], jnp.floor_divide(self.pWF['fMin'], self.pWF['deltaF'])*self.pWF['deltaF'])
        
        def PNRTuned_false(_):
            self.M_MAX = 1.0 #FIXME this is definietly not the right value
            self.integration_buffer = jnp.where(self.pWF['deltaF']>0.0, 3*self.pWF['deltaF'], 0.5) 
            return (self.pWF['fMin'] - self.integration_buffer)*2 / self.M_MAX
        
        return jax.lax.cond(self.PNRUseTunedAngles, PNRTuned_true, PNRTuned_false, operand = None)







def IMRPhenomXPCheckMaxOpeningAngle(self):

    #    Helper function to check if maximum opening angle > pi/2 or pi/4 and issues a warning. See discussion in https://dcc.ligo.org/LIGO-T1500602

    '''
    if L + SL < 0 & chi_p>0: print error
        if q>7: turn off multibanding
    elif: max_beta > pi/4
        print('Pathological waveform')
    '''

    # This is purely sanity check to disable multibanding. Skipping this funcation for now. 

    return None




def get_deltaF_from_wfstruct(pWF: dict):
    """
    To be tested the jnp functions
    """
    #seglen=XLALSimInspiralChirpTimeBound(pWF['fRef'], pWF['m1_SI'], pWF['m2_SI'], pWF['chi1L'],pWF['chi2L'])
    #deltaFv1= 1./jnp.max(4.,jnp.pow(2, jnp.ceil(jnp.log(seglen)/jnp.log(2))))
    #deltaF = jnp.min(deltaFv1,0.1)
    #deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF,pWF['Mtot'])
    deltaMF = None
    return deltaMF


def XLALSimIMRPhenomXUtilsHztoMf():
    return None

def XLALSimInspiralChirpTimeBound():


    return None


def IMRPhenomX_InspiralAngles_SpinTaylor(chi1x: float, chi1y: float, chi1z: float, 
                                         chi2x: float, chi2y: float, chi2z: float,
                                         fmin: float, PrecVersion: int, pWF: dict, lalParams: dict):
    
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

    phaseO = jnp.where(lalParams['phaseO']==-1, 7, lalParams['phaseO'])
    spinO = jnp.where(lalParams['spinO']==-1, 6, lalParams['spinO'])
    tideO = jnp.where(lalParams['tideO']==-1, 12, lalParams['tideO'])

    lnhatx = 0.0
    lnhaty = 0.0
    e1y = 0.0
    e1z = 0.0
    lnhatz = 1.0
    e1x = 1.0
    lscorr = 0.0


    approx = lalParams['approx_name']

    fMECO_Hz = XLALSimIMRPhenomXUtilsMftoHz(pWF['fMECO'], pWF['Mtot'])

    fmin = jax.lax.select((fmin > fMECO_Hz) & ((PrecVersion==320) | (PrecVersion==321)), fMECO_Hz, fmin)

    fCut = XLALSimIMRPhenomXUtilsMftoHz(pWF['fRING']+8 * pWF['fDAMP'], pWF['Mtot'])

    deltaT_coarse = .5 * lalParams['coarse_fac'] / fCut
    fS = fmin
    fE = fCut


    PNEvolveOrbit_operands = [fRef, fmin, deltaT_coarse, m1_SI, m2_SI, fS, fE, s1x, s1y, s1z, s2x, s2y, s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx]

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = jax.lax.cond(True, fRef_equal_to_fmin, fRef_greater_than_fmin, operands = PNEvolveOrbit_operands)

    if lalParams['coarse_fac'] > 1:
        lenLow = len(V)
        nbuffer = min(9, lenLow-1)

        if (lenLow-1-nbuffer<0):
            nbuffer = lenLow-1-nbuffer

        vtrans = V[lenLow-1-nbuffer]
        ftrans = pow(vtrans, 3)/piGM

        LNhatx_trans=LNhatx[lenLow-1-nbuffer]
        LNhaty_trans=LNhaty[lenLow-1-nbuffer]
        LNhatz_trans=LNhatz[lenLow-1-nbuffer]

        E1x_trans = E1x[lenLow-1-nbuffer]
        E1y_trans = E1y[lenLow-1-nbuffer]
        E1z_trans = E1z[lenLow-1-nbuffer]

        S1x_trans = S1x[lenLow-1-nbuffer]
        S1y_trans = S1y[lenLow-1-nbuffer]
        S1z_trans = S1z[lenLow-1-nbuffer]

        S2x_trans = S2x[lenLow-1-nbuffer]
        S2y_trans = S2y[lenLow-1-nbuffer]
        S2z_trans = S2z[lenLow-1-nbuffer]
                            
        fS=ftrans
        fE=fCut
        deltaT = 0.5/(fCut)


        V_PN, Phi_PN, S1x_PN, S1y_PN, S1z_PN, S2x_PN, S2y_PN, S2z_PN, LNhatx_PN, LNhaty_PN, LNhatz_PN, E1x_PN, E1y_PN, E1z_PN, =XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT, m1_SI, m2_SI,fS,fE,S1x_trans,S1y_trans,S1z_trans,S2x_trans,S2y_trans,S2z_trans,LNhatx_trans,LNhaty_trans,LNhatz_trans,E1x_trans, E1y_trans, E1z_trans,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)

        lenPN=lenLow-nbuffer-1+len(V_PN)

        #if(lenPN < 4):
        #    XLALPrintError("Error in %s: no. of points is insufficient for spline interpolation",__func__)
        #    XLAL_ERROR(XLAL_EFUNC)
                    

        V_PN = V_PN[-(lenLow-nbuffer-1):lenPN]
        LNhatx = LNhatx[-(lenLow-nbuffer-1):lenPN]
        LNhaty_PN = LNhaty_PN[-(lenLow-nbuffer-1):lenPN]
        LNhatz_PN = LNhatz_PN[-(lenLow-nbuffer-1):lenPN]
        S1x_PN = S1x_PN[-(lenLow-nbuffer-1):lenPN]
        S1y_PN = S1y_PN[-(lenLow-nbuffer-1):lenPN]
        S1z_PN = S1z_PN[-(lenLow-nbuffer-1):lenPN]

        S2x_PN = S2x_PN[-(lenLow-nbuffer-1):lenPN]
        S2y_PN = S2y_PN[-(lenLow-nbuffer-1):lenPN]
        S2z_PN = S2z_PN[-(lenLow-nbuffer-1):lenPN]

    else:
        copyLength=len(V)-1
        #if(copyLength < 4) {
        #XLALPrintError("Error in %s: no. of points is insufficient for spline interpolation",__func__)
        #XLAL_ERROR(XLAL_EFUNC)
        ## Just create these arrays..
    

    ## copy coarse-grid data to fine-grid
    ## destroy coarse-grid

    fminPN=jnp.power(V_PN[0],3.)/piGM
    if (fminPN<0.) | (fminPN>fmin): 
        return "Failure"

    PhenomXPInspiralArrays = None
    return PhenomXPInspiralArrays, 0


def fRef_equal_to_fmin(fRef, fmin, deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z

def fRef_greater_than_fmin(fRef, fmin, deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    fS =  fRef
    fE = fmin - 0.5

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)

    if len(V['data']) > 1:
        V2, Phi2, S1x2, S1y2, S1z2, S2x2, S2y2, S2z2, LNhatx2, LNhaty2, LNhatz2, E1x2, E1y2, E1z2 = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, 
                                                                                                                                            m1_SI, m2_SI, fS, fE, s1x, s1y, s1z, s2x, s2y,
                                                                                                                                            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1,lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx)
        V = jnp.append(V, V2)
        Phi = jnp.append(Phi, Phi2)
        S1x = jnp.append(S1x, S1x2)
        S1y = jnp.append(S1y, S1y2)
        S1z = jnp.append(S1z, S1z2)

        S2x = jnp.append(S2x, S2x2)
        S2y = jnp.append(S2y, S2y2)
        S2z = jnp.append(S2z, S2z2)

        LNhatx = jnp.append(LNhatx, LNhatx2)
        LNhaty = jnp.appnd(LNhaty, LNhaty2)
        LNhatz = jnp.append(LNhatz, LNhatz2)
        
        E1x = jnp.append(E1x, E1x2)
        E1y = jnp.append(E1y, E1y2)
        E1z = jnp.append(E1z, E1z2)

    else:
        # This means the generation failed.
        V = jnp.array([0])
        Phi = jnp.array([0])
        S1x = jnp.array([0])
        S1y = jnp.array([0])
        S1z = jnp.array([0])

        S2x = jnp.array([0])
        S2y = jnp.array([0])
        S2z = jnp.array([0])

        LNhatx =jnp.array([0])
        LNhaty = jnp.array([0])
        LNhatz = jnp.array([0])
        
        E1x = jnp.array([0])
        E1y = jnp.array([0])
        E1z = jnp.array([0])


    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z


def XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT: float, m1_SI: float, m2_SI: float, fStart: float, fEnd: float,
                                           s1x: float, s1y: float, s1z: float, s2x: float, s2y: float, s2z: float, 
                                           lnhatx: float, lnhaty: float, lnhatz: float, e1x: float, e1y: float, e1z: float,
                                           lambda1: float, lambda2: float, quadparam1: float, quadparam2: float, spin0: int,
                                           tide0: int, phase0: float, lscorr: int, approx: str):
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral_spin_taylor__c.html#ga35cfdf3082e09cc97cda9e11ba4c2bff

    """
    spin0 >= 7 is not allowed.
    fStart < 0 is not allowed. 
    fEnd < 0 is not allowed.
    if fEnd<fStart && fEnd != 0.0, sgn = -1 else sign 1.
    
    """

    if approx=='SpinTaylorT4':
        pass
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


    return None


def XLALSimIMRPhenomXLPNAnsatz():
    return None


def IMRPhenomX_SetPrecessingRemnantParams():
    return None


def IMRPhenomX_PNR_GetAndSetPNRVariables():
    return None



def IMRPhenomX_PNR_GetAndSetCoPrecParams():
    return None

def XLALSimIMRPhenomXUtilsMftoHz():
    return None



def IMRPhenomX_Initialize_MSA_System():
    return None

def IMRPhenomX_GetandSetModes(ModeArray: list, IMRPhenomXPrecessionStruct: dict):
    return None


def IMRPhenomX_rotate_z(angle, vx, vy, vz): 
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

    vx_rot = vx * cosa - vy * sina
    vy_rot = vx * sina + vy * cosa
    vz_rot = vz  # unchanged

    return jnp.array([vx_rot, vy_rot, vz_rot])


def IMRPhenomX_rotate_y(angle, vx, vy, vz):
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

    vx_rot =  vx * cosa + vz * sina
    vy_rot =  vy  # unchanged
    vz_rot = -vx * sina + vz * cosa

    return jnp.array([vx_rot, vy_rot, vz_rot])
