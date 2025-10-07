import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C
import jax

class IMRPhenomXGetAndSetPrecessionVariables:

    def __init__(self, pWF: dict, m1_SI: float, m2_SI: float, chi1x: float, chi1y: float, chi1z: float, chi2x: float, chi2y: float, chi2z: float, lalParams: dict, debug_flag: bool):
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
        #self.pWF['LALparams'] = lalParams
        self.lalParams = lalParams
        self.debug_prec = debug_flag

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

        #### Skipping the multibanding bookkeeping

        ########## Define a number of convenient local parameters #############
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

        q = m1/m2


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


        self.PNRUseTunedAngles, lalParams['PNRUseTunedAngles'], self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']  = jnp.where((chi_in_plane<1e-7) & (self.PNRUseTunedAngles == 1), jnp.array([False, False, False, False, False]), jnp.array([self.PNRUseTunedAngles, lalParams['PNRUseTunedAngles'], self.AntisymmetricWaveform, lalParams['AntisymmetricWaveform'], lalParams['PNRUseTunedCoprec']]))
            


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

        #/* start of SpinTaylor code */



        #####################################
        precversionTag_3_true = None
        precversionTag_3_true = False
        self.precversionTag3()


            
            ### up to line 405
            

    def precversionTag3(self)->None:

        self.L_MAX_PNR = jnp.max(self.lalParams['ModeArray'])
        self.ModeArray = self.lalParams['ModeArray']

        #self.pWF['deltaMF'] = get_deltaF_from_wfstruct(self.pWF) #FIXME
        flow = self.compute_flow()

        assert flow>0.

        self.PNarrays, self.fmin_integration = IMRPhenomX_InspiralAngles_SpinTaylor(self.chi1x, self.chi1y, self.chi1z, self.chi2x, self.chi2y, self.chi2z, flow, self.IMRPhenomXPrecVersion, self.pWF, self.lalParams)        

        self.Mfmin_integration = XLALSimIMRPhenomXUtilsHztoMf(self.fmin_integration, self.pWF['Mtot'])

        
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
        #XLALPrintError("Error in %s: no. of points is insufficient for spline interpolation",__func__);
        #XLAL_ERROR(XLAL_EFUNC);
        ## Just create these arrays..
    

    ## copy coarse-grid data to fine-grid
    ## destroy coarse-grid
    ##

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


def XLALSimInspiralSpinTaylorPNEvolveOrbit():
    return None












def XLALSimIMRPhenomXUtilsMftoHz():
    return None