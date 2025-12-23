import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, Event
from jax import jit, lax
from . import LALSimInspiralPNCoefficients as pncoefficients

from functools import partial

# Constants (from LAL)
LAL_MSUN_SI = 1.988409870698050731911960804878414216e30
LAL_MTSUN_SI = 4.925490947641266978197229498498379006e-6

LAL_G_SI = 6.67430e-11
LAL_C_SI = 299792458.0
LAL_GAMMA = 0.5772156649015329

LAL_ST4_ABSOLUTE_TOLERANCE = 1.0e-14
LAL_ST4_RELATIVE_TOLERANCE = 1.0e-14
LAL_NUM_ST4_VARIABLES = 14

LAL_REAL4_EPS = jnp.float32(2.0 ** -23)


@dataclass
class REAL8TimeSeries:
    """JAX equivalent of LAL REAL8TimeSeries"""
    data: jax.Array
    deltaT: float
    epoch: float = 0.0

def normsq(x, y, z):
    """Compute squared norm of 3-vector"""
    return x*x + y*y + z*z

def cdot(ax, ay, az, bx, by, bz):
    """Vector dot product stub."""
    return ax*bx + ay*by + az*bz


def omegashift(S1sq, S2sq, S1S2, LNhS1, LNhS2, OmS1, OmS2):
    """Stub for omegashift (replace with your physics formula)."""
    # Just return zero for now
    return -0.25*(OmS1*OmS1*(S1sq-LNhS1*LNhS1)+OmS2*OmS2*(S2sq-LNhS2*LNhS2)+2.*OmS1*OmS2*(S1S2-LNhS1*LNhS2))

def cross_vec(a, b):
    """Cross product of 3-vectors a and b. a,b shape=(3,)"""
    return jnp.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def _get(p, name, default=0.0):
    """Safe getter from params PyTree (works if params is a simple object or dict)."""
    return getattr(p, name, p.get(name, default) if isinstance(p, dict) else default)


@dataclass(frozen=True)
class XLALSimInspiralSpinTaylorTxCoeffs:
    """Parameters for SpinTaylor evolution (replaces C struct)"""
    m1_SI: float
    m2_SI: float
    fStart: float
    fEnd: float
    lambda1: float
    lambda2: float
    quadparam1: float
    quadparam2: float
    spinO: int
    tideO: int
    phaseO: int
    lscorr: int
    # Derived quantities
    m1sec: float
    m2sec: float
    Msec: float
    Mcsec: float
    eta: float
    norm1: float
    norm2: float
    wdotcoeff: dict
    intermediate_wdotspin: dict
    wdotnewt: float
    omegashiftS1: float
    omegashiftS2: float
    S1dot6S2Avg: float 
    S2dot6S1Avg: float
    S1dot6S2OAvg: float 
    S1dot6S1OAvg: float 
    S2dot6S1OAvg: float 
    S2dot6S2OAvg: float
    S1dot6QMS1OAvg: float 
    S2dot6QMS2OAvg: float
    S1dot3: float
    S2dot3: float
    S1dot4S2Avg:float
    S1dot4S2OAvg:float
    S1dot4QMS1OAvg:float
    S2dot4QMS2OAvg: float
    S1dot5:float
    S2dot5:float
    cS1:float
    cS2:float
    cS1L:float
    cS2L:float
    energy_PNTermsAvg: dict
    phenomtp: bool


    def to_dict(self):
        return {
            'm1_SI': self.m1_SI,
            'm2_SI': self.m2_SI,
            'fStart': self.fStart,
            'fEnd': self.fEnd,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'quadparam1': self.quadparam1,
            'quadparam2': self.quadparam2,
            'spinO': self.spinO,
            'tideO': self.tideO,
            'phaseO': self.phaseO,
            'lscorr': self.lscorr,
            'm1sec': self.m1sec,
            'm2sec': self.m2sec,
            'Msec': self.Msec,
            'Mcsec': self.Mcsec,
            'eta': self.eta,
            'norm1': self.norm1,
            'norm2': self.norm2,
            'wdotcoeff': self.wdotcoeff,
            'intermediate_wdotspin': self.intermediate_wdotspin,
            'wdotnewt': self.wdotnewt,
            'omegashiftS1': self.omegashiftS1,
            'omegashiftS2': self.omegashiftS2,
            'S1dot6S2Avg': self.S1dot6S2Avg,
            'S2dot6S1Avg': self.S2dot6S1Avg,
            'S1dot6S2OAvg': self.S1dot6S2OAvg,
            'S1dot6S1OAvg': self.S1dot6S1OAvg,
            'S2dot6S1OAvg': self.S2dot6S1OAvg,
            'S2dot6S2OAvg': self.S2dot6S2OAvg,
            'S1dot6QMS1OAvg': self.S1dot6QMS1OAvg,
            'S2dot6QMS2OAvg': self.S2dot6QMS2OAvg,
            'S1dot3': self.S1dot3,
            'S2dot3': self.S2dot3,
            'S1dot4S2Avg': self.S1dot4S2Avg,
            'S1dot4S2OAvg': self.S1dot4S2OAvg,
            'S1dot4QMS1OAvg': self.S1dot4QMS1OAvg,
            'S2dot4QMS2OAvg':  self.S2dot4QMS2OAvg,
            'S1dot5': self.S1dot5,
            'S2dot5': self.S2dot5,
            'cS1': self.cS1,
            'cS2': self.cS2,
            'cS1L': self.cS1L,
            'cS2L': self.cS2L,
            'energy_PNTermsAvg': self.energy_PNTermsAvg,
            'phenomtp': self.phenomtp
            
        }

def XLALSimInspiralSpinTaylorT4Setup(
    m1_SI: float,
    m2_SI: float,
    fStart: float,
    fEnd: float,
    lambda1: float,
    lambda2: float,
    quadparam1: float,
    quadparam2: float,
    spinO: int,
    tideO: int,
    phaseO: int,
    lscorr: int,
    phenomtp: bool
) -> XLALSimInspiralSpinTaylorTxCoeffs:
    """Setup parameters for SpinTaylorT4"""
    m1sec = m1_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    m2sec = m2_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    Msec = m1sec + m2sec
    eta = m1sec * m2sec / (Msec * Msec)
    Mcsec = Msec * jnp.power(eta, 0.6)
    norm1 = m1sec * m1sec / Msec / Msec
    norm2 = m2sec * m2sec / Msec / Msec
    m1=m1_SI/LAL_MSUN_SI
    m2=m2_SI/LAL_MSUN_SI
    M = m1+m2
    m1M = m1/M
    m2M = m2/M

    wdotcoeff = {}

    wdotnewt = 96. / 5. * eta
    wdotcoeff[0] = 1
    wdotcoeff[1] = 0
    wdotcoeff[2] =  pncoefficients.XLALSimInspiralTaylorT4wdot_2PNCoeff(eta)
    wdotcoeff[3] = pncoefficients.XLALSimInspiralTaylorT4wdot_3PNCoeff(eta)
    wdotcoeff[4] = pncoefficients.XLALSimInspiralTaylorT4wdot_4PNCoeff(eta)
    wdotcoeff[5] = pncoefficients.XLALSimInspiralTaylorT4wdot_5PNCoeff(eta)
    wdotcoeff[6] = pncoefficients.XLALSimInspiralTaylorT4wdot_6PNCoeff(eta)
    wdotcoeff[6_1]  = pncoefficients.XLALSimInspiralTaylorT4wdot_6PNLogCoeff(eta) # logcoeff

    wdotcoeff[7] = pncoefficients.XLALSimInspiralTaylorT4wdot_7PNCoeff(eta)
    wdotcoeff[10_1] = lambda1 * pncoefficients.XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(m1M) + lambda2 * pncoefficients.XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(m2M)
    wdotcoeff[12_1] = lambda1 * pncoefficients.XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(m1M) + lambda2 * pncoefficients.XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(m2M)

    intermediate_wdotspin = pncoefficients.add_wdot_spin_coefficients_to_setup(m1M, m2M, eta, quadparam1, quadparam2)
    omegashiftS1 = pncoefficients.XLALSimInspiralLDot_3PNSOCoeff(m1M)
    omegashiftS2 = pncoefficients.XLALSimInspiralLDot_3PNSOCoeff(m2M)

    S1dot6S2Avg = pncoefficients.XLALSimInspiralSpinDot_6PNS2CoeffAvg(m1M)
    S2dot6S1Avg = pncoefficients.XLALSimInspiralSpinDot_6PNS2CoeffAvg(m2M)

    S1dot6S2OAvg = pncoefficients.XLALSimInspiralSpinDot_6PNS2OCoeffAvg(m1M)
    S1dot6S1OAvg = pncoefficients.XLALSimInspiralSpinDot_6PNS1OCoeffAvg(m1M)


    S2dot6S1OAvg = pncoefficients.XLALSimInspiralSpinDot_6PNS2OCoeffAvg(m2M)
    S2dot6S2OAvg = pncoefficients.XLALSimInspiralSpinDot_6PNS1OCoeffAvg(m2M)
    
    S1dot6QMS1OAvg =  quadparam1 * pncoefficients.XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(m1M)
    S2dot6QMS2OAvg = quadparam2 * pncoefficients.XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(m2M)

    S1dot3 = pncoefficients.XLALSimInspiralSpinDot_3PNCoeff(m1M)
    S2dot3 = pncoefficients.XLALSimInspiralSpinDot_3PNCoeff(m2M)

    
    S1dot4S2Avg = pncoefficients.XLALSimInspiralSpinDot_4PNS2CoeffAvg
    S1dot4S2OAvg = pncoefficients.XLALSimInspiralSpinDot_4PNS2OCoeffAvg
    S1dot4QMS1OAvg = quadparam1 * pncoefficients.XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(m1M)
    S2dot4QMS2OAvg = quadparam2 * pncoefficients.XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(m2M)

    S1dot5 =  pncoefficients.XLALSimInspiralSpinDot_5PNCoeff(m1M)
    S2dot5 = pncoefficients.XLALSimInspiralSpinDot_5PNCoeff(m2M)

    cS1 = pncoefficients.XLALSimInspiralL_3PNSicoeffAvg(m1M)
    cS2 = pncoefficients.XLALSimInspiralL_3PNSiLcoeffAvg(m1M)
    cS1L = pncoefficients.XLALSimInspiralL_3PNSicoeffAvg(m2M)
    cS2L = pncoefficients.XLALSimInspiralL_3PNSiLcoeffAvg(m2M)

    energy_PNTermsAvg = compute_XLALSimInspiralSetEnergyPNTermsAvg(m1M, m2M, eta, lambda1, lambda2, quadparam1, quadparam2)

    return XLALSimInspiralSpinTaylorTxCoeffs(
        m1_SI=m1_SI, m2_SI=m2_SI, fStart=fStart, fEnd=fEnd,
        lambda1=lambda1, lambda2=lambda2,
        quadparam1=quadparam1, quadparam2=quadparam2,
        spinO=spinO, tideO=tideO, phaseO=phaseO, lscorr=lscorr,
        m1sec=m1sec, m2sec=m2sec, Msec=Msec, Mcsec=Mcsec,
        eta=eta, norm1=norm1, norm2=norm2, wdotcoeff = wdotcoeff, wdotnewt = wdotnewt, 
        intermediate_wdotspin = intermediate_wdotspin,
        omegashiftS1 = omegashiftS1, omegashiftS2 = omegashiftS2,
        S1dot6S2Avg = S1dot6S2Avg,
        S2dot6S1Avg = S2dot6S1Avg,
        S1dot6S2OAvg = S1dot6S2OAvg,
        S1dot6S1OAvg = S1dot6S1OAvg,
        S2dot6S1OAvg = S2dot6S1OAvg,
        S2dot6S2OAvg = S2dot6S2OAvg,
        S1dot6QMS1OAvg = S1dot6QMS1OAvg,
        S2dot6QMS2OAvg = S2dot6QMS2OAvg,
        S1dot3 = S1dot3,
        S2dot3 = S2dot3,
        S1dot4S2Avg = S1dot4S2Avg,
        S1dot4S2OAvg = S1dot4S2OAvg,
        S1dot4QMS1OAvg = S1dot4QMS1OAvg,
        S2dot4QMS2OAvg =  S2dot4QMS2OAvg,
        S1dot5 = S1dot5,
        S2dot5 = S2dot5,
        cS1 = cS1,
        cS2 = cS2,
        cS1L = cS1L,
        cS2L = cS2L,
        energy_PNTermsAvg = energy_PNTermsAvg,
        phenomtp = phenomtp  
    )


def XLALSimInspiralSpinTaylorT4DerivativesAvg(t, values, params):
    """
    JAX translation of XLALSimInspiralSpinTaylorT4DerivativesAvg
    with stub helper functions.

    """

    omega = values[1]
    LNhx, LNhy, LNhz = values[2:5]
    S1x, S1y, S1z     = values[5:8]
    S2x, S2y, S2z     = values[8:11]
    E1x, E1y, E1z     = values[11:14]

    # Guard for omega ≤ 0
    # In JAX, we cannot "return" early, we must use jnp.where
    omega = jnp.where(omega <= 0, 1e-20, omega)


    # Basic quantities
    v = jnp.cbrt(omega)
    v2 = v * v
    v11 = omega * omega * omega * v2

    # Dot products
    LNhdotS1 = cdot(LNhx, LNhy, LNhz, S1x, S1y, S1z)
    LNhdotS2 = cdot(LNhx, LNhy, LNhz, S2x, S2y, S2z)
    S1dotS2  = cdot(S1x, S1y, S1z, S2x, S2y, S2z)
    S1sq     = cdot(S1x, S1y, S1z, S1x, S1y, S1z)
    S2sq     = cdot(S2x, S2y, S2z, S2x, S2y, S2z)
    

    # -----------------------------------------------------
    # Spin contributions (stubbed as zero)
    # You can later insert the switch-case logic here.
    # -----------------------------------------------------
    wspin_dict = pncoefficients.compute_wdotspin(params['intermediate_wdotspin'], LNhdotS1, LNhdotS2, S1dotS2, S1sq, S2sq, params['spinO'])

    wspin3 = wspin_dict['wspin3']

    wspin4Avg = wspin_dict['wspin4Avg']
    wspin5 = wspin_dict['wspin5']

    wspin6Avg = wspin_dict['wspin6Avg']

    # -----------------------------------------------------
    # domega
    # -----------------------------------------------------
    domega = params['wdotnewt'] * v11 * (
        params['wdotcoeff'][0]
        + v * (
            params['wdotcoeff'][1]
            + v * (
                params['wdotcoeff'][2]
                + v * (
                    params['wdotcoeff'][3] + wspin3
                    + v * (
                        params['wdotcoeff'][4] + wspin4Avg
                        + v * (
                            params['wdotcoeff'][5] + wspin5
                            + v * (
                                params['wdotcoeff'][6] + wspin6Avg
                                + params['wdotcoeff'][6_1] * jnp.log(v)
                                + v * (
                                    params['wdotcoeff'][7]
                                    + omega * (
                                        params['wdotcoeff'][10_1]
                                        + v2 * params['wdotcoeff'][12_1]
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    # -----------------------------------------------------
    # Spin derivatives (currently stub)
    # -----------------------------------------------------
    (
        dLNhx, dLNhy, dLNhz,
        dE1x, dE1y, dE1z,
        dS1x, dS1y, dS1z,
        dS2x, dS2y, dS2z
    ) = XLALSimInspiralSpinDerivativesAvg(
        v, LNhx, LNhy, LNhz, E1x, E1y, E1z,
        S1x, S1y, S1z, S2x, S2y, S2z,
        LNhdotS1, LNhdotS2, params
    )

    # -----------------------------------------------------
    # dphi
    # -----------------------------------------------------
    shift = omegashift(S1sq, S2sq, S1dotS2,
                       LNhdotS1, LNhdotS2,
                       params['omegashiftS1'],
                       params['omegashiftS2'])

    dphi = omega * (1 + omega*omega*shift)

    return jnp.array([
        dphi, domega,
        dLNhx, dLNhy, dLNhz,
        dS1x,  dS1y,  dS1z,
        dS2x,  dS2y,  dS2z,
        dE1x,  dE1y,  dE1z
    ])


def XLALSimInspiralSpinTaylorStoppingTest(t, y, dvalues, params)->bool:
    """
    Stopping test for integration.
    
    Returns positive value to continue, negative to stop.
    
    Args:
        t: current time
        y: current state values
        dvalues: current derivatives
        params: parameter dictionary/object
        
    Returns:
        Positive = continue integration
        Negative = stop integration
    """
    # Extract values from state vector
    omega = y[1]
    v = jnp.cbrt(omega)
    LNhx, LNhy, LNhz = y[2], y[3], y[4]
    S1x, S1y, S1z = y[5], y[6], y[7]
    S2x, S2y, S2z = y[8], y[9], y[10]
    
    # Compute dot products
    LNhdotS1 = cdot(LNhx, LNhy, LNhz, S1x, S1y, S1z)
    LNhdotS2 = cdot(LNhx, LNhy, LNhz, S2x, S2y, S2z)
    S1sq = normsq(S1x, S1y, S1z)
    S2sq = normsq(S2x, S2y, S2z)
    S1dotS2 = cdot(S1x, S1y, S1z, S2x, S2y, S2z)
    
    # Get parameters
    M = _get(params, 'M', _get(params, 'Msec', 0.0) / LAL_MTSUN_SI)
    fStart = _get(params, 'fStart', 0.0)
    fEnd = _get(params, 'fEnd', 0.0)
    
    # omega = PI G M f_GW / c^3
    omegaStart = jnp.pi * M * LAL_MTSUN_SI * fStart
    omegaEnd = jnp.pi * M * LAL_MTSUN_SI * fEnd
    
    # Get spin corrections to energy
    Espin3, Espin4, Espin5, Espin6, Espin7 = XLALSimInspiralSetEnergyPNTermsAvg(
        params, LNhdotS1, LNhdotS2, S1sq, S2sq, S1dotS2
    )
    
    # Get energy coefficients
    #print(params['energy_PNTermsAvg'])
    _energy_coeffs = _get(params, 'energy_PNTermsAvg', {})
    Ecoeff = _get(_energy_coeffs, 'Ecoeff', jnp.zeros(8))
    '''Ecoeff = jnp.array([
        energy_coeffs.get(0, 1.0),
        energy_coeffs.get(1, 0.0),
        energy_coeffs.get(2, 0.0),
        energy_coeffs.get(3, 0.0),
        energy_coeffs.get(4, 0.0),
        energy_coeffs.get(5, 0.0),
        energy_coeffs.get(6, 0.0),
        energy_coeffs.get(7, 0.0)
    ])
    '''
    

    Etidal10 = _get(_energy_coeffs, 'Etidal10', 0.0)
    Etidal12 = _get(_energy_coeffs, 'Etidal12', 0.0)
    
    # Energy test: dE/domega
    # If test < 0, energy is increasing (unphysical)
    v2 = v * v
    test = 2.0 + v2 * (
        4.0 * Ecoeff[2]
        + v * (
            5.0 * (Ecoeff[3] + Espin3)
            + v * (
                6.0 * (Ecoeff[4] + Espin4)
                + v * (
                    7.0 * (Ecoeff[5] + Espin5)
                    + v * (
                        8.0 * (Ecoeff[6] + Espin6)
                        + v * (
                            9.0 * (Ecoeff[7] + Espin7)
                            + v * v * v * (
                                12.0 * Etidal10
                                + v2 * (14.0 * Etidal12)
                            )
                        )
                    )
                )
            )
        )
    )
    

    # Check d^2omega/dt^2 > 0
    #prev_domega = _get(params, 'prev_domega', 0.0)

    ddomega = 0.0

    # Run tests (in JAX, we return a single value that encodes the result)
    # Positive = continue, negative values encode different failure modes
    
    # Test 1: frequency bound (upper)
    freq_above = (jnp.abs(omegaEnd) > LAL_REAL4_EPS) & (omegaEnd > omegaStart) & (omega > omegaEnd)
    
    # Test 2: frequency bound (lower)
    freq_below = (jnp.abs(omegaEnd) > 1e-6) & (omegaEnd < omegaStart) & (omega < omegaEnd)
    
    # Test 3: energy test fails
    energy_fail = test < 0.0
    
    # Test 4: omega is nan
    omega_nan = jnp.isnan(omega)
    
    # Test 5: v >= 1 (velocity exceeds speed of light)
    large_v = v >= 1.0
    
    # Test 6: d^2omega/dt^2 <= 0
    omegadot_fail = False#(ddomega <= 0.0) & (prev_domega != 0.0)
    
    # Combine tests: return negative if any test fails, positive otherwise
    # Using different negative values to encode which test failed
    result = jnp.where(freq_above, -1.0,
             jnp.where(freq_below, -2.0,
             jnp.where(energy_fail, -3.0,
             jnp.where(omega_nan, -4.0,
             jnp.where(large_v, -5.0,
             jnp.where(omegadot_fail, -6.0,
             1.0))))))  # Success = 1.0
    #jax.debug.print("JAX Time {} Energy evolution {} Result {}", t, test, result)
    return result


def XLALSimInspiralSetEnergyPNTermsAvg(params, LNhdotS1, LNhdotS2, S1sq, S2sq, S1dotS2):
    """
    Compute spin corrections to energy at various PN orders.
    
    JAX translation of XLALSimInspiralSetEnergyPNTermsAvg.
    Returns tuple (Espin3, Espin4, Espin5, Espin6, Espin7)
    
    Args:
        params: parameter dictionary/object
        LNhdotS1: LNhat . S1
        LNhdotS2: LNhat . S2
        S1sq: S1 . S1
        S2sq: S2 . S2
        S1dotS2: S1 . S2
    """
    # Initialize all spin energy terms to zero
    Espin3 = 0.0
    Espin4 = 0.0
    Espin5 = 0.0
    Espin6 = 0.0
    Espin7 = 0.0
    
    # Get spinO parameter

    spinO = params['spinO']
    phenomtp = params['phenomtp']
    
    # Define spin order constants (matching LAL definitions)
    LAL_SIM_INSPIRAL_SPIN_ORDER_ALL = -1
    LAL_SIM_INSPIRAL_SPIN_ORDER_0PN = 0
    LAL_SIM_INSPIRAL_SPIN_ORDER_05PN = 1
    LAL_SIM_INSPIRAL_SPIN_ORDER_1PN = 2
    LAL_SIM_INSPIRAL_SPIN_ORDER_15PN = 3
    LAL_SIM_INSPIRAL_SPIN_ORDER_2PN = 4
    LAL_SIM_INSPIRAL_SPIN_ORDER_25PN = 5
    LAL_SIM_INSPIRAL_SPIN_ORDER_3PN = 6
    LAL_SIM_INSPIRAL_SPIN_ORDER_35PN = 7
    
    # Create boolean masks for each order (includes this order and higher)
    # In C, the switch-case with fallthrough includes all lower orders
    include_15PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_15PN) | (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    include_2PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_2PN) | (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    include_25PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_25PN) | (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    include_3PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_3PN) | (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    include_35PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_35PN) | (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    include_ALL = (spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL)
    
    # 1.5PN spin-orbit correction to energy
    # Computed if spinO >= 15PN (3) or spinO == ALL (-1)
    energy_params_dict = params['energy_PNTermsAvg']
    E3S1O = _get(energy_params_dict, 'E3S1O', 0.0)
    E3S2O = _get(energy_params_dict, 'E3S2O', 0.0)
    Espin3 = jnp.where(
        include_15PN,
        E3S1O * LNhdotS1 + E3S2O * LNhdotS2,
        Espin3
    )
    
    # 2PN spin-spin and quadrupole-monopole corrections
    # Computed if spinO >= 2PN (4) or spinO == ALL (-1)
    E4S1S2Avg = _get(energy_params_dict, 'E4S1S2Avg', 0.0)
    E4S1OS2OAvg = _get(energy_params_dict, 'E4S1OS2OAvg', 0.0)
    E4QMS1S1Avg = _get(energy_params_dict, 'E4QMS1S1Avg', 0.0)
    E4QMS2S2Avg = _get(energy_params_dict, 'E4QMS2S2Avg', 0.0)
    E4QMS1OS1OAvg = _get(energy_params_dict, 'E4QMS1OS1OAvg', 0.0)
    E4QMS2OS2OAvg = _get(energy_params_dict, 'E4QMS2OS2OAvg', 0.0)
    
    Espin4_SS = E4S1S2Avg * S1dotS2 + E4S1OS2OAvg * LNhdotS1 * LNhdotS2
    Espin4_QM = (E4QMS1S1Avg * S1sq + E4QMS2S2Avg * S2sq +
                 E4QMS1OS1OAvg * LNhdotS1 * LNhdotS1 +
                 E4QMS2OS2OAvg * LNhdotS2 * LNhdotS2)
    
    Espin4 = jnp.where(
        include_2PN,
        Espin4_SS + Espin4_QM,
        Espin4
    )
    
    # 2.5PN spin-orbit correction to energy
    # Computed if spinO >= 25PN (5) or spinO == ALL (-1)
    E5S1O = _get(energy_params_dict, 'E5S1O', 0.0)
    E5S2O = _get(energy_params_dict, 'E5S2O', 0.0)
    Espin5 = jnp.where(
        include_25PN,
        E5S1O * LNhdotS1 + E5S2O * LNhdotS2,
        Espin5
    )
    
    # 3PN spin-spin corrections (only if NOT phenomtp)
    # Computed if spinO >= 3PN (6) or spinO == ALL (-1)
    E6S1S2Avg = _get(energy_params_dict, 'E6S1S2Avg', 0.0)
    E6S1OS2OAvg = _get(energy_params_dict, 'E6S1OS2OAvg', 0.0)
    E6S1S1Avg = _get(energy_params_dict, 'E6S1S1Avg', 0.0)
    E6QMS1S1Avg = _get(energy_params_dict, 'E6QMS1S1Avg', 0.0)
    E6S2S2Avg = _get(energy_params_dict, 'E6S2S2Avg', 0.0)
    E6QMS2S2Avg = _get(energy_params_dict, 'E6QMS2S2Avg', 0.0)
    E6S1OS1OAvg = _get(energy_params_dict, 'E6S1OS1OAvg', 0.0)
    E6QMS1OS1OAvg = _get(energy_params_dict, 'E6QMS1OS1OAvg', 0.0)
    E6S2OS2OAvg = _get(energy_params_dict, 'E6S2OS2OAvg', 0.0)
    E6QMS2OS2OAvg = _get(energy_params_dict, 'E6QMS2OS2OAvg', 0.0)
    
    Espin6_val = (E6S1S2Avg * S1dotS2 + E6S1OS2OAvg * LNhdotS1 * LNhdotS2 +
                  (E6S1S1Avg + E6QMS1S1Avg) * S1sq +
                  (E6S2S2Avg + E6QMS2S2Avg) * S2sq +
                  (E6S1OS1OAvg + E6QMS1OS1OAvg) * LNhdotS1 * LNhdotS1 +
                  (E6S2OS2OAvg + E6QMS2OS2OAvg) * LNhdotS2 * LNhdotS2)
    
    # Only apply if include_3PN AND not phenomtp
    Espin6 = jnp.where(
        include_3PN & (~phenomtp),
        Espin6_val,
        Espin6
    )
    
    # 3.5PN (or higher for ALL) spin corrections (only if phenomtp)
    # Computed if spinO == ALL (-1) and phenomtp == True
    E7S1O = _get(energy_params_dict, 'E7S1O', 0.0)
    E7S2O = _get(energy_params_dict, 'E7S2O', 0.0)
    
    Espin7 = jnp.where(
        include_ALL & phenomtp,
        E7S1O * LNhdotS1 + E7S2O * LNhdotS2,
        Espin7
    )
    
    return Espin3, Espin4, Espin5, Espin6, Espin7



def  compute_XLALSimInspiralSetEnergyPNTermsAvg(m1M, m2M, eta, lambda1, lambda2, quadparam1, quadparam2):


    output = {}

    output['E3S1O'] = pncoefficients.XLALSimInspiralPNEnergy_3PNSOCoeff(m1M)
    output['E3S2O'] = pncoefficients.XLALSimInspiralPNEnergy_3PNSOCoeff(m2M)

    output['E4S1S2Avg'] = pncoefficients.XLALSimInspiralPNEnergy_4PNS1S2CoeffAvg(eta)
    output['E4S1OS2OAvg'] = pncoefficients.XLALSimInspiralPNEnergy_4PNS1OS2OCoeffAvg(eta)
    output['E4QMS1S1Avg'] = quadparam1 * pncoefficients.XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(m1M)
    output['E4QMS2S2Avg'] = quadparam2 * pncoefficients.XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(m2M)
    output['E4QMS1OS1OAvg'] =  quadparam1 * pncoefficients.XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(m1M)
    output['E4QMS2OS2OAvg'] =  quadparam2 * pncoefficients.XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(m2M)


    output['E5S1O'] = pncoefficients.XLALSimInspiralPNEnergy_5PNSOCoeff(m1M)
    output['E5S2O'] = pncoefficients.XLALSimInspiralPNEnergy_5PNSOCoeff(m2M)



    output['E6S1S2Avg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1S2CoeffAvg(eta)
    output['E6S1OS2OAvg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1OS2OCoeffAvg(eta)

    output['E6S1S1Avg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(m1M)
    output['E6QMS1S1Avg'] = quadparam1 * pncoefficients.XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(m1M)

    output['E6S2S2Avg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(m2M)
    output['E6QMS2S2Avg'] = quadparam2 * pncoefficients.XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(m2M)


    output['E6S1OS1OAvg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(m1M)
    output['E6QMS1OS1OAvg'] = quadparam1 * pncoefficients.XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(m1M)

    output['E6S2OS2OAvg'] = pncoefficients.XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(m2M)
    output['E6QMS2OS2OAvg'] = quadparam2 * pncoefficients.XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(m2M)

    output['E7S1O'] =  pncoefficients.XLALSimInspiralPNEnergy_7PNSOCoeff(m1M)
    output['E7S2O'] = pncoefficients.XLALSimInspiralPNEnergy_7PNSOCoeff(m2M)

    output['Etidal10'] =  lambda1 * pncoefficients.XLALSimInspiralPNEnergy_10PNTidalCoeff(m1M) + lambda2 * pncoefficients.XLALSimInspiralPNEnergy_10PNTidalCoeff(m2M)

    output['Etidal12'] = lambda1 * pncoefficients.XLALSimInspiralPNEnergy_12PNTidalCoeff(m1M) + lambda2 * pncoefficients.XLALSimInspiralPNEnergy_12PNTidalCoeff(m2M)

    output['Ecoeff'] = jnp.array([
    1.0,  
    0.0, 
    pncoefficients.XLALSimInspiralPNEnergy_2PNCoeff(eta),  # 1PN
    0.0,  # 1.5PN
    pncoefficients.XLALSimInspiralPNEnergy_4PNCoeff(eta),  # 2PN
    0.0,  # 2.5PN
    pncoefficients.XLALSimInspiralPNEnergy_6PNCoeff(eta),  # 3PN
    0.0   # 3.5PN
])

    return output



def XLALSimInspiralSpinDerivativesAvg(
    v,
    LNhx, LNhy, LNhz,
    E1x, E1y, E1z,
    S1x, S1y, S1z,
    S2x, S2y, S2z,
    LNhdotS1,
    LNhdotS2,
    params
):
    """
    JAX port accepting `params` as a runtime PyTree (dict or object).
    Returns a jnp.array([dLNhx, dLNhy, dLNhz, dE1x, dE1y, dE1z,
                        dS1x, dS1y, dS1z, dS2x, dS2y, dS2z])
    Notes:
      - All control flow depends on JAX ops (comparisons, lax.cond) so this
        function is compatible with jax.jit with params as a runtime argument.
      - `params` must contain numeric fields used below (or they default to 0.0).
    """
    # Pack vectors
    LNhat = jnp.array([LNhx, LNhy, LNhz])
    E1 = jnp.array([E1x, E1y, E1z])
    S1 = jnp.array([S1x, S1y, S1z])
    S2 = jnp.array([S2x, S2y, S2z])

    # Basic derived quantities
    v2 = v * v
    omega = v * v2  # v^3
    omega2 = omega * omega
    # LN magnitude baseline as in C: LN0mag = eta / v
    eta = params['eta']
    LN0mag = eta / v
    #LNmag = LN0mag

    # Read spin-order and flags as runtime JAX values
    # spinO may be negative for "all" in C; support that by treating negative -> True in masks
    spinO = jnp.asarray(params['spinO'])
    lscorr = jnp.asarray(params['lscorr'])
    phenomtp = params['phenomtp']

    LNmag = LN0mag
    # Add 1PN correction if spinO >= 5
    include_1PN = (spinO >= 5) | (spinO < 0)
    L1PN = pncoefficients.XLALSimInspiralL_2PN(eta)  
    LNmag = jnp.where(include_1PN, LNmag + LN0mag * v2 * L1PN, LNmag)

    # boolean masks (0.0 or 1.0 floats) for each PN spin-order level
    m3 = jnp.where((spinO >= 3) | (spinO < 0), 1.0, 0.0)   # include LO spin (v^5)
    m4 = jnp.where((spinO >= 4) | (spinO < 0), 1.0, 0.0)   # include v^6 terms
    m5 = jnp.where((spinO >= 5) | (spinO < 0), 1.0, 0.0)   # include v^7 terms
    m6 = jnp.where((spinO >= 6) | (spinO < 0), 1.0, 0.0)   # include v^8 terms
    m7 = jnp.where((spinO >= 7) | (spinO < 0), 1.0, 0.0)   # for phenomtp branch potential v^9/v^? terms

    # Safe parameter reads (defaults to 0.0)
    S1dot3 = params["S1dot3"]
    S2dot3 = params["S2dot3"]

    # Precompute cross products used in multiple places
    LNh_x_S1 = cross_vec(LNhat, S1)
    LNh_x_S2 = cross_vec(LNhat, S2)
    S1_x_S2 = cross_vec(S1, S2)

    # ---------- LO spin (v^5) ----------
    # In C: dS1_lo = S1dot3 * v^5 * (LNh x S1)
    v5 = omega * v2  # v^5
    dS1_lo = S1dot3 * v5 * LNh_x_S1 * m3
    dS2_lo = S2dot3 * v5 * LNh_x_S2 * m3

    # dLNhat contributions from LO spins: dLNhat_lo = -(dS1_lo + dS2_lo)
    dLNhat_lo = -(dS1_lo + dS2_lo)

    # ---------- next (v^6) contributions ----------
    # coefficients that appear in v^6 (omega^2) terms
    # Names used in original C: wdot-like names were for domega; spin names here:
    S1dot4S2Avg = params["S1dot4S2Avg"]
    S1dot4S2OAvg = params["S1dot4S2OAvg"]
    S1dot4QMS1OAvg = params["S1dot4QMS1OAvg"]
    S2dot4QMS2OAvg = params["S2dot4QMS2OAvg"]

    # build v^6 contributions (note: omega2 = v^6)
    pref_v6 = omega2 * m4
    # dS1_v6 = pref_v6 * (-S1dot4S2Avg * (S1xS2) + S1dot4S2OAvg * LNhdotS2 * (LNh x S1))
    dS1_v6 = pref_v6 * (-S1dot4S2Avg * S1_x_S2 + S1dot4S2OAvg * LNhdotS2 * LNh_x_S1)
    dS2_v6 = pref_v6 * ( S1dot4S2Avg * S1_x_S2 + S1dot4S2OAvg * LNhdotS1 * LNh_x_S2)

    # Add QM self-spin v^6 terms
    dS1_v6 = dS1_v6 + pref_v6 * (S1dot4QMS1OAvg * LNhdotS1 * LNh_x_S1)
    dS2_v6 = dS2_v6 + pref_v6 * (S2dot4QMS2OAvg * LNhdotS2 * LNh_x_S2)

    dLNhat_v6 = -(dS1_v6 + dS2_v6)

    # ---------- v^7 contributions ----------
    S1dot5 = params["S1dot5"]
    S2dot5 = params["S2dot5"]
    # v^7 prefactor: v7 = omega2 * v (since omega2=v^6)
    v7 = omega2 * v
    dS1_v7 = S1dot5 * v7 * LNh_x_S1 * m5
    dS2_v7 = S2dot5 * v7 * LNh_x_S2 * m5
    dLNhat_v7 = -(dS1_v7 + dS2_v7) 

    # lscorr corrections (applied within certain orders in C) -> treat as v^2 * eta factor times some combos
    cS1 = params["cS1"]
    cS2 = params["cS2"]
    cS1L = params["cS1L"]
    cS2L = params["cS2L"]

    # For lscorr small correction terms, follow C's structure: add when m5 or m6 etc.
    # We'll compute a representative lscorr_v? contribution used in C proportional to eta*v^2
    lscorr_pref = eta * v2
    # Use dS1_lo and dS2_lo and dS1_v6/dS2_v6 etc in combinations similar to the C code.
    # We'll only include terms that were referenced in the provided C: (cS1*dS1_lo + cS2*dS2_lo) etc.
    dLNhat_lscorr_from_lo = - lscorr_pref * (cS1 * dS1_lo + cS2 * dS2_lo) * m5 * lscorr
    dLNhat_lscorr_from_v6 = - lscorr_pref * (cS1 * dS1_v6 + cS2 * dS2_v6) * m6 * lscorr
    # Additional lscorr pieces with cS1L,cS2L couple to dL_lo; approximate using dLNhat_lo scaled:
    dLNhat_lscorr_from_L = - lscorr_pref * ((cS1L * LNhdotS1 + cS2L * LNhdotS2) * dLNhat_lo / LN0mag) * m6 * lscorr

    # ---------- v^8 / v^? (N^3LO) contributions ----------
    # There are two branches: phenomtp True or False in original C.
    # We'll implement both and select via lax.cond.
    def non_phenom_branch(args):
        # compute v^8 contributions (v8 = omega2 * v2)
        v8 = omega2 * v2
        # read coefficients
        S1dot6S2Avg = params["S1dot6S2Avg"]
        S1dot6S1OAvg = params["S1dot6S1OAvg"]
        S1dot6S2OAvg = params["S1dot6S2OAvg"]
        S2dot6S1Avg = params["S2dot6S1Avg"]
        S2dot6S1OAvg = params["S2dot6S1OAvg"]
        S2dot6S2OAvg = params["S2dot6S2OAvg"]
        S1dot6QMS1OAvg = params["S1dot6QMS1OAvg"]
        S2dot6QMS2OAvg = params["S2dot6QMS2OAvg"]

        dS1_v8 = v8 * (-S1dot6S2Avg * S1_x_S2 + (S1dot6S1OAvg * LNhdotS1 + S1dot6S2OAvg * LNhdotS2) * LNh_x_S1)
        dS2_v8 = v8 * ( S2dot6S1Avg * S1_x_S2 + (S2dot6S1OAvg * LNhdotS1 + S2dot6S2OAvg * LNhdotS2) * LNh_x_S2)

        dS1_v8 = dS1_v8 + v8 * (S1dot6QMS1OAvg * LNhdotS1 * LNh_x_S1)
        dS2_v8 = dS2_v8 + v8 * (S2dot6QMS2OAvg * LNhdotS2 * LNh_x_S2)

        dLNhat_v8 = jnp.array([0.0, 0.0, 0.0])#-(dS1_v8 + dS2_v8)
        # additional lscorr terms included in non-phenom branch:
        dLNhat_v8_lscorr = - lscorr_pref * (cS1 * dS1_v6 + cS2 * dS2_v6 + (cS1L * LNhdotS1 + cS2L * LNhdotS2) * dLNhat_lo / LN0mag) * m6 * lscorr

        return dS1_v8 * m6, dS2_v8 * m6, dLNhat_v8 * m6, dLNhat_v8_lscorr

    def phenom_branch(args):
        # phenomtp branch uses different PN ordering where a v^? (e.g., v^9) block enters
        # We'll implement the C's "phenomtp" path that updates LNmag using L_4PN and uses omega^3 contributions
        L_4PN_val = _get(params, "L_4PN_val", 0.0)
        # update LNmag with LN0mag * v^4 * L_4PN
        LNmag_local = LNmag + LN0mag * (v2 * v2) * L_4PN_val
        # compute omega^3 (omega2 * omega)
        omega3 = omega2 * omega
        S1dot7S2 = _get(params, "S1dot7S2", 0.0)
        S2dot7S1 = _get(params, "S2dot7S1", 0.0)
        dS1_p = S1dot7S2 * omega3 * LNh_x_S1 * m7
        dS2_p = S2dot7S1 * omega3 * LNh_x_S2 * m7
        dLNhat_p = -(dS1_p + dS2_p)
        return dS1_p, dS2_p, dLNhat_p, jnp.zeros(3)

    # select branch for high-order contributions
    dS1_v8_branch, dS2_v8_branch, dLNhat_v8_branch, dLNhat_v8_lscorr = lax.cond(
        jnp.logical_not(phenomtp),
        non_phenom_branch,
        phenom_branch,
        operand=None
    )

    # Combine all contributions (LO + v6 + v7 + v8branch)
    dS1_total = dS1_lo + dS1_v6 + dS1_v7 + dS1_v8_branch
    dS2_total = dS2_lo + dS2_v6 + dS2_v7 + dS2_v8_branch

    dLNhat_total = dLNhat_lo + dLNhat_v6 + dLNhat_v7 + dLNhat_v8_branch + dLNhat_lscorr_from_lo + dLNhat_lscorr_from_v6 + dLNhat_lscorr_from_L + dLNhat_v8_lscorr

    # If spin orders were <3, masks set the contributions to zero; so totals are zero appropriately.

    # Normalize dLNhat by LNmag (C divides dLNhat by LNmag)
    dLNhat = dLNhat_total / LNmag

    # Compute Om = LNhat x dLNhat
    Om = cross_vec(LNhat, dLNhat)

    # dLNh = Om x LNhat
    dLNh = cross_vec(Om, LNhat)

    # dE1 = Om x E1
    dE1 = cross_vec(Om, E1)

    # Pack spin derivatives (we already built dS1_total and dS2_total)
    # dS1_total and dS2_total are length-3 arrays
    # Return order as in C: dLNhx,dLNhy,dLNhz, dE1x,dE1y,dE1z, dS1x,dS1y,dS1z, dS2x,dS2y,dS2z
    out = jnp.concatenate([dLNh, dE1, dS1_total, dS2_total])

    return out




def stopping_event(t, y, args, **kwargs):
    """Event function for stopping conditions"""
    # Compute derivatives at current state
    #t = state.t
    #y = state.y
    #args = state.args
    dy = XLALSimInspiralSpinTaylorT4DerivativesAvg(t, y, args)
    # Call stopping test
    result = XLALSimInspiralSpinTaylorStoppingTest(t, y, dy, args)
    # Event triggers when result crosses zero (from positive to negative)
    return result

# --------------------------
# Main JAX function
# --------------------------

def XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT: float,
    m1_SI: float,
    m2_SI: float,
    fStart: float,
    fEnd: float,
    s1x: float,
    s1y: float,
    s1z: float,
    s2x: float,
    s2y: float,
    s2z: float,
    lnhatx: float,
    lnhaty: float,
    lnhatz: float,
    e1x: float,
    e1y: float,
    e1z: float,
    lambda1: float,
    lambda2: float,
    quadparam1: float,
    quadparam2: float,
    spinO: int,
    tideO: int,
    phaseO: int,
    lscorr: int,
    approx: int,
    phenomtp: bool,
    max_len: int = 1000) -> Tuple[REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries,
           REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries,
           REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries,
           REAL8TimeSeries, REAL8TimeSeries]:
    """
    Integrate SpinTaylor PN equations.
    
    JAX version of XLALSimInspiralSpinTaylorPNEvolveOrbit from LALSimulation.
    Returns tuple of (V, Phi, S1x, S1y, S1z, S2x, S2y, S2z,
                      LNhatx, LNhaty, LNhatz, E1x, E1y, E1z)
    """
    
    # Setup params
    params = XLALSimInspiralSpinTaylorT4Setup(
        m1_SI, m2_SI, fStart, fEnd, lambda1, lambda2,
        quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, phenomtp
    )

    params_dict = params.to_dict()
    
    m1sec = params.m1sec
    m2sec = params.m2sec
    Msec = params.Msec
    Mcsec = params.Mcsec
    
    # Set sign of time step according to direction of integration
    sgn = jnp.where((fEnd < fStart) & (fEnd != 0.), -1, 1)
    
    # Estimate length using Newtonian t(f) formula
    dtStart = (5.0/256.0) * jnp.power(jnp.pi, -8.0/3.0) * \
              jnp.power(Mcsec * fStart, -5.0/3.0) / fStart

    dtEnd = jnp.where(
        fEnd == 0.,
        0.,
        (5.0/256.0) * jnp.power(jnp.pi, -8.0/3.0) * \
        jnp.power(Mcsec * fEnd, -5.0/3.0) / fEnd
    )

    lengths = dtStart - dtEnd
    
    # Put initial values into array
    norm1 = params.norm1
    norm2 = params.norm2
    
    yinit = jnp.array([
        0.,                          # phi: initial orbital phase = 0
        jnp.pi * Msec * fStart,     # omega: \hat{omega} = pi M f
        lnhatx, lnhaty, lnhatz,     # LNhat
        norm1 * s1x,                # S1 (normalized by M^2)
        norm1 * s1y,
        norm1 * s1z,
        norm2 * s2x,                # S2
        norm2 * s2y,
        norm2 * s2z,
        e1x, e1y, e1z,
    ])
    
    # Time span in dimensionless units \hat{t} = t/M
    t0 = 0.0
    t1 = lengths / Msec
    dt0 = sgn * deltaT / Msec
    
    # Determine number of steps

    n_steps = jnp.minimum(
        jnp.abs(jnp.floor(lengths / deltaT).astype(int)) + 1,
        max_len
    )
    jax.debug.print('n_steps {}',n_steps)
    sgnt1 = sgn * t1
    save_ts = jnp.linspace(t0, sgnt1, 1000) 

    # Run integration
    term = ODETerm(XLALSimInspiralSpinTaylorT4DerivativesAvg)
   
    solver = Tsit5()
    saveat = SaveAt(ts=save_ts)
    stepsize_controller = PIDController(
        rtol=LAL_ST4_RELATIVE_TOLERANCE,
        atol=LAL_ST4_ABSOLUTE_TOLERANCE
    )

    sol = diffeqsolve(
        term, solver,
        t0=t0, t1=sgn * t1, dt0=dt0,
        y0=yinit,
        args=params_dict,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_len, 
        event=Event(cond_fn = stopping_event)
    )

    yout = sol.ys  

    valid_mask = jnp.all(jnp.isfinite(yout), axis=1)
    n_invalid = jnp.sum(~valid_mask)
    
   
    if n_invalid > 0:
        print(f'Removing {n_invalid} points with inf/nan')
        print(f'yout shape before: {yout.shape}')

        yout = yout[valid_mask]
        print(f'yout shape after: {yout.shape}')

        print(f'Last 5 rows of yout:')
        print(yout[-5:, :]) 
        len_result = yout.shape[0]    




    # Handle cutoff at fEnd
    cutlen = len_result
    if fEnd != 0.:
        wEnd = jnp.pi * Msec * fEnd
        omega_series = yout[:, 1]
        
        if fEnd < fStart:
            # Backward integration
            crosses = omega_series < wEnd
        else:
            # Forward integration  
            crosses = omega_series > wEnd
        
        first_cross = jnp.argmax(crosses)
        has_crossing = jnp.any(crosses)
        cutlen = jnp.where(has_crossing, first_cross + 1, len_result)
    
    # Slice to cutlen
    yout = yout[:cutlen]
    
    # If integrated backwards, reverse order
    yout = jnp.where(
        (fEnd < fStart) & (fEnd != 0.),
        jnp.flip(yout, axis=0),
        yout
    )
    
    # Extract variables
    # yout columns: [phi, omega, LNx, LNy, LNz, S1x, S1y, S1z, S2x, S2y, S2z, E1x, E1y, E1z]
    Phi_data = yout[:, 0]
    omega = yout[:, 1]
    V_data = jnp.cbrt(omega)
    
    LNhatx_data = yout[:, 2]
    LNhaty_data = yout[:, 3]
    LNhatz_data = yout[:, 4]
    
    # Spins returned in standard convention (denormalized)
    S1x_data = yout[:, 5] / norm1
    S1y_data = yout[:, 6] / norm1
    S1z_data = yout[:, 7] / norm1
    
    S2x_data = yout[:, 8] / norm2
    S2y_data = yout[:, 9] / norm2
    S2z_data = yout[:, 10] / norm2
    
    E1x_data = yout[:, 11]
    E1y_data = yout[:, 12]
    E1z_data = yout[:, 13]
    
    # Create REAL8TimeSeries objects
    V = REAL8TimeSeries(data=V_data, deltaT=deltaT)
    Phi = REAL8TimeSeries(data=Phi_data, deltaT=deltaT)
    S1x = REAL8TimeSeries(data=S1x_data, deltaT=deltaT)
    S1y = REAL8TimeSeries(data=S1y_data, deltaT=deltaT)
    S1z = REAL8TimeSeries(data=S1z_data, deltaT=deltaT)
    S2x = REAL8TimeSeries(data=S2x_data, deltaT=deltaT)
    S2y = REAL8TimeSeries(data=S2y_data, deltaT=deltaT)
    S2z = REAL8TimeSeries(data=S2z_data, deltaT=deltaT)
    LNhatx = REAL8TimeSeries(data=LNhatx_data, deltaT=deltaT)
    LNhaty = REAL8TimeSeries(data=LNhaty_data, deltaT=deltaT)
    LNhatz = REAL8TimeSeries(data=LNhatz_data, deltaT=deltaT)
    E1x = REAL8TimeSeries(data=E1x_data, deltaT=deltaT)
    E1y = REAL8TimeSeries(data=E1y_data, deltaT=deltaT)
    E1z = REAL8TimeSeries(data=E1z_data, deltaT=deltaT)
    

    return (V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, 
            LNhatx, LNhaty, LNhatz, E1x, E1y, E1z)



def compute_n_steps(fStart: float, fEnd: float, Mcsec: float, deltaT: float, max_len: int = 100000)->int:

    dtStart = (5.0/256.0) * jnp.power(jnp.pi, -8.0/3.0) * \
              jnp.power(Mcsec * fStart, -5.0/3.0) / fStart
    dtEnd = jnp.where(
        fEnd == 0.,
        0.,
        (5.0/256.0) * jnp.power(jnp.pi, -8.0/3.0) * \
        jnp.power(Mcsec * fEnd, -5.0/3.0) / fEnd
    )

    lengths = dtStart - dtEnd

    n_steps = jnp.minimum(
        jnp.abs(jnp.floor(lengths / deltaT).astype(int)) + 1,
        max_len
    )
    return int(n_steps)

def example():
    """Example matching C API style"""
    
    m1_SI = 36 * LAL_MSUN_SI
    m2_SI = 29 * LAL_MSUN_SI

    m1sec = m1_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    m2sec = m2_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    Msec = m1sec + m2sec
    eta = m1sec * m2sec / (Msec * Msec)
    Mcsec = Msec * jnp.power(eta, 0.6)

   

    # Call function

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = \
        XLALSimInspiralSpinTaylorPNEvolveOrbit(
            deltaT=0.05,
            m1_SI=m1_SI,
            m2_SI=m2_SI,
            fStart=20.0,
            fEnd=1000.0,
            s1x=0.0, s1y=0.0, s1z=0.3,
            s2x=0.0, s2y=0.0, s2z=0.2,
            lnhatx=0.0, lnhaty=0.0, lnhatz=1.0,
            e1x=1.0, e1y=0.0, e1z=0.0,
            lambda1=0.0,
            lambda2=0.0,
            quadparam1=1.0,
            quadparam2=1.0,
            spinO=-1,
            tideO=-1,
            phaseO=-1,
            lscorr=0,
            approx=4
        )
    
    print(f"V length: {len(V.data)}")
    print(f"Final V: {V.data[-1]:.6f}")
    
    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z



#-1.1270493097 0.0358329850 -1.4194513987 -13.0249912073
#wdotspins four values... -1.1270493096646943 0.03583298496055225 -1.4194513986827415 -13.024991207348954


#-1.1320704670 0.0710770890 -1.4548400623 -13.2207779975

#-1.1321081507 0.0713704444 -1.4552037264 -13.2223704447