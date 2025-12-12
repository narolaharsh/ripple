import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from jax import jit, lax
from functools import partial

# Constants (from LAL)
LAL_PI = jnp.pi
LAL_MSUN_SI = 1.98892e30
LAL_MTSUN_SI = 4.925491025543576e-06
LAL_G_SI = 6.67430e-11
LAL_C_SI = 299792458.0

LAL_ST4_ABSOLUTE_TOLERANCE = 1.0e-11
LAL_ST4_RELATIVE_TOLERANCE = 1.0e-9
LAL_NUM_ST4_VARIABLES = 14

@dataclass
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
    lscorr: int
) -> XLALSimInspiralSpinTaylorTxCoeffs:
    """Setup parameters for SpinTaylorT4"""
    m1sec = m1_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    m2sec = m2_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    Msec = m1sec + m2sec
    eta = m1sec * m2sec / (Msec * Msec)
    Mcsec = Msec * jnp.power(eta, 0.6)
    norm1 = m1sec * m1sec / Msec / Msec
    norm2 = m2sec * m2sec / Msec / Msec
    
    return XLALSimInspiralSpinTaylorTxCoeffs(
        m1_SI=m1_SI, m2_SI=m2_SI, fStart=fStart, fEnd=fEnd,
        lambda1=lambda1, lambda2=lambda2,
        quadparam1=quadparam1, quadparam2=quadparam2,
        spinO=spinO, tideO=tideO, phaseO=phaseO, lscorr=lscorr,
        m1sec=m1sec, m2sec=m2sec, Msec=Msec, Mcsec=Mcsec,
        eta=eta, norm1=norm1, norm2=norm2
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
    wspin3 = 0.0
    wspin4 = 0.0
    wspin5 = 0.0
    wspin6 = 0.0

    # -----------------------------------------------------
    # domega
    # -----------------------------------------------------
    domega = params.wdotnewt * v11 * (
        params.wdotcoeff[0]
        + v * (
            params.wdotcoeff[1]
            + v * (
                params.wdotcoeff[2]
                + v * (
                    params.wdotcoeff[3] + wspin3
                    + v * (
                        params.wdotcoeff[4] + wspin4
                        + v * (
                            params.wdotcoeff[5] + wspin5
                            + v * (
                                params.wdotcoeff[6] + wspin6
                                + params.wdotlogcoeff * jnp.log(v)
                                + v * (
                                    params.wdotcoeff[7]
                                    + omega * (
                                        params.wdottidal10
                                        + v2 * params.wdottidal12
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
                       params.omegashiftS1,
                       params.omegashiftS2)

    dphi = omega * (1 + omega*omega*shift)

    # -----------------------------------------------------
    # Output vector (matches dvalues in C)
    # -----------------------------------------------------
    return jnp.array([
        dphi, domega,
        dLNhx, dLNhy, dLNhz,
        dS1x,  dS1y,  dS1z,
        dS2x,  dS2y,  dS2z,
        dE1x,  dE1y,  dE1z
    ])

def XLALSimInspiralSpinTaylorStoppingTest(t, y, params):
    """Stopping test for integration"""
    return 1.0  # Positive = continue

@dataclass
class REAL8TimeSeries:
    """JAX equivalent of LAL REAL8TimeSeries"""
    data: jax.Array
    deltaT: float
    epoch: float = 0.0

# --------------------------
# Stub helper functions
# --------------------------

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


@jit
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
    eta = _get(params, "eta", 0.0)
    LN0mag = eta / v
    LNmag = LN0mag

    # Read spin-order and flags as runtime JAX values
    # spinO may be negative for "all" in C; support that by treating negative -> True in masks
    spinO = jnp.asarray(_get(params, "spinO", 0))
    lscorr = jnp.asarray(int(_get(params, "lscorr", 0)))
    phenomtp = bool(_get(params, "phenomtp", False))

    # boolean masks (0.0 or 1.0 floats) for each PN spin-order level
    m3 = jnp.where((spinO >= 3) | (spinO < 0), 1.0, 0.0)   # include LO spin (v^5)
    m4 = jnp.where((spinO >= 4) | (spinO < 0), 1.0, 0.0)   # include v^6 terms
    m5 = jnp.where((spinO >= 5) | (spinO < 0), 1.0, 0.0)   # include v^7 terms
    m6 = jnp.where((spinO >= 6) | (spinO < 0), 1.0, 0.0)   # include v^8 terms
    m7 = jnp.where((spinO >= 7) | (spinO < 0), 1.0, 0.0)   # for phenomtp branch potential v^9/v^? terms

    # Safe parameter reads (defaults to 0.0)
    S1dot3 = _get(params, "S1dot3", 0.0)
    S2dot3 = _get(params, "S2dot3", 0.0)

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
    S1dot4S2Avg = _get(params, "S1dot4S2Avg", 0.0)
    S1dot4S2OAvg = _get(params, "S1dot4S2OAvg", 0.0)
    S1dot4QMS1OAvg = _get(params, "S1dot4QMS1OAvg", 0.0)
    S2dot4QMS2OAvg = _get(params, "S2dot4QMS2OAvg", 0.0)

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
    S1dot5 = _get(params, "S1dot5", 0.0)
    S2dot5 = _get(params, "S2dot5", 0.0)
    # v^7 prefactor: v7 = omega2 * v (since omega2=v^6)
    v7 = omega2 * v
    dS1_v7 = S1dot5 * v7 * LNh_x_S1 * m5
    dS2_v7 = S2dot5 * v7 * LNh_x_S2 * m5
    dLNhat_v7 = -(dS1_v7 + dS2_v7)

    # lscorr corrections (applied within certain orders in C) -> treat as v^2 * eta factor times some combos
    cS1 = _get(params, "cS1", 0.0)
    cS2 = _get(params, "cS2", 0.0)
    cS1L = _get(params, "cS1L", 0.0)
    cS2L = _get(params, "cS2L", 0.0)

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
        S1dot6S2Avg = _get(params, "S1dot6S2Avg", 0.0)
        S1dot6S1OAvg = _get(params, "S1dot6S1OAvg", 0.0)
        S1dot6S2OAvg = _get(params, "S1dot6S2OAvg", 0.0)
        S2dot6S1Avg = _get(params, "S2dot6S1Avg", 0.0)
        S2dot6S1OAvg = _get(params, "S2dot6S1OAvg", 0.0)
        S2dot6S2OAvg = _get(params, "S2dot6S2OAvg", 0.0)
        S1dot6QMS1OAvg = _get(params, "S1dot6QMS1OAvg", 0.0)
        S2dot6QMS2OAvg = _get(params, "S2dot6QMS2OAvg", 0.0)

        dS1_v8 = v8 * (-S1dot6S2Avg * S1_x_S2 + (S1dot6S1OAvg * LNhdotS1 + S1dot6S2OAvg * LNhdotS2) * LNh_x_S1)
        dS2_v8 = v8 * ( S2dot6S1Avg * S1_x_S2 + (S2dot6S1OAvg * LNhdotS1 + S2dot6S2OAvg * LNhdotS2) * LNh_x_S2)

        dS1_v8 = dS1_v8 + v8 * (S1dot6QMS1OAvg * LNhdotS1 * LNh_x_S1)
        dS2_v8 = dS2_v8 + v8 * (S2dot6QMS2OAvg * LNhdotS2 * LNh_x_S2)

        dLNhat_v8 = -(dS1_v8 + dS2_v8)
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

# --------------------------
# Main JAX function
# --------------------------

@jax.jit
def XLALSimInspiralSpinTaylorPNEvolveOrbit(
    deltaT: float,
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
    n_steps: int,
    max_len: int = 100000,
) -> Tuple[REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries, REAL8TimeSeries,
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
        quadparam1, quadparam2, spinO, tideO, phaseO, lscorr
    )
    
    m1sec = params.m1sec
    m2sec = params.m2sec
    Msec = params.Msec
    Mcsec = params.Mcsec
    
    # Set sign of time step according to direction of integration
    sgn = jnp.where((fEnd < fStart) & (fEnd != 0.), -1, 1)
    
    # Estimate length using Newtonian t(f) formula
    dtStart = (5.0/256.0) * jnp.power(LAL_PI, -8.0/3.0) * \
              jnp.power(Mcsec * fStart, -5.0/3.0) / fStart
    dtEnd = jnp.where(
        fEnd == 0.,
        0.,
        (5.0/256.0) * jnp.power(LAL_PI, -8.0/3.0) * \
        jnp.power(Mcsec * fEnd, -5.0/3.0) / fEnd
    )
    lengths = dtStart - dtEnd
    
    # Put initial values into array
    norm1 = params.norm1
    norm2 = params.norm2
    
    yinit = jnp.array([
        0.,                          # phi: initial orbital phase = 0
        LAL_PI * Msec * fStart,     # omega: \hat{omega} = pi M f
        lnhatx, lnhaty, lnhatz,     # LNhat
        norm1 * s1x,                # S1 (normalized by M^2)
        norm1 * s1y,
        norm1 * s1z,
        norm2 * s2x,                # S2
        norm2 * s2y,
        norm2 * s2z,
        e1x, e1y, e1z               # E1
    ])
    
    # Time span in dimensionless units \hat{t} = t/M
    t0 = 0.0
    t1 = lengths / Msec
    dt0 = sgn * deltaT / Msec
    
    # Determine number of steps
    '''
    n_steps = jnp.minimum(
        jnp.abs(jnp.floor(lengths / deltaT).astype(int)) + 1,
        max_len
    )
    '''
    print('set up save times', t0, sgn * t1, n_steps)
    save_ts = jnp.linspace(t0, sgn * t1, n_steps)
    
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
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_len * 10
    )
    
    yout = sol.ys  # shape: (len, 14)
    len_result = yout.shape[0]
    
    # Handle cutoff at fEnd
    cutlen = len_result
    if fEnd != 0.:
        wEnd = LAL_PI * Msec * fEnd
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



def compute_n_steps(fStart: float, fEnd: float, Mcsec: float, deltaT: float, max_len: int = 100000):

    dtStart = (5.0/256.0) * jnp.power(LAL_PI, -8.0/3.0) * \
              jnp.power(Mcsec * fStart, -5.0/3.0) / fStart
    dtEnd = jnp.where(
        fEnd == 0.,
        0.,
        (5.0/256.0) * jnp.power(LAL_PI, -8.0/3.0) * \
        jnp.power(Mcsec * fEnd, -5.0/3.0) / fEnd
    )

    lengths = dtStart - dtEnd

    n_steps = jnp.minimum(
        jnp.abs(jnp.floor(lengths / deltaT).astype(int)) + 1,
        max_len
    )
    return n_steps

# Example usage
#appros: SpinTaylorT4.......key int(4)
def example():
    """Example matching C API style"""
    
    m1_SI = 1.4 * LAL_MSUN_SI
    m2_SI = 1.4 * LAL_MSUN_SI

    m1sec = m1_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    m2sec = m2_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    Msec = m1sec + m2sec
    eta = m1sec * m2sec / (Msec * Msec)
    Mcsec = Msec * jnp.power(eta, 0.6)

    n_steps = compute_n_steps(fStart = 20.0, fEnd = 1000.0, Mcsec=Mcsec, deltaT = 0.1)
   
    print('steps', n_steps)
    # Call function
    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = \
        XLALSimInspiralSpinTaylorPNEvolveOrbit(
            deltaT=0.1,
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
            spinO=6,
            tideO=12,
            phaseO=7,
            lscorr=0,
            n_steps = n_steps,
            approx=4
        )
    
    print(f"V length: {len(V.data)}")
    print(f"Final V: {V.data[-1]:.6f}")
    
    return V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z


example()