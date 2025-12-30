import jax.numpy as jnp
import math
from ..typing import Array
from ..constants import G, MSUN, C, MTSUN_SI, GAMMA
import jax
from .spherical_harmonics import *
from .IMRPhenomXPHM_utils import *
from .LALSimInspiralSpinTaylor import XLALSimInspiralSpinTaylorPNEvolveOrbit
from dataclasses import dataclass, field
from jax_dataclasses import pytree_dataclass

#from .LALSimIMRPhenomX_PNR_internals import IMRPhenomX_PNR_HMInterpolationDeltaF



@pytree_dataclass
class CommonConstants:
    sqrt2: float = 1.4142135623730951
    sqrt5: float = 2.23606797749978981
    sqrt6: float = 2.44948974278317788
    sqrt7: float = 2.64575131106459072
    sqrt10: float = 3.16227766016838
    sqrt14: float = 3.74165738677394133
    sqrt15: float = 3.87298334620741702
    sqrt70: float = 8.36660026534075563
    sqrt30: float = 5.477225575051661
    sqrt2p5: float = 1.58113883008419
    log16: float = 2.772588722239781
    power_of_lalpi_2: float = 9.869604401089358
    MAX_TOL_ATAN: float = 1.0e-15


@pytree_dataclass
class IMRPhenomXGetAndSetPrecessionVariables:
    """
    lalsuite: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html#af089ef2586c52b12016c0d791b176121

    Parameters
    ----------
    pWF : dict
        Some useful waveform parameters
    m1_SI : float
        Mass of object 1 (heavier one) in SI units
    m2_SI : float
        Mass of object 2 (lighter one) in SI units
    chi1x : float
        Object 1 spin x-component
    chi1y : float
        Object 1 spin y-component
    chi1z : float
        Object 1 spin z-component
    chi2x : float
        Object 2 spin x-component
    chi2y : float
        Object 2 spin y-component
    chi2z : float
        Object 2 spin z-component
    lalParams : dict
        LAL parameters
    debug_flag : bool
        Debug flag

    Note: We assume m1 > m2, q > 1, dm = m1 - m2 = delta = sqrt(1-4eta) > 0
    """
    # Input parameters
    pWF: dict
    m1_SI: float
    m2_SI: float
    chi1x: float
    chi1y: float
    chi1z: float
    chi2x: float
    chi2y: float
    chi2z: float
    lalParams: dict
    debug_flag: bool

    # Common constants
    common_constants: CommonConstants = field(default_factory=CommonConstants)

    # Computed attributes (initialized in __post_init__)
    m1: float = field(init=False)
    m2: float = field(init=False)
    M: float = field(init=False)
    q: float = field(init=False)

    # Mass powers
    m1_2: float = field(init=False)
    m1_3: float = field(init=False)
    m1_4: float = field(init=False)
    m1_5: float = field(init=False)
    m1_6: float = field(init=False)
    m1_7: float = field(init=False)
    m1_8: float = field(init=False)
    m2_2: float = field(init=False)

    # Eta (symmetric mass ratio) powers
    eta: float = field(init=False)
    eta2: float = field(init=False)
    eta3: float = field(init=False)
    eta4: float = field(init=False)
    eta5: float = field(init=False)
    eta6: float = field(init=False)

    # Delta powers
    delta: float = field(init=False)
    delta2: float = field(init=False)
    delta3: float = field(init=False)

    # Inverse eta
    inveta: float = field(init=False)
    inveta2: float = field(init=False)
    inveta3: float = field(init=False)
    inveta4: float = field(init=False)
    sqrt_inveta: float = field(init=False)

    # Effective spin
    chi_eff: float = field(init=False)

    # Gravitational constants
    twopiGM: float = field(init=False)
    piGM: float = field(init=False)

    # Spin norms
    chi1_norm: float = field(init=False)
    chi2_norm: float = field(init=False)

    # Dimensionful spins
    S1x: float = field(init=False)
    S1y: float = field(init=False)
    S1z: float = field(init=False)
    S1_norm: float = field(init=False)
    S2x: float = field(init=False)
    S2y: float = field(init=False)
    S2z: float = field(init=False)
    S2_norm: float = field(init=False)

    # Spin norm powers
    S1_norm_2: float = field(init=False)
    S2_norm_2: float = field(init=False)

    # Perpendicular spin components
    chi1_perp: float = field(init=False)
    chi2_perp: float = field(init=False)
    S1_perp: float = field(init=False)
    S2_perp: float = field(init=False)

    # Total perpendicular spin
    STot_perp: float = field(init=False)
    chiTot_perp: float = field(init=False)

    # Effective precessing spin parameters (Schmidt et al, PRD 91, 024043, 2015)
    A1: float = field(init=False)
    A2: float = field(init=False)
    ASp1: float = field(init=False)
    ASp2: float = field(init=False)
    chip: float = field(init=False)
    chi1L: float = field(init=False)
    chi2L: float = field(init=False)
    chi_p: float = field(init=False)
    phi0_aligned: float = field(init=False)
    SL: float = field(init=False)
    Sperp: float = field(init=False)
    pWF22AS: None = field(init=False, default=None)

    # Version-specific flags
    IMRPhenomXPrecVersion: int = field(init=False)
    PolarizationSymmetry: float = field(init=False)



    def __post_init__(self):
        """Compute all derived quantities."""
        self._setup_version_flags()
        self._compute_masses()
        self._compute_mass_powers()
        self._update_pWF_dict()
        self._compute_eta_quantities()
        self._compute_gravitational_constants()
        self._compute_spin_quantities()
        self._compute_effective_spin_parameters()
        self._validate_kerr_bound()
        

    def _compute_masses(self):
        """Compute normalized masses and mass ratio."""
        object.__setattr__(self, 'm1', self.m1_SI / self.pWF['Mtot_SI'])
        object.__setattr__(self, 'm2', self.m2_SI / self.pWF['Mtot_SI'])
        object.__setattr__(self, 'M', self.m1 + self.m2)
        object.__setattr__(self, 'q', self.m1 / self.m2)

    def _compute_mass_powers(self):
        """Compute useful powers of masses."""
        # Powers of m1
        object.__setattr__(self, 'm1_2', self.m1 * self.m1)
        object.__setattr__(self, 'm1_3', self.m1 * self.m1_2)
        object.__setattr__(self, 'm1_4', self.m1 * self.m1_3)
        object.__setattr__(self, 'm1_5', self.m1 * self.m1_4)
        object.__setattr__(self, 'm1_6', self.m1 * self.m1_5)
        object.__setattr__(self, 'm1_7', self.m1 * self.m1_6)
        object.__setattr__(self, 'm1_8', self.m1 * self.m1_7)

        # Powers of m2
        object.__setattr__(self, 'm2_2', self.m2 * self.m2)

    def _update_pWF_dict(self):
        """Update pWF dictionary with computed mass values."""
        self.pWF['M'] = self.M
        self.pWF['m1_2'] = self.m1_2
        self.pWF['m2_2'] = self.m2_2

    def _compute_eta_quantities(self):
        """Compute eta (symmetric mass ratio) and delta related quantities."""
        # Powers of eta
        object.__setattr__(self, 'eta', self.pWF['eta'])
        object.__setattr__(self, 'eta2', self.eta * self.eta)
        object.__setattr__(self, 'eta3', self.eta * self.eta2)
        object.__setattr__(self, 'eta4', self.eta * self.eta3)
        object.__setattr__(self, 'eta5', self.eta * self.eta4)
        object.__setattr__(self, 'eta6', self.eta * self.eta5)

        # Delta in terms of q > 1
        object.__setattr__(self, 'delta', self.pWF['delta'])
        object.__setattr__(self, 'delta2', self.delta * self.delta)
        object.__setattr__(self, 'delta3', self.delta * self.delta2)

        # Inverse eta (cached for efficiency)
        object.__setattr__(self, 'inveta', 1.0 / self.eta)
        object.__setattr__(self, 'inveta2', 1.0 / self.eta2)
        object.__setattr__(self, 'inveta3', 1.0 / self.eta3)
        object.__setattr__(self, 'inveta4', 1.0 / self.eta4)
        object.__setattr__(self, 'sqrt_inveta', 1.0 / jnp.sqrt(self.eta))

        # Effective aligned spin
        object.__setattr__(self, 'chi_eff', self.pWF['chiEff'])

    def _compute_gravitational_constants(self):
        """Compute gravitational constants."""
        object.__setattr__(self, 'twopiGM', 2 * jnp.pi * G * (self.m1_SI + self.m2_SI) / C / C / C)
        object.__setattr__(self, 'piGM', jnp.pi * (self.m1_SI + self.m2_SI) * (G / C) / (C * C))

    def _compute_spin_quantities(self):
        """Compute all spin-related quantities."""
        # Spin norms
        object.__setattr__(self, 'chi1_norm', jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y + self.chi1z * self.chi1z))
        object.__setattr__(self, 'chi2_norm', jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y + self.chi2z * self.chi2z))

        # Dimensionful spins
        object.__setattr__(self, 'S1x', self.chi1x * self.m1_2)
        object.__setattr__(self, 'S1y', self.chi1y * self.m1_2)
        object.__setattr__(self, 'S1z', self.chi1z * self.m1_2)
        object.__setattr__(self, 'S1_norm', jnp.abs(self.chi1_norm) * self.m1_2)

        object.__setattr__(self, 'S2x', self.chi2x * self.m2_2)
        object.__setattr__(self, 'S2y', self.chi2y * self.m2_2)
        object.__setattr__(self, 'S2z', self.chi2z * self.m2_2)
        object.__setattr__(self, 'S2_norm', jnp.abs(self.chi2_norm) * self.m2_2)

        # Spin norm powers
        object.__setattr__(self, 'S1_norm_2', self.S1_norm * self.S1_norm)
        object.__setattr__(self, 'S2_norm_2', self.S2_norm * self.S2_norm)

        # Perpendicular spin components
        object.__setattr__(self, 'chi1_perp', jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y))
        object.__setattr__(self, 'chi2_perp', jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y))

        # Spin projections
        object.__setattr__(self, 'S1_perp', self.m1_2 * jnp.sqrt(self.chi1x * self.chi1x + self.chi1y * self.chi1y))
        object.__setattr__(self, 'S2_perp', self.m2_2 * jnp.sqrt(self.chi2x * self.chi2x + self.chi2y * self.chi2y))

        # Total perpendicular spin (norm of in-plane vector sum: Norm[S1perp + S2perp])
        object.__setattr__(self, 'STot_perp', jnp.sqrt((self.S1x + self.S2x) * (self.S1x + self.S2x) + (self.S1y + self.S2y) * (self.S1y + self.S2y)))

        # chiTot_perp (distinguishes from Sperp used in construction of chi_p)
        # For normalization, see Sec. IV D of arXiv:2004.06503
        object.__setattr__(self, 'chiTot_perp', self.STot_perp * (self.M * self.M) / self.m1_2)

        # Store chiTot_perp to pWF for use in XCP modifications (PNRUseTunedCoprec)
        self.pWF['chiTot_perp'] = self.chiTot_perp

    def _setup_version_flags(self):
        """Setup version-specific flags and configuration parameters."""
        # Get IMRPhenomX precession version from LAL dictionary
        object.__setattr__(self, 'IMRPhenomXPrecVersion', self.lalParams['IMRPhenomXPrecVersion'])

        # Convert version 300 to 223
        version = jnp.where(self.IMRPhenomXPrecVersion == 300, 223, self.IMRPhenomXPrecVersion)
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)

        # Calculate in-plane spin magnitude
        chi_in_plane = jnp.sqrt(
            self.chi1x * self.chi1x + self.chi1y * self.chi1y +
            self.chi2x * self.chi2x + self.chi2y * self.chi2y
        )

        # Default to NNLO angles if in-plane spins are negligible and version 330 is selected
        # The solutions would be dominated by numerical noise
        version = jnp.where(
            (chi_in_plane < 1e-6) & (self.IMRPhenomXPrecVersion == 330),
            102,
            self.IMRPhenomXPrecVersion
        )
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)

        # Default to NNLO if in-plane spins are negligible and SpinTaylor option is selected
        version = jnp.where(
            (chi_in_plane < 1e-7) & (self.IMRPhenomXPrecVersion//100 == 3),
            102,
            self.IMRPhenomXPrecVersion
        )
        object.__setattr__(self, 'IMRPhenomXPrecVersion', version)
        object.__setattr__(self, 'PolarizationSymmetry', 1.0)

        #Line 245-255
        # Disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit
        cond = (chi_in_plane<1e-7) & (self.lalParams['PNRUseTunedAngles'] == 1) & (self.pWF['PNR_SINGLE_SPIN']!=1)
        self.lalParams['PNRUseTunedAngles'] = jnp.where(cond, False, self.lalParams['PNRUseTunedAngles'])
        self.lalParams['AntisymmetricWaveform'] = jnp.where(cond, False, self.lalParams['AntisymmetricWaveform'])
        self.lalParams['PNRUseTunedCoprec'] = jnp.where(cond, False, self.lalParams['PNRUseTunedCoprec'])

    def _compute_effective_spin_parameters(self):
        """
        Calculate the effective precessing spin parameter.
        Reference: Schmidt et al, PRD 91, 024043, 2015
        Note: m1 > m2, so body 1 is the larger black hole
        """
        # Compute A1 and A2 coefficients
        object.__setattr__(self, 'A1', 2.0 + (3.0 * self.m2) / (2.0 * self.m1))
        object.__setattr__(self, 'A2', 2.0 + (3.0 * self.m1) / (2.0 * self.m2))
        object.__setattr__(self, 'ASp1', self.A1 * self.S1_perp)
        object.__setattr__(self, 'ASp2', self.A2 * self.S2_perp)

        # S_p = max(A1 S1_perp, A2 S2_perp)
        num = jnp.where(self.ASp2 > self.ASp1, self.ASp2, self.ASp1)
        den = jnp.where(self.m2 > self.m1, self.A2 * self.m2_2, self.A1 * self.m1_2)

        # chi_p = max(A1 * Sp1, A2 * Sp2) / (A_i * m_i^2) where i is the index of the larger BH
        object.__setattr__(self, 'chip', num / den)
        object.__setattr__(self, 'chi1L', self.chi1z)
        object.__setattr__(self, 'chi2L', self.chi2z)

        object.__setattr__(self, 'chi_p', self.chip)
        # Store chi_p to pWF (used in PNRUseTunedCoprec)
        self.pWF['chi_p'] = self.chi_p
        object.__setattr__(self, 'phi0_aligned', self.pWF['phi0'])

        # Effective (dimensionful) aligned spin
        object.__setattr__(self, 'SL', self.chi1L * self.m1_2 + self.chi2L * self.m2_2)

        # Effective (dimensionful) in-plane spin (m1 > m2)
        object.__setattr__(self, 'Sperp', self.chi_p * self.m1_2)

        # Initialize pWF22AS for SpinTaylor code
        object.__setattr__(self, 'pWF22AS', None)

    def _validate_kerr_bound(self):
        """Validate that spin magnitudes are within Kerr bound."""
        kerr_bound_flag = check_kerr_bound(
            self.lalParams['PNRUseTunedAngles'],
            self.pWF['PNR_SINGLE_SPIN'],
            self.chi1_norm,
            self.chi2_norm
        )


def check_kerr_bound(pnr_use_tuned_angles, pnr_single_spin, chi1_norm, chi2_norm):
    """
    Check if spin magnitudes violate the Kerr bound.

    Parameters
    ----------
    pnr_use_tuned_angles : bool
        Whether PNR tuned angles are used
    pnr_single_spin : int
        PNR single spin flag
    chi1_norm : float
        Magnitude of spin 1
    chi2_norm : float
        Magnitude of spin 2

    Returns
    -------
    bool
        True if valid, False if Kerr bound is violated
    """
    # Condition to apply check
    should_check = jnp.logical_or(
        jnp.logical_not(pnr_use_tuned_angles),
        pnr_single_spin != 1)

    # Compute violations
    chi1_violation = jnp.abs(chi1_norm) > 1.0
    chi2_violation = jnp.abs(chi2_norm) > 1.0

    # Only raise error if we should check AND there's a violation
    error_condition = jnp.logical_and(
        should_check,
        jnp.logical_or(chi1_violation, chi2_violation)
    )

    # Return success/error flag
    return jnp.where(
        error_condition,
        False,  # Error case
        True    # Success case
    )
