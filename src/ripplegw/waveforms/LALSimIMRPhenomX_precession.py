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

from .LALSimIMRPhenomX_PNR_internals import IMRPhenomX_PNR_HMInterpolationDeltaF

from .LALSimIMRPhenomX_PNR_alpha import IMRPhenomX_PNR_precompute_alpha_coefficients
from .LALSimIMRPhenomX_PNR_beta import (
    IMRPhenomX_PNR_precompute_beta_coefficients,
    IMRPhenomX_PNR_BetaConnectionFrequencies
)

from .LALSimIMRPhenomXHM_internals import IMRPhenomXHM_GenerateRingdownFrequency




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

    def _setup_spintaylor_prescription(self):
        """
        Setup SpinTaylor prescription for self.IMRPhenomXPrecVersion//100 == 3.
        Reference: Lines 242-389 in raw_LALSimIMRPhenomX_precession.py
        """

        object.__setattr__(self, 'L_MAX_PNR', jnp.max(jnp.array(self.lalParams['ModeArray'])))
        object.__setattr__(self, 'M_MAX', jnp.max(jnp.array(self.lalParams['ModeArray'][:, 1])))
        flow = self.pWF['fMin']

        # Handle deltaF == 0 case
        deltaMF = jnp.where(
            self.pWF['deltaF'] == 0,
            get_deltaF_from_wfstruct(self.pWF),
            -1 #FIXME
        )
        self.pWF['deltaMF'] = deltaMF

        # Branch on PNRUseTunedAngles
        # Path 1: PNRUseTunedAngles == False
        integration_buffer_path1 = jnp.where(self.pWF['deltaF'] > 0., 3. * self.pWF['deltaF'], 0.5)
        flow_path1 = (self.pWF['fMin'] - integration_buffer_path1) * 2 / self.M_MAX

        # Path 2: PNRUseTunedAngles == True
        # Line 257-260
        iStart_here_path2 = jnp.where(
            self.pWF['deltaF'] == 0.,
            0,
            jnp.floor(self.pWF['fMin'] / self.pWF['deltaF']).astype(int)
        )
        flow_temp_path2 = jnp.where(
            self.pWF['deltaF'] == 0.,
            0.,
            iStart_here_path2 * self.pWF['deltaF']
        )
        fmin_HM_inspiral = flow_temp_path2 * 2.0 / self.M_MAX

        # Temporarily set version to 223 for PNR variable computation
        precVersion_save = self.IMRPhenomXPrecVersion
        object.__setattr__(self, 'IMRPhenomXPrecVersion', 223)


        # These functions would need to be implemented
        # IMRPhenomX_PNR_GetAndSetPNRVariables(self, self.pWF)

        alphaParams = IMRPhenomX_PNR_precompute_alpha_coefficients(self.pWF, self)
        betaParams = IMRPhenomX_PNR_precompute_beta_coefficients(self.pWF, self)
        Mf_beta_lower, Mf_beta_upper = IMRPhenomX_PNR_BetaConnectionFrequencies(betaParams)

        # Restore version
        object.__setattr__(self, 'IMRPhenomXPrecVersion', precVersion_save)

        # Line 274-276
        Mf_alpha_upper = alphaParams.A4 / 3.0
        Mf_low_cut = (3.0 / 3.5) * Mf_alpha_upper
        MF_high_cut = Mf_beta_lower

        # Line 280-287: First conditional assignment
        MF_high_cut = jnp.where(
            jnp.logical_or(
                MF_high_cut > self.pWF['fCutDef'],
                MF_high_cut < 0.1 * self.pWF['fRING']
            ),
            self.pWF['fRING'],
            MF_high_cut
        )

        # Line 290-297: Second conditional assignment
        Mf_low_cut = jnp.where(
            jnp.logical_or(
                Mf_low_cut > self.pWF['fCutDef'],
                MF_high_cut < Mf_low_cut
            ),
            MF_high_cut / 2.0,
            Mf_low_cut
        )

        # Line 300
        flow_alpha = XLALSimIMRPhenomXUtilsMftoHz(
            Mf_low_cut * 0.65 * self.M_MAX / 2.0,
            self.pWF['Mtot']
        )

        # Line 305-316: Compute else branch for main conditional
        Mf_RD_22 = self.pWF['fRING']
        Mf_RD_lm = IMRPhenomXHM_GenerateRingdownFrequency(self.L_MAX_PNR, self.M_MAX, self.pWF)

        fmin_HM_ringdown = XLALSimIMRPhenomXUtilsMftoHz(
            XLALSimIMRPhenomXUtilsHztoMf(flow_temp_path2, self.pWF['Mtot']) - (Mf_RD_lm - Mf_RD_22),
            self.pWF['Mtot']
        )
        else_branch_result = jnp.where(
            jnp.logical_and(fmin_HM_ringdown < fmin_HM_inspiral, fmin_HM_ringdown > 0.0),
            fmin_HM_ringdown,
            fmin_HM_inspiral
        )

        # Line 319: Main conditional
        flow_path2_intermediate = jnp.where(
            flow_alpha < flow_temp_path2,
            fmin_HM_inspiral / 1.5,
            else_branch_result
        )

        # Line 322-327
        pnr_interpolation_deltaf = IMRPhenomX_PNR_HMInterpolationDeltaF(flow_path2_intermediate, self.pWF, self)

        integration_buffer_path2 = 1.4 * pnr_interpolation_deltaf
        flow_path2 = jnp.where(
            flow_path2_intermediate - 2.0 * pnr_interpolation_deltaf < 0,
            flow_path2_intermediate / 2.0,
            flow_path2_intermediate - 2.0 * pnr_interpolation_deltaf
        )

        iStart_here_path2_final = jnp.floor(flow_path2 / pnr_interpolation_deltaf).astype(int)
        flow_path2 = iStart_here_path2_final * pnr_interpolation_deltaf

        # Select path based on PNRUseTunedAngles
        integration_buffer = jnp.where(self.lalParams['PNRUseTunedAngles'], integration_buffer_path2, integration_buffer_path1)
        flow = jnp.where(self.lalParams['PNRUseTunedAngles'], flow_path2, flow_path1)

        object.__setattr__(self, 'integration_buffer', integration_buffer)

        PNarrays, fmin_integration = IMRPhenomX_InspiralAngles_SpinTaylor(
             self.chi1x, self.chi1y, self.chi1z,
             self.chi2x, self.chi2y, self.chi2z,
             flow, self.IMRPhenomXPrecVersion,
             self.pWF, self.lalParams)
        object.__setattr__(self, 'Mfmin_integration',
                           XLALSimIMRPhenomXUtilsHztoMf(fmin_integration, self.pWF['Mtot']))

        # Line 335-381: Handle IMRPhenomXPrecVersion == 330 case
        # Extract evolved spins from PNarrays and apply rotation
        # Using jnp.where to make it JAX-compatible

        # Extract final values from PNarrays (last element)
        lenPN = len(PNarrays[0])

        chi1x_temp = PNarrays[1][lenPN-1]
        chi1y_temp = PNarrays[2][lenPN-1]
        chi1z_temp = PNarrays[3][lenPN-1]

        chi2x_temp = PNarrays[4][lenPN-1]
        chi2y_temp = PNarrays[5][lenPN-1]
        chi2z_temp = PNarrays[6][lenPN-1]

        Lx = PNarrays[7][lenPN-1]
        Ly = PNarrays[8][lenPN-1]
        Lz = PNarrays[9][lenPN-1]

        # Calculate rotation angles from angular momentum
        phi = jnp.arctan2(Ly, Lx)
        L_mag = jnp.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        theta = jnp.arccos(Lz / L_mag)

        # Rotate chi1 vector
        _v1 = IMRPhenomX_rotate_z(-phi, jnp.array([chi1x_temp, chi1y_temp, chi1z_temp]))
        chi1_rotated = IMRPhenomX_rotate_y(-theta, _v1)

        # Rotate chi2 vector
        _v2 = IMRPhenomX_rotate_z(-phi, jnp.array([chi2x_temp, chi2y_temp, chi2z_temp]))
        chi2_rotated = IMRPhenomX_rotate_y(-theta, _v2)

        # Conditionally use rotated or original values based on IMRPhenomXPrecVersion
        is_version_330 = (self.IMRPhenomXPrecVersion == 330)

        chi1x_evolved = jnp.where(is_version_330, chi1_rotated[0], self.chi1x)
        chi1y_evolved = jnp.where(is_version_330, chi1_rotated[1], self.chi1y)
        chi1z_evolved = jnp.where(is_version_330, chi1_rotated[2], self.chi1z)

        chi2x_evolved = jnp.where(is_version_330, chi2_rotated[0], self.chi2x)
        chi2y_evolved = jnp.where(is_version_330, chi2_rotated[1], self.chi2y)
        chi2z_evolved = jnp.where(is_version_330, chi2_rotated[2], self.chi2z)

        # Set evolved spin attributes
        object.__setattr__(self, 'chi1x_evolved', chi1x_evolved)
        object.__setattr__(self, 'chi1y_evolved', chi1y_evolved)
        object.__setattr__(self, 'chi1z_evolved', chi1z_evolved)
        object.__setattr__(self, 'chi2x_evolved', chi2x_evolved)
        object.__setattr__(self, 'chi2y_evolved', chi2y_evolved)
        object.__setattr__(self, 'chi2z_evolved', chi2z_evolved)


def get_deltaF_from_wfstruct(pWF: dict) -> float:
    """
    Get deltaF from waveform structure
    
    Args:
        pWF: Waveform structure dictionary (dict)
        
    Returns:
        float: Delta frequency in dimensionless units
    """
    
    seglen = XLALSimInspiralChirpTimeBound(
        pWF['fRef'], pWF['m1_SI'], pWF['m2_SI'], pWF['chi1L'], pWF['chi2L']
    )
    
    deltaFv1 = 1.0 / jnp.maximum(4.0, jnp.power(2, jnp.ceil(jnp.log(seglen)/jnp.log(2))))
    deltaF = jnp.minimum(deltaFv1, 0.1)
    deltaMF = XLALSimIMRPhenomXUtilsHztoMf(deltaF, pWF['Mtot'])
    
    return deltaMF



def XLALSimInspiralChirpTimeBound(fstart: float, m1: float, m2: float, s1: float, s2: float) -> float:
    """
    Calculate chirp time bound for inspiral
    
    Args:
        fstart: Starting frequency (float)
        m1: Mass of object 1 (float)
        m2: Mass of object 2 (float)
        s1: Spin of object 1 (float)
        s2: Spin of object 2 (float)
        
    Returns:
        float: Chirp time bound
    """
    
    M = m1 + m2  # total mass
    mu = m1 * m2 / M  # reduced mass
    eta = mu / M  # symmetric mass ratio
    
    # chi = (s1*m1 + s2*m2)/M <= max(|s1|,|s2|)
    # over-estimate of chi
    chi = jnp.abs(jnp.where(jnp.abs(s1) > jnp.abs(s2), s1, s2))
    
    # note: for some reason these coefficients are named wrong...
    # "2PN" should be "1PN", "4PN" should be "2PN", etc.
    c0 = jnp.abs(XLALSimInspiralTaylorT2Timing_0PNCoeff(M, eta))
    c2 = XLALSimInspiralTaylorT2Timing_2PNCoeff(eta)
    
    # the 1.5pN spin term is in TaylorT2 is 8*beta/5 [Citation ??]
    # where beta = (113/12 + (25/4)(m2/m1))*(s1*m1^2/M^2) + 2 <-> 1
    # [Cutler & Flanagan, Physical Review D 49, 2658 (1994), Eq. (3.21)]
    # which can be written as (113/12)*chi - (19/6)(s1 + s2)
    # and we drop the negative contribution
    c3 = (226.0/15.0) * chi
    
    # there is also a 1.5PN term with eta, but it is negative so do not include it
    c4 = XLALSimInspiralTaylorT2Timing_4PNCoeff(eta)
    
    v = jnp.power(jnp.pi * G * M * fstart, 1.0/3.0) / C
    
    return c0 * jnp.power(v, -8) * (1.0 + (c2 + (c3 + c4 * v) * v) * v * v)

def XLALSimInspiralTaylorT2Timing_0PNCoeff(totalmass: float, eta: float) -> float:
    """
    Calculate 0PN coefficient for TaylorT2 timing
    
    Args:
        totalmass: Total mass in kilograms (float)
        eta: Symmetric mass ratio (float)
        
    Returns:
        float: 0PN timing coefficient
    """
    
    # convert totalmass from kilograms to seconds
    totalmass *= G / jnp.power(C, 3.0)
    
    return -5.0 * totalmass / (256.0 * eta)


def XLALSimInspiralTaylorT2Timing_2PNCoeff(eta: float) -> float:
    """
    Calculate 2PN coefficient for TaylorT2 timing
    
    Args:
        eta: Symmetric mass ratio (float)
        
    Returns:
        float: 2PN timing coefficient
    """
    
    return 7.43/2.52 + 11.0/3.0 * eta


def XLALSimInspiralTaylorT2Timing_4PNCoeff(eta: float) -> float:

    return 30.58673/5.08032 + 54.29/5.04*eta + 61.7/7.2*eta*eta



def XLALSimIMRPhenomXUtilsHztoMf(fHz: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from Hz to geometric units (Mf).

    Parameters
    ----------
    fHz : float
        Frequency in Hz
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Geometric frequency Mf
    """
    # Mtot in seconds = Mtot_Msun * MTSUN_SI
    MTSUN_SI = 4.925491025543575903411922162094833998e-6  # seconds
    return fHz * Mtot_Msun * MTSUN_SI


def XLALSimIMRPhenomXUtilsMftoHz(Mf: float, Mtot_Msun: float) -> float:
    """
    Convert frequency from geometric units (Mf) to Hz.

    Parameters
    ----------
    Mf : float
        Geometric frequency
    Mtot_Msun : float
        Total mass in solar masses

    Returns
    -------
    float
        Frequency in Hz
    """
    MTSUN_SI = 4.925491025543575903411922162094833998e-6  # seconds
    return Mf / (Mtot_Msun * MTSUN_SI)


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
    #if(coarse_fac  < 1) { XLAL_ERROR(XLAL_EDOM, "Coarse factor must be >= 1!\n")}

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




def integrate_forward(fRef, fmin, fCut, deltaT_coarse, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr, approx):
    '''
    If fRef is zero or is equal to fmin, we only need to integrate from fmin to fCut, i.e., forward. 
    This function is called to perform forward integration. 
    Line 4690-4697
    '''

    fS = fmin
    fE = fCut

    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)
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
    V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, m1_SI, m2_SI,fS,fE,s1x,s1y,s1z,s2x,s2y,s2z,lnhatx,lnhaty,lnhatz,e1x,e1y,e1z,lambda1,lambda2,quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)


    fS = fRef
    fE = fCut
    #Skipping the sanity check of if...else. Just jump to forward integration. 
    V_forward, Phi_forward, S1x_forward, S1y_forward, S1z_forward, S2x_forward, S2y_forward, S2z_forward, LNhatx_forward, LNhaty_forward, LNhatz_forward, E1x_forward, E1y_forward, E1z_forward = XLALSimInspiralSpinTaylorPNEvolveOrbit(deltaT_coarse, 
                                                                                                                                            m1_SI, m2_SI, fS, fE, s1x, s1y, s1z, s2x, s2y,
                                                                                                                                            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1,lambda2, quadparam1, quadparam2, spinO, tideO, phaseO, lscorr)
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
