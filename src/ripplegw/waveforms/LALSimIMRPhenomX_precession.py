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
    # Basic parameters

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

    common_constants: CommonConstants = field(default_factory=CommonConstants)

    m1: float = m1_SI / pWF['Mtot_SI']







