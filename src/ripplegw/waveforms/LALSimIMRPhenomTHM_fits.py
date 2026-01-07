import jax.numpy as jnp

def evaluate_QNMfit_fring21(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring21
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    # (In JAX, we can't raise errors in jit-compiled code)
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    
    numerator = (0.059471695665734674 - 0.07585416297991414*finalDimlessSpin + 
                 0.021967909664591865*x2 - 0.0018964744613388146*x3 + 
                 0.001164879406179587*x4 - 0.0003387374454044957*x5)
    
    denominator = (1 - 1.4437415542456158*finalDimlessSpin + 0.49246920313191234*x2)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring33(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring33
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.22799517865876945*finalDimlessSpin + 
                 0.13402916709362475*x2 + 0.03343753057911253*x3 - 
                 0.030848060170259615*x4 - 0.006756504382964637*x5 + 
                 0.0027301732074159835*x6)
    
    denominator = (1 - 2.7265947806178334*finalDimlessSpin + 2.144070539525238*x2 - 
                   0.4706873667569393*x4 + 0.05321818246993958*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring32(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring32
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.13628306966373951*finalDimlessSpin + 
                 0.030099881830507727*x2 - 0.000673589757007597*x3 + 
                 0.0118277880067919*x4 + 0.0020533816327907334*x5 - 
                 0.0015206141948469621*x6)
    
    denominator = (1 - 1.6531854335715193*finalDimlessSpin + 0.5634705514193629*x2 + 
                   0.12256204148002939*x4 - 0.027297817699401976*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring44(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring44
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.1287821193485683 - 0.21224284094693793*finalDimlessSpin + 
                 0.0710926778043916*x2 + 0.015487322972031054*x3 - 
                 0.002795401084713644*x4 + 0.000045483523029172406*x5 + 
                 0.00034775290179000503*x6)
    
    denominator = (1 - 1.9931645124693607*finalDimlessSpin + 1.0593147376898773*x2 - 
                   0.06378640753152783*x4)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp21(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp21 (damping frequency for (2,1) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    # (In JAX, we can't raise errors in jit-compiled code)
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2

    numerator = (2.0696914454467294 - 3.1358071947583093*finalDimlessSpin +
                 0.14456081596393977*x2 + 1.2194717985037946*x3 -
                 0.2947372598589144*x4 + 0.002943057145913646*x5)

    denominator = (146.1779212636481 - 219.81790388304876*finalDimlessSpin +
                   17.7141194900164*x2 + 75.90115083917898*x3 -
                   18.975287709794745*x4)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp33(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp33 (damping frequency for (3,3) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2

    numerator = (0.014754148319335946 - 0.03124423610028678*finalDimlessSpin +
                 0.017192623913708124*x2 + 0.001034954865629645*x3 -
                 0.0015925124814622795*x4 - 0.0001414350555699256*x5)

    denominator = (1.0 - 2.0963684630756894*finalDimlessSpin +
                   1.196809702382645*x2 - 0.09874113387889819*x4)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp32(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp32 (damping frequency for (3,2) mode)

    NOTE: This function is not defined in the original LALSimulation C code.
    For now, this returns the same as fdamp33 as a placeholder approximation.

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """
    # Placeholder: use fdamp33 as approximation since fdamp32 is not defined in C code
    return evaluate_QNMfit_fdamp33(finalDimlessSpin)


def evaluate_QNMfit_fdamp44(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp44 (damping frequency for (4,4) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (0.014986847152355699 - 0.01722587715950451*finalDimlessSpin -
                 0.0016734788189065538*x2 + 0.0002837322846047305*x3 +
                 0.002510528746148588*x4 + 0.00031983835498725354*x5 +
                 0.000812185411753066*x6)

    denominator = (1.0 - 1.1350205970682399*finalDimlessSpin -
                   0.0500827971270845*x2 + 0.13983808071522857*x4 +
                   0.051876225199833995*x6)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)
