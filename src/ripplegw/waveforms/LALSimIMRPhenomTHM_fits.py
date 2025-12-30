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

