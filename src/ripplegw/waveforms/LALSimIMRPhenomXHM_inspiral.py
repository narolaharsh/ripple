"""
IMRPhenomXHM inspiral amplitude functions converted to JAX.

This module contains the inspiral amplitude coefficient functions for the
IMRPhenomXHM waveform model, converted from the LALSimulation C code to JAX.
"""

import jax.numpy as jnp


def IMRPhenomXHM_Insp_Amp_21_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.037868557189995156 + 0.10740090317702103 * eta + 1.963812986867654 * eta2
            - 16.706455229589558 * eta3 + 69.75910808095745 * eta4 - 98.3062466823662 * eta5
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.007963757232702219 + 0.10627108779259965 * eta - 0.008044970210401218 * S
            + eta2 * (-0.4735861262934258 - 0.5985436493302649 * S - 0.08217216660522082 * S2)
        )

        uneqSpin = (
            -0.257787704938017 * pWF['dchi'] * eta2 * (1. + 8.75928187268504 * eta2)
            - 0.2597503605427412 * pWF['dchi'] * eta2 * S
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = jnp.abs(
            chidiff1 * eta5 * (-3962.5020052272976 + 987.635855365408 * S1 - 134.98527058315528 * S2)
            + delta * (19.30531354642419 + 16.6640319856064 * eta1 - 120.58166037019478 * eta2
                      + 220.77233521626252 * eta3) * sqroot
            + chidiff1 * delta * (31.364509907424765 * eta1 - 843.6414532232126 * eta2
                                 + 2638.3077554662905 * eta3) * sqroot
            + chidiff1 * delta * (32.374226994179054 * eta1 - 202.86279451816662 * eta2
                                 + 347.1621871204769 * eta3) * S1 * sqroot
            + delta * S1 * (
                -16.75726972301224 * (1.1787350890261943 - 7.812073811917883 * eta1
                                     + 99.47071002831267 * eta2 - 500.4821414428368 * eta3
                                     + 876.4704270866478 * eta4)
                + 2.3439955698372663 * (0.9373952326655807 + 7.176140122833879 * eta1
                                       - 279.6409723479635 * eta2 + 2178.375177755584 * eta3
                                       - 4768.212511142035 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_21_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.05511628628738656 - 0.12579599745414977 * eta + 2.831411618302815 * eta2
            - 14.27268643447161 * eta3 + 28.3307320191161 * eta4
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.008692738851491525 + eta * (0.09512553997347649 + 0.116470975986383 * S)
            - 0.009520793625590234 * S + eta2 * (-0.3409769288480959 - 0.8321002363767336 * S
                                                   - 0.13099477081654226 * S2)
            - 0.006383232900211555 * S2
        )

        uneqSpin = (
            -0.2962753588645467 * pWF['dchi'] * eta2 * (1. + 1.3993978458830476 * eta2)
            - 0.17100612756133535 * pWF['dchi'] * eta2 * S * (1. + 18.974303741922743 * eta2 * delta)
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = jnp.abs(
            chidiff1 * eta5 * (-2898.9172078672705 + 580.9465034962822 * S1 + 22.251142639924076 * S2)
            + delta * (
                chidiff2 * (-18.541685007214625 * eta1 + 166.7427445020744 * eta2 - 417.5186332459383 * eta3)
                + chidiff1 * (41.61457952037761 * eta1 - 779.9151607638761 * eta2 + 2308.6520892707795 * eta3)
            ) * sqroot
            + delta * (11.414934585404561 + 30.883118528233638 * eta1 - 260.9979123967537 * eta2
                      + 1046.3187137392433 * eta3 - 1556.9475493549746 * eta4) * sqroot
            + delta * S1 * (
                -10.809007068469844 * (1.1408749895922659 - 18.140470190766937 * eta1
                                      + 368.25127088896744 * eta2 - 3064.7291458207815 * eta3
                                      + 11501.848278358668 * eta4 - 16075.676528787526 * eta5)
                + 1.0088254664333147 * (1.2322739396680107 - 192.2461213084741 * eta1
                                       + 4257.760834055382 * eta2 - 35561.24587952242 * eta3
                                       + 130764.22485304279 * eta4 - 177907.92440833704 * eta5) * S1
            ) * sqroot
            + delta * (
                chidiff1 * (36.88578491943111 * eta1 - 321.2569602623214 * eta2 + 748.6659668096737 * eta3) * S1
                + chidiff1 * (-95.42418611585117 * eta1 + 1217.338674959742 * eta2
                             - 3656.192371615541 * eta3) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_21_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.059110044024271766 - 0.0024538774422098405 * eta + 0.2428578654261086 * eta2
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.007044339356171243 - 0.006952154764487417 * S
            + eta2 * (-0.016643018304732624 - 0.12702579620537421 * S + 0.004623467175906347 * S2)
            - 0.007685497720848461 * S2
        )

        uneqSpin = (
            -0.3172310538516028 * pWF['dchi'] * (1. - 2.9155919835488024 * eta2) * eta2
            - 0.11975485688200693 * pWF['dchi'] * eta2 * S * (1. + 17.27626751837825 * eta2 * delta)
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = jnp.abs(
            chidiff1 * eta5 * (-2282.9983216879655 + 157.94791186394787 * S1 + 16.379731479465033 * S2)
            + chidiff1 * delta * (21.935833431534224 * eta1 - 460.7130131927895 * eta2
                                 + 1350.476411541137 * eta3) * sqroot
            + delta * (5.390240326328237 + 69.01761987509603 * eta1 - 568.0027716789259 * eta2
                      + 2435.4098320959706 * eta3 - 3914.3390484239667 * eta4) * sqroot
            + chidiff1 * delta * (29.731007410186827 * eta1 - 372.09609843131386 * eta2
                                 + 1034.4897198648962 * eta3) * S1 * sqroot
            + delta * S1 * (
                -7.1976397556450715 * (0.7603360145475428 - 6.587249958654174 * eta1
                                      + 120.87934060776237 * eta2 - 635.1835857158857 * eta3
                                      + 1109.0598539312573 * eta4)
                - 0.0811847192323969 * (7.951454648295709 + 517.4039644814231 * eta1
                                       - 9548.970156895082 * eta2 + 52586.63520999897 * eta3
                                       - 93272.17990295641 * eta4) * S1
                - 0.28384547935698246 * (-0.8870770459576875 + 180.0378964169756 * eta1
                                        - 2707.9572896559484 * eta2 + 14158.178124971111 * eta3
                                        - 24507.800226675925 * eta4) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        S2 = S ** 2

        noSpin = (
            jnp.sqrt(1. - 4. * eta) * (-0.056586690934283326 - 0.14374841547279146 * eta
                                       + 0.5584776628959615 * eta2)
        ) / (-0.3996185676368123 + eta)

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            (0.056042044149691175 + 0.12482426029674777 * S) * S
            + eta * (2.1108074577110343 - 1.7827773156978863 * S2)
            + eta2 * (-7.657635515668849 - 0.07646730296478217 * S + 5.343277927456605 * S2)
        )

        uneqSpin = 0.45866449225302536 * pWF['dchi'] * (1. - 9.603750707244906 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = (
            chidiff1 * eta5 * (155.1434307076563 + 26.852777193715088 * S1 + 1.4157230717300835 * S2)
            + chidiff1 * delta * (6.296698171560171 * eta1 + 15.81328761563562 * eta2
                                 - 141.85538063933927 * eta3) * sqroot
            + delta * (20.94372147101354 + 68.14577638017842 * eta1 - 898.470298591732 * eta2
                      + 4598.64854748635 * eta3 - 8113.199260593833 * eta4) * sqroot
            + chidiff1 * delta * (29.221863857271703 * eta1 - 348.1658322276406 * eta2
                                 + 965.4670353331536 * eta3) * S1 * sqroot
            + delta * S1 * (
                -9.753610761811967 * (1.7819678168496158 - 44.07982999150369 * eta1
                                     + 750.8933447725581 * eta2 - 5652.44754829634 * eta3
                                     + 19794.855873435758 * eta4 - 26407.40988450443 * eta5)
                + 0.014210376114848208 * (-196.97328616330392 + 7264.159472864562 * eta1
                                         - 125763.47850622259 * eta2 + 1.1458022059130718e6 * eta3
                                         - 4.948175330328345e6 * eta4 + 7.911048294733888e6 * eta5) * S1
                - 0.26859293613553986 * (-8.029069605349488 + 888.7768796633982 * eta1
                                        - 16664.276483466252 * eta2 + 128973.72291098491 * eta3
                                        - 462437.2690007375 * eta4 + 639989.1197424605 * eta5) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.2137734510411439 - 0.7692194209223682 * eta + 26.10570221351058 * eta2
            - 316.0643979123107 * eta3 + 2090.9063511488234 * eta4 - 6897.3285171507105 * eta5
            + 8968.893362362503 * eta6
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            0.018546836505210842 + 0.05924304311104228 * S
            + eta * (1.6484440612224325 - 0.4683932646001618 * S - 2.110311135456494 * S2)
            + 0.10701786057882816 * S2
            + eta2 * (-6.51575737684721 + 1.6692205620001157 * S + 8.351789152096782 * S2)
        )

        uneqSpin = 0.3929315188124088 * pWF['dchi'] * (1. - 11.289452844364227 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = (
            chidiff1 * eta5 * (161.62678370819597 + 37.141092711336846 * S1 - 0.16889712161410445 * S2)
            + chidiff1 * delta * (3.4895829486899825 * eta1 + 51.07954458810889 * eta2
                                 - 249.71072528701757 * eta3) * sqroot
            + delta * (12.501397517602173 + 35.75290806646574 * eta1 - 357.6437296928763 * eta2
                      + 1773.8883882162215 * eta3 - 3100.2396041211605 * eta4) * sqroot
            + chidiff1 * delta * (13.854211287141906 * eta1 - 135.54916401086845 * eta2
                                 + 327.2467193417936 * eta3) * S1 * sqroot
            + delta * S1 * (
                -5.2580116732827085 * (1.7794900975289085 - 48.20753331991333 * eta1
                                      + 861.1650630146937 * eta2 - 6879.681319382729 * eta3
                                      + 25678.53964955809 * eta4 - 36383.824902258915 * eta5)
                + 0.028627002336747746 * (-50.57295946557892 + 734.7581857539398 * eta1
                                         - 2287.0465658878725 * eta2 + 15062.821881048358 * eta3
                                         - 168311.2370167227 * eta4 + 454655.37836367317 * eta5) * S1
                - 0.15528289788512326 * (-12.738184090548508 + 1129.44485109116 * eta1
                                        - 25091.14888164863 * eta2 + 231384.03447562453 * eta3
                                        - 953010.5908118751 * eta4 + 1.4516597366230418e6 * eta5) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.2363760327127446 + 0.2855410252403732 * eta - 10.159877125359897 * eta2
            + 162.65372389693505 * eta3 - 1154.7315106095564 * eta4 + 3952.61320206691 * eta5
            - 5207.67472857814 * eta6
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            0.04573095188775319 + 0.048249943132325494 * S
            + eta * (0.15922377052827502 - 0.1837289613228469 * S - 0.2834348500565196 * S2)
            + 0.052963737236081304 * S2
        )

        uneqSpin = 0.25187274502769835 * pWF['dchi'] * (1. - 12.172961866410864 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        eta7 = eta1 * eta6
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            chidiff1 * delta * (-0.5869777957488564 * eta1 + 32.65536124256588 * eta2
                               - 110.10276573567405 * eta3)
            + chidiff1 * delta * (3.524800489907584 * eta1 - 40.26479860265549 * eta2
                                 + 113.77466499598913 * eta3) * S1
            + delta * S1 * (
                -1.2846335585108297 * (0.09991079016763821 + 1.37856806162599 * eta1
                                      + 23.26434219690476 * eta2 - 34.842921754693386 * eta3
                                      - 70.83896459998664 * eta4)
                - 0.03496714763391888 * (-0.230558571912664 + 188.38585449575902 * eta1
                                        - 3736.1574640444287 * eta2 + 22714.70643022915 * eta3
                                        - 43221.0453556626 * eta4) * S1
            )
            + chidiff1 * eta7 * (2667.3441342894776 + 47.94869769580204 * chidiff2
                                + 793.5988192446642 * S1 + 293.89657731755483 * S2)
            + delta * (5.148353856800232 + 148.98231189649468 * eta1 - 2774.5868652930294 * eta2
                      + 29052.156454239772 * eta3 - 162498.31493332976 * eta4 + 460912.76402476896 * eta5
                      - 521279.50781871413 * eta6) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        eta8 = eta6 * eta * eta
        S2 = S ** 2
        S3 = S2 * S

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.019069933430190773 - 0.19396651989685837 * eta + 11.95224600241255 * eta2
            - 158.90113442757382 * eta3 + 1046.65239329071 * eta4 - 3476.940285294999 * eta5
            + 4707.249209858949 * eta6
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.0046910348789512895 + 0.40231360805609434 * eta - 0.0038263656140933152 * S
            + 0.018963579407636953 * S2
            + eta2 * (-1.955352354930108 + 2.3753413452420133 * S - 0.9085620866763245 * S3)
            + 0.02738043801805805 * S3
            + eta3 * (7.977057990568723 - 7.9259853291789515 * S + 0.49784942656123987 * S2
                     + 5.2255665027119145 * S3)
        )

        uneqSpin = (
            0.058560321425018165 * pWF['dchi'] ** 2 * (1. - 19.936477485971217 * eta2) * eta2
            + 1635.4240644598524 * pWF['dchi'] * eta8 * delta
            + 0.2735219358839411 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (-0.739317114582042 * eta1 - 47.473246070362634 * eta2
                                + 278.9717709112207 * eta3 - 566.6420939162068 * eta4)
             + chidiff2 * (-0.5873680378268906 * eta1 + 6.692187014925888 * eta2
                          - 24.37776782232888 * eta3 + 23.783684827838247 * eta4)) * sqroot
            + (3.2940434453819694 + 4.94285331708559 * eta1 - 343.3143244815765 * eta2
              + 3585.9269057886418 * eta3 - 19279.186145681153 * eta4 + 51904.91007211022 * eta5
              - 55436.68857586653 * eta6) * sqroot
            + chidiff1 * delta * (12.488240781993923 * eta1 - 209.32038774208385 * eta2
                                 + 1160.9833883184604 * eta3 - 2069.5349737049073 * eta4) * S1 * sqroot
            + S1 * (
                0.6343034651912586 * (-2.5844888818001737 + 78.98200041834092 * eta1
                                     - 1087.6241783616488 * eta2 + 7616.234910399297 * eta3
                                     - 24776.529123239357 * eta4 + 30602.210950069973 * eta5)
                - 0.062088720220899465 * (6.5586380356588565 + 36.01386705325694 * eta1
                                         - 3124.4712274775407 * eta2 + 33822.437731298516 * eta3
                                         - 138572.93700180828 * eta4 + 198366.10615196894 * eta5) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta8 = eta4 * eta4
        S2 = S ** 2
        S3 = S2 * S

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.024621376891809633 - 0.09692699636236377 * eta + 2.7200998230836158 * eta2
            - 16.160563094841066 * eta3 + 32.930430889650836 * eta4
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.008522695567479373 - 1.1104639098529456 * eta2 - 0.00362963820787208 * S
            + 0.016978054142418417 * S2
            + eta * (0.24280554040831698 + 0.15878436411950506 * S - 0.1470288177047577 * S3)
            + 0.029465887557447824 * S3
            + eta3 * (4.649438233164449 - 0.7550771176087877 * S + 0.3381436950547799 * S2
                     + 2.5663386135613093 * S3)
        )

        uneqSpin = (
            -0.007061187955941243 * pWF['dchi'] ** 2 * (1. - 2.024701925508361 * eta2) * eta2
            + 215.06940561269835 * pWF['dchi'] * eta8 * delta
            + 0.1465612311350642 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.03940151060321499 * eta1 + 1.9034209537174116 * eta2 - 8.78587250202154 * eta3)
             + chidiff1 * delta * (-1.704299788495861 * eta1 - 4.923510922214181 * eta2
                                  + 0.36790005839460627 * eta3)) * sqroot
            + (2.2911849711339123 - 5.1846950040514335 * eta1 + 60.10368251688146 * eta2
              - 1139.110227749627 * eta3 + 7970.929280907627 * eta4 - 25472.73682092519 * eta5
              + 30950.67053883646 * eta6) * sqroot
            + S1 * (
                0.7718201508695763 * (-1.3012906461000349 + 26.432880113146012 * eta1
                                     - 186.5001124789369 * eta2 + 712.9101229418721 * eta3
                                     - 970.2126139442341 * eta4)
                + 0.04832734931068797 * (-5.9999628512498315 + 78.98681284391004 * eta1
                                        + 1.8360177574514709 * eta2 - 2537.636347529708 * eta3
                                        + 6858.003573909322 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta8 = eta2 * eta2 * eta2 * eta2
        S2 = S ** 2
        S3 = S2 * S

        noSpin = (
            jnp.sqrt(1. - 3. * eta) * (-0.006150151041614737 + 0.017454430190035 * eta
                                       + 0.02620962593739105 * eta2 - 0.019043090896351363 * eta3)
        ) / (-0.2655505633361449 + eta)

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.011073381681404716 + 0.00347699923233349 * S
            + eta * S * (0.05592992411391443 - 0.15666140197050316 * S2)
            + 0.012079324401547036 * S2
            + eta2 * (0.5440307361144313 - 0.008730335213434078 * S + 0.04615964369925028 * S2
                     + 0.6703688097531089 * S3)
            + 0.016323101357296865 * S3
        )

        uneqSpin = (
            -0.020140175824954427 * pWF['dchi'] ** 2 * (1. - 12.675522774051249 * eta2) * eta2
            - 417.3604094454253 * pWF['dchi'] * eta8 * delta
            + 0.10464021067936538 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.6358511175987503 * eta1 + 5.555088747533164 * eta2 - 14.078156877577733 * eta3)
             + chidiff1 * delta * (0.23205448591711159 * eta1 - 19.46049432345157 * eta2
                                  + 36.20685853857613 * eta3)) * sqroot
            + (1.1525594672495008 + 7.380126197972549 * eta1 - 17.51265776660515 * eta2
              - 976.9940395257111 * eta3 + 8880.536804741967 * eta4 - 30849.228936891763 * eta5
              + 38785.53683146884 * eta6) * sqroot
            + chidiff1 * delta * (1.904350804857431 * eta1 - 25.565242391371093 * eta2
                                 + 80.67120303906654 * eta3) * S1 * sqroot
            + S1 * (
                0.785171689871352 * (-0.4634745514643032 + 18.70856733065619 * eta1
                                    - 167.9231114864569 * eta2 + 744.7699462372949 * eta3
                                    - 1115.008825153004 * eta4)
                + 0.13469300326662165 * (-2.7311391326835133 + 72.17373498208947 * eta1
                                        - 483.7040402103785 * eta2 + 1136.8367114738041 * eta3
                                        - 472.02962341590774 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.06190013067931406 + 0.1928897813606222 * eta + 1.9024723168424225 * eta2
            - 15.988716302668415 * eta3 + 35.21461767354364 * eta4
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.011454874900772544 + 0.044702230915643903 * S
            + eta * (0.6600413908621988 + 0.12149520289658673 * S - 0.4482406547006759 * S2)
            + 0.07327810908370004 * S2
            + eta2 * (-2.1705970511116486 - 0.6512813450832168 * S + 1.1237234702682313 * S2)
        )

        uneqSpin = (
            0.4766851579723911 * pWF['dchi'] * (1. - 15.950025762198988 * eta2) * eta2
            + 0.127900699645338 * pWF['dchi'] ** 2 * (1. - 15.79329306044842 * eta2) * eta2
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (0.5697308729057493 * eta1 + 8.895576813118867 * eta2
                                - 34.98399465240273 * eta3)
             + chidiff2 * (1.6370346538130884 * eta1 - 14.597095790380884 * eta2
                          + 33.182723737396294 * eta3)) * sqroot
            + (5.2601381002242595 - 3.557926105832778 * eta1 - 138.9749850448088 * eta2
              + 603.7453704122706 * eta3 - 923.5495700703648 * eta4) * sqroot
            + S1 * (
                -0.41839636169678796 * (5.143510231379954 + 104.62892421207803 * eta1
                                       - 4232.508174045782 * eta2 + 50694.024801783446 * eta3
                                       - 283097.33358214336 * eta4 + 758333.2655404843 * eta5
                                       - 788783.0559069642 * eta6)
                - 0.05653522061311774 * (5.605483124564013 + 694.00652410087 * eta1
                                        - 17551.398321516353 * eta2 + 165236.6480734229 * eta3
                                        - 761661.9645651339 * eta4 + 1.7440315410044065e6 * eta5
                                        - 1.6010489769238676e6 * eta6) * S1
                - 0.023693246676754775 * (16.437107575918503 - 2911.2154288136217 * eta1
                                         + 89338.32554683842 * eta2 - 1.0803340811860575e6 * eta3
                                         + 6.255666490084672e6 * eta4 - 1.7434160932177313e7 * eta5
                                         + 1.883460394974573e7 * eta6) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        S2 = S ** 2
        S3 = S2 * S

        noSpin = (
            0.08406011695496626 - 0.1469952725049322 * eta + 0.2997223283799925 * eta2
            - 1.2910560244510723 * eta3
        )

        eqSpin = (
            (0.023924074703897662 + 0.26110236039648027 * eta - 1.1536009170220438 * eta2) * S
            + (0.04479727299752669 - 0.1439868858871802 * eta + 0.05736387085230215 * eta2) * S2
            + (0.06028104440131858 - 0.4759412992529712 * eta + 1.1090751649419717 * eta2) * S3
        )

        uneqSpin = (
            0.10346324686812074 * pWF['dchi'] ** 2 * (1. - 16.135903382018213 * eta2) * eta2
            + 0.2648241309154185 * pWF['dchi'] * eta2 * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        S3 = S1 * S2
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.8318312659717388 * eta1 + 7.6541168007977864 * eta2 - 16.648660653220123 * eta3)
             + chidiff1 * delta * (2.214478316304753 * eta1 - 7.028104574328955 * eta2
                                  + 5.56587823143958 * eta3)) * sqroot
            + (3.173191054680422 + 6.707695566702527 * eta1 - 155.22519772642607 * eta2
              + 604.0067075996933 * eta3 - 876.5048298377644 * eta4) * sqroot
            + chidiff1 * delta * (4.749663394334708 * eta1 - 42.62996105525792 * eta2
                                 + 97.01712147349483 * eta3) * S1 * sqroot
            + S1 * (
                -0.2627203100303006 * (6.460396349297595 - 52.82425783851536 * eta1
                                      - 552.1725902144143 * eta2 + 12546.255587592654 * eta3
                                      - 81525.50289542897 * eta4 + 227254.37897941095 * eta5
                                      - 234487.3875219032 * eta6)
                - 0.008424003742397579 * (-109.26773035716548 + 15514.571912666677 * eta1
                                         - 408022.6805482195 * eta2 + 4.620165968920881e6 * eta3
                                         - 2.6446950627957724e7 * eta4 + 7.539643948937692e7 * eta5
                                         - 8.510662871580401e7 * eta6) * S1
                - 0.008830881730801855 * (-37.49992494976597 + 1359.7883958101172 * eta1
                                         - 23328.560285901796 * eta2 + 260027.4121353132 * eta3
                                         - 1.723865744472182e6 * eta4 + 5.858455766230802e6 * eta5
                                         - 7.756341721552802e6 * eta6) * S2
                - 0.027167813927224657 * (34.281932237450256 - 3312.7658728016568 * eta1
                                         + 84126.14531363266 * eta2 - 956052.0170024392 * eta3
                                         + 5.570748509263883e6 * eta4 - 1.6270212243584689e7 * eta5
                                         + 1.8855858173287075e7 * eta6) * S3
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = (
            0.08212436946985402 - 0.025332770704783136 * eta - 3.2466088293309885 * eta2
            + 28.404235115663706 * eta3 - 111.36325359782991 * eta4 + 157.05954559045156 * eta5
        )

        eqSpin = S * (
            0.03488890057062679 + 0.039491331923244756 * S
            + eta * (-0.08968833480313292 - 0.12754920943544915 * S - 0.11199012099701576 * S2)
            + 0.034468577523793176 * S2
        )

        uneqSpin = 0.2062291124580944 * pWF['dchi'] * eta2 * delta

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (1.4739380748149558 * eta1 + 0.06541707987699942 * eta2
                                - 9.473290540936633 * eta3)
             + chidiff2 * (-0.3640838331639651 * eta1 + 3.7369795937033756 * eta2
                          - 8.709159662885131 * eta3)) * sqroot
            + (1.7335503724888923 + 12.656614578053683 * eta1 - 139.6610487470118 * eta2
              + 456.78649322753824 * eta3 - 599.2709938848282 * eta4) * sqroot
            + chidiff1 * delta * (2.3532739003216254 * eta1 - 21.37216554136868 * eta2
                                 + 53.35003268489743 * eta3) * S1 * sqroot
            + S1 * (
                -0.15782329022461472 * (6.0309399412954345 - 229.16361598098678 * eta1
                                       + 3777.477006415653 * eta2 - 31109.307191210424 * eta3
                                       + 139319.8239886073 * eta4 - 324891.4001578353 * eta5
                                       + 307714.3954026392 * eta6)
                - 0.03050157254864058 * (4.232861441291087 + 1609.4251694451375 * eta1
                                        - 51213.27604422822 * eta2 + 612317.1751155312 * eta3
                                        - 3.5589766538499263e6 * eta4 + 1.0147654212772278e7 * eta5
                                        - 1.138861230369246e7 * eta6) * S1
                - 0.026407497690308382 * (-17.184685557542196 + 744.4743953122965 * eta1
                                         - 10494.512487701073 * eta2 + 66150.52694069289 * eta3
                                         - 184787.79377504133 * eta4 + 148102.4257785174 * eta5
                                         + 128167.89151782403 * eta6) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv3: version {InspAmpFlag} is not valid.")

    return total
