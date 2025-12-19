import jax.numpy as jnp

# Constants (from LAL)
LAL_PI = jnp.pi
LAL_MSUN_SI = 1.98892e30
LAL_MTSUN_SI = 4.925491025543576e-06
LAL_G_SI = 6.67430e-11
LAL_C_SI = 299792458.0
LAL_GAMMA = 0.5772156649015329

LAL_ST4_ABSOLUTE_TOLERANCE = 1.0e-11
LAL_ST4_RELATIVE_TOLERANCE = 1.0e-9
LAL_NUM_ST4_VARIABLES = 14


XLALSimInspiralSpinDot_4PNS2CoeffAvg=0.5
XLALSimInspiralSpinDot_4PNS2OCoeffAvg=-1.5


def XLALSimInspiralTaylorT4wdot_7PNCoeff(eta):
    return (LAL_PI/12096.0) * (-13245.0 + 717350.0*eta + 731960.0*eta*eta)


def XLALSimInspiralTaylorT4wdot_6PNCoeff(eta):
    return ( 16447.322263/139.7088 - 1712./105.
                * LAL_GAMMA - 561.98689/2.17728 * eta + LAL_PI * LAL_PI
                * (16./3. + 451./48. * eta) + 541./896. * eta * eta
                - 5605./2592. * eta * eta * eta - 856./105. * jnp.log(16.) )

def XLALSimInspiralTaylorT4wdot_6PNLogCoeff(eta):
    return -(1712.0/105.0)


def XLALSimInspiralTaylorT4wdot_5PNCoeff(eta):
    return ( -(1.0/672.0) * LAL_PI * (4159.0 + 15876.0*eta) )  

def XLALSimInspiralTaylorT4wdot_4PNCoeff(eta):
    return (34103. + 122949.*eta + 59472.*eta*eta)/18144.
 
def XLALSimInspiralTaylorT4wdot_3PNCoeff(eta):
    return 4.0 * LAL_PI

def XLALSimInspiralTaylorT4wdot_2PNCoeff(eta):
    return ( -(1.0/336.0) * (743.0 + 924.0*eta) )

def XLALSimInspiralTaylorT4wdot_0PNCoeff(eta):
    return 96. / 5. * eta

def XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(mByM):

        return 6.*mByM*mByM*mByM*mByM * (12.-11.*mByM)


def XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(mByM):

        return mByM*mByM*mByM*mByM * (4421./56. - 12263./56.*mByM + 1893./4.*mByM*mByM - 661./2.*mByM*mByM*mByM)
 

def XLALSimInspiralLDot_3PNSOCoeff(mByM):
    return 0.5+1.5/mByM


def XLALSimInspiralSpinDot_6PNS2CoeffAvg(mByM):
    return XLALSimInspiralSpinDot_6PNS2Coeff(mByM)+0.5*(XLALSimInspiralSpinDot_6PNS2nCoeff(mByM)+XLALSimInspiralSpinDot_6PNS2vCoeff(mByM))

def XLALSimInspiralSpinDot_6PNS2Coeff(mByM):

    return -1.5 -mByM

def XLALSimInspiralSpinDot_6PNS2nCoeff(mByM):
     return 1.5 +2.*mByM+mByM*mByM


def XLALSimInspiralSpinDot_6PNS2vCoeff(mByM):
     return 1.5 +mByM


def XLALSimInspiralSpinDot_6PNS2OCoeffAvg(mByM):
    return -0.5*(XLALSimInspiralSpinDot_6PNS2nCoeff(mByM)+XLALSimInspiralSpinDot_6PNS2vCoeff(mByM))

def XLALSimInspiralSpinDot_6PNS1OCoeffAvg(mByM):
    return -0.5*(XLALSimInspiralSpinDot_6PNS1nCoeff(mByM)+XLALSimInspiralSpinDot_6PNS1vCoeff(mByM))


def XLALSimInspiralSpinDot_6PNS1nCoeff(mByM):
     return 3.5-3./mByM-.5*mByM*mByM

def XLALSimInspiralSpinDot_6PNS1vCoeff(mByM):
     return 3. -1.5*mByM-1.5/mByM

def XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(mByM):
    return -0.5*(XLALSimInspiralSpinDot_6PNQMSnCoeff(mByM)+XLALSimInspiralSpinDot_6PNQMSvCoeff(mByM));


def XLALSimInspiralSpinDot_6PNQMSnCoeff(mByM):
     return 3. * (.5/mByM + 1. - mByM - .5*mByM*mByM)

def XLALSimInspiralSpinDot_6PNQMSvCoeff(mByM):
     return 3. * (1./mByM -1.)



def XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(mByM):
    return 1.5 * (1. - 1./mByM)


def XLALSimInspiralSpinDot_5PNCoeff(mByM):
    return 9./8. - mByM/2. + 7.*mByM*mByM/12. - 7.*mByM*mByM*mByM/6. - mByM*mByM*mByM*mByM/24.

def XLALSimInspiralL_3PNSicoeffAvg(mByM):
    return -0.75-0.25/mByM


def XLALSimInspiralL_3PNSiLcoeffAvg(mByM):
    return -(1./3.+9./mByM)/4.


def XLALSimInspiralSpinDot_3PNCoeff(mByM):
    return 3./2. -mByM - mByM*mByM/2.



def XLALSimInspiralPNEnergy_3PNSOCoeff(mByM):
     return 2. / 3. + 2. / mByM


def XLALSimInspiralPNEnergy_4PNS1S2CoeffAvg(eta):
     return 1./eta


def XLALSimInspiralPNEnergy_4PNS1OS2OCoeffAvg(eta):
     return -3./eta


def XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(mByM):
     return .5/mByM/mByM

def XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(mByM):
     return -1.5/mByM/mByM


def XLALSimInspiralPNEnergy_5PNSOCoeff(mByM):
     return 5./3. + 3./mByM + 29.*mByM/9. + mByM*mByM/9.


def XLALSimInspiralPNEnergy_6PNS1S2CoeffAvg(eta):
    return 2./eta -11./6.

def XLALSimInspiralPNEnergy_6PNS1OS2OCoeffAvg(eta):
    return -11./3./eta + 2.3/1.8

def XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(mByM):
     return -1./(mByM*mByM) - 1./6./mByM -0.5

def XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(mByM):
     return 1.25/mByM/mByM + 1.25/mByM + 5./12.



def XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(mByM):
     return 6./(mByM*mByM) -1.5/mByM -1.1/1.8

def XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(mByM):
     return -3.75/mByM/mByM - 3.75/mByM - 1.25

def XLALSimInspiralPNEnergy_7PNSOCoeff(mByM):
     return -75./4. + 27./(4.*mByM) + 53.*mByM/2. + 67*mByM*mByM/6. + 17.*mByM*mByM*mByM/12. - mByM*mByM*mByM*mByM/12.



def XLALSimInspiralPNEnergy_10PNTidalCoeff(mByM):
     return -9.0 * mByM*mByM*mByM*mByM*(1.-mByM)

def XLALSimInspiralPNEnergy_12PNTidalCoeff(mByM):
     return 6./(mByM*mByM) -1.5/mByM -1.1/1.8



def XLALSimInspiralPNEnergy_2PNCoeff(eta):
     return -(0.75 + eta/12.0)
def XLALSimInspiralPNEnergy_4PNCoeff(eta):
     return -(27.0/8.0 - 19.0/8.0 * eta + 1./24.0 * eta*eta)


def XLALSimInspiralPNEnergy_6PNCoeff(eta):
     return -(67.5/6.4 - (344.45/5.76 - 20.5/9.6 * LAL_PI*LAL_PI) * eta + 15.5/9.6 * eta*eta + 3.5/518.4 * eta*eta*eta)