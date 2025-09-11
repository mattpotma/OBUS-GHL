"""
biometry_utils.py

Various utilities related to biometry.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import numpy as np
import pandas as pd

MISSING = -9999


def fill_missing_ga_values(df: pd.DataFrame) -> np.ndarray:
    """
    Fill missing values in the specified columns of the dataframe.

    Parameters:
        df: pd.DataFrame                Dataframe to fill missing values in.

    Returns:
        Tuple[np.ndarray, np.ndarray]:  Filled feature values, records of
                                          source of GA values
    """
    # get stratification features and biometrics
    ga = df['GA'].values
    ga[ga == ''] = str(MISSING)
    ega = df['ega'].values
    ega[ega == ''] = str(MISSING)
    crl = df['CRL'].values
    crl[crl == ''] = str(MISSING)
    bpd = df['BPD'].values
    bpd[bpd == ''] = str(MISSING)
    ac = df['AC'].values
    ac[ac == ''] = str(MISSING)
    hc = df['HC'].values
    hc[hc == ''] = str(MISSING)
    fl = df['FL'].values
    fl[fl == ''] = str(MISSING)

    # final returned feature array
    GA = ga.astype(float)
    # record of what GAs were available, missing, and computed
    source = np.full(GA.shape, fill_value='GA', dtype=object)

    # enumerate missing values
    missing = np.where(ga == str(MISSING))[0]
    source[missing] = 'missing'
    # run through each missing row
    for idx in missing:
        if ega[idx] != str(MISSING):
            # fill the missing GA with ega
            GA[idx] = float(ega[idx])
            source[idx] = 'ega'
        elif crl[idx] != str(MISSING):
            # or compute GA from CRL
            GA[idx] = ga_from_crl(float(crl[idx]))
            source[idx] = 'crl'
        elif (bpd[idx] == str(MISSING)
              and hc[idx] != str(MISSING)
              and ac[idx] != str(MISSING)
              and fl[idx] != str(MISSING)):
            # if all biometrics are missing
            GA[idx] = None
            source[idx] = 'missing'
        else:
            # compute GA from biometrics
            GA[idx] = ga_from_biometrics(bpd[idx],
                                         ac[idx],
                                         hc[idx],
                                         fl[idx])
            source[idx] = 'hadlock'

    return GA, source


def ga_from_crl(crl: float) -> float:
    """
    Compute GA from CRL (Crown-rump length).

    Reference:
        Robinson HP, Fleming JE. A critical evaluation of sonar
        "crown-rump length" measurements. Br J Obstet Gynaecol.
        1975;82(9):702-710.

    Parameters:
        crl: float      Crown-rump length.

    Returns:
        float:          Estimated gestational age.
    """
    # constants for the regression
    ga = 8.052 * np.sqrt(crl) + 23.73
    return ga


def ga_from_biometrics(bpd: str, ac: str, hc: str, fl: str) -> float:
    """
    Compute GA from biometrics. Some values could be missing.

    Reference for 15 GA Hadlock formulas:
        Hadlock FP, Deter RL, Harrist RB, Park SK, Computer-assisted
        analysis of fetal age in the third trimester using multiple
        fetal growth parameters. J Clinic Ultrasound 11: 313-316, 1983.

    Reference for fetal weight using the Hadlock 4-component formula:
        Hadlock et al, Estimation of fetal weight with the use of head,
        body, and femur measurements--A prospective study, Am J Obs Gyn,
        151 (3), pp 333-337, 1985.

    Parameters:
        bpd: str      Biparietal diameter.
        ac: str       Abdominal circumference.
        hc: str       Head circumference.
        fl: str       Femur length.

    Returns:
        Optional[float]:          Estimated gestational age.
    """
    # tabulating available biometrics
    present = np.zeros(4, dtype=bool)

    if bpd != str(MISSING):
        present[0] = True
        bpd = float(bpd)
    if ac != str(MISSING):
        present[1] = True
        ac = float(ac)
    if hc != str(MISSING):
        present[2] = True
        hc = float(hc)
    if fl != str(MISSING):
        present[3] = True
        fl = float(fl)

    # priorities of equations (based on hadlock errors)
    priorities = np.array(list({
        0: 0,
        1: 1,
        2: 2,
        3: 6,
        4: 4,
        5: 5,
        6: 7,
        7: 8,
        8: 3,
        9: 10,
        10: 9,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
    }.values()))

    # alternatives, given the available biometrics
    alternative_codes = {
        0: None,
        1: None,
        2: None,
        3: [1, 2],
        4: None,
        5: [1, 4],
        6: [2, 4],
        7: [1, 2, 3, 4, 5, 6],
        8: None,
        9: [1, 8],
        10: [2, 8],
        11: [1, 2, 3, 8, 9, 10],
        12: [4, 8],
        13: [1, 4, 5, 8, 9, 12],
        14: [2, 4, 6, 8, 10, 12],
        15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    }

    # define the 15-way hadlock equations
    ga_hadlock_eqs = {
        1: ga_hadlock_1,
        2: ga_hadlock_2,
        3: ga_hadlock_3,
        4: ga_hadlock_4,
        5: ga_hadlock_5,
        6: ga_hadlock_6,
        7: ga_hadlock_7,
        8: ga_hadlock_8,
        9: ga_hadlock_9,
        10: ga_hadlock_10,
        11: ga_hadlock_11,
        12: ga_hadlock_12,
        13: ga_hadlock_13,
        14: ga_hadlock_14,
        15: ga_hadlock_15,
    }

    # calculate current code based on available biometrics
    code = np.dot(present, 2 ** np.arange(4, dtype=int)).sum()

    # Check if any biometrics were available
    if code == 0:
        # Don't return anything
        return None

    # get the priority of the current code
    priority = priorities[code]

    # see if alternatives for the current code are better
    alt_codes = alternative_codes[code]
    if alt_codes is not None:
        alt_priorities = priorities[alt_codes]
        if np.any(alt_priorities > priority):
            code = alt_codes[np.argmax(alt_priorities)]

    # compute GA based on the (best) code
    ga = ga_hadlock_eqs[code](bpd, ac, hc, fl)

    return ga


# #### HADLOCK EQUATIONS #### #

# noinspection PyUnusedLocal
def ga_hadlock_1(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(9.54 + 1.482*bpd + 0.1676*bpd*bpd)


# noinspection PyUnusedLocal
def ga_hadlock_2(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(8.14 + 0.753*ac + 0.0036*ac*ac)


# noinspection PyUnusedLocal
def ga_hadlock_3(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(9.57 + 0.524*ac + 0.122*bpd*bpd)


# noinspection PyUnusedLocal
def ga_hadlock_4(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(8.96 + 0.54*hc + 0.0003*hc*hc*hc)


# noinspection PyUnusedLocal
def ga_hadlock_5(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.32 + 0.009*hc*hc + 1.32*bpd + 0.00012*hc*hc*hc)


# noinspection PyUnusedLocal
def ga_hadlock_6(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.31 + 0.012*hc*hc + 0.385*ac)


# noinspection PyUnusedLocal
def ga_hadlock_7(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.58 + 0.005*hc*hc + 0.3635*ac + 0.02864*bpd*ac)


# noinspection PyUnusedLocal
def ga_hadlock_8(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.35 + 2.46*fl + 0.17*fl*fl)


# noinspection PyUnusedLocal
def ga_hadlock_9(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.50 + 0.197*bpd*fl + 0.95*fl + 0.73*bpd)


# noinspection PyUnusedLocal
def ga_hadlock_10(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.47 + 0.442*ac + 0.314*fl*fl - 0.0121*fl*fl*fl)


# noinspection PyUnusedLocal
def ga_hadlock_11(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.61 + 0.175*bpd*fl + 0.297*ac + 0.71*fl)


# noinspection PyUnusedLocal
def ga_hadlock_12(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(11.19 + 0.07*hc*fl + 0.263*hc)


# noinspection PyUnusedLocal
def ga_hadlock_13(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(11.38 + 0.07*hc*fl + 0.98*bpd)


# noinspection PyUnusedLocal
def ga_hadlock_14(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.33 + 0.031*hc*fl + 0.361*hc + 0.0298*ac*fl)


def ga_hadlock_15(bpd: float, ac: float, hc: float, fl: float) -> float:
    return 7.0*(10.85 + 0.06*hc*fl + 0.67*bpd + 0.168*ac)


def efw_hadlock_4component(bpd: float, ac: float, hc: float, fl: float) -> float:
    """
    Reference for fetal weight using the Hadlock 4-component formula:
        Hadlock et al, Estimation of fetal weight with the use of head,
        body, and femur measurements--A prospective study, Am J Obs Gyn,
        151 (3), pp 333-337, 1985.

    Parameters:
        bpd: float      Biparietal diameter.
        ac: float       Abdominal circumference.
        hc: float       Head circumference.
        fl: float       Femur length.

    Returns:
        float:          effective fetal weight.
    """
    log10_efw = 1.3596 \
              - 0.00386 * ac * fl \
              + 0.0064 * hc \
              + 0.00061 * bpd * ac \
              + 0.0424 * ac + 0.174 * fl
    efw = 10 ** log10_efw
    return efw
