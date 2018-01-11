#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Set of functions to perform event selection for AlphaAnalysis
In all functions, the data frames are assumed to be in the KDST form
"""

import os
import functools
import time
from collections import namedtuple


import numpy             as np
import tables            as tb
import pandas            as pd
import matplotlib.pyplot as plt

import invisible_cities.core.core_functions as coref
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.reco.dst_functions  as dstf
import invisible_cities.io  .dst_io         as dstio
import invisible_cities.reco.corrections    as corrf

#------------------------------------------------------

def s1Filter(data,
             minCharge=-np.inf,    maxCharge=np.inf,
             minWidth=-np.inf,     maxWidth=np.inf,
             minStartTime=-np.inf, maxStartTime=np.inf):
    """ 
    Selection of S1 peaks based on peak charge, time width, start time

    Parameters:
    -----------
    data: panda dataframe
    minCharge, maxCharge: minimum and maximum for peak charge
    minWidth, maxWidth: minimum and maximum for peak time width
    minStartTime, maxStartTime: minimum and maximum for peak start time

    Returns:
    --------
    sel: boolean np.array, one entry per data row
    inputEntries: input number of entries
    outputEntries: output number of entries
    """
    selCharge    = coref.in_range(data.S1e, minCharge,    maxCharge)
    selWidth     = coref.in_range(data.S1w, minWidth,     maxWidth)
    selStartTime = coref.in_range(data.S1t, minStartTime, maxStartTime)

    sel          = np.logical_and(selCharge, selWidth)
    sel          = np.logical_and(sel,       selStartTime)
    inputEntries=len(sel)
    outputEntries=sum(sel)

    return sel, inputEntries, outputEntries

#------------------------------------------------------

def s2Filter(data,
             minCharge=-np.inf,    maxCharge=np.inf,
             minWidth=-np.inf,     maxWidth=np.inf,
             minStartTime=-np.inf, maxStartTime=np.inf):
    """ 
    Selection of S2 peaks based on peak charge, time width, start time

    Parameters:
    -----------
    data: panda dataframe
    minCharge, maxCharge: minimum and maximum for peak charge
    minWidth, maxWidth: minimum and maximum for peak time width
    minStartTime, maxStartTime: minimum and maximum for peak start time

    Returns:
    --------
    sel: boolean np.array, one entry per data row
    inputEntries: input number of entries
    outputEntries: output number of entries
    """
    selCharge    = coref.in_range(data.S2e, minCharge,    maxCharge)
    selWidth     = coref.in_range(data.S2w, minWidth,     maxWidth)
    selStartTime = coref.in_range(data.S2t, minStartTime, maxStartTime)

    sel          = np.logical_and(selCharge, selWidth)
    sel          = np.logical_and(sel,       selStartTime)
    inputEntries=len(sel)
    outputEntries=sum(sel)

    return sel, inputEntries, outputEntries

#------------------------------------------------------

def inclusiveAlphaFilter(data,
                         minS1Charge=-np.inf,    maxS1Charge=np.inf,
                         minS1Width=-np.inf,     maxS1Width=np.inf,
                         minS1StartTime=-np.inf, maxS1StartTime=np.inf,
                         minS2Charge=-np.inf,    maxS2Charge=np.inf,
                         minS2Width=-np.inf,     maxS2Width=np.inf,
                         minS2StartTime=-np.inf, maxS2StartTime=np.inf):
    """
    Inclusive alpha selection
    Goal: keep events with one and only one alpha-like S1, and one and only one alpha-like S2
    Implementation:
    First: select KDST entries where the S1 and S2 peaks are both alpha-like
    Second: only for entries where S1 and S2 peaks are both alpha-like, select ones where a given event number appears only once. This means there is a unique alpha candidate in that event
    Third: make an AND between the alpha-like peaks selection and the unique alpha selection

    Parameters:
    -----------
    data: panda dataframe
    minS1Charge, maxS1Charge: minimum and maximum for S1 charge
    minS1Width, maxS1Width: minimum and maximum for S1 time width
    minS1StartTime, maxS1StartTime: minimum and maximum for S1 start time
    minS2Charge, maxS2Charge: minimum and maximum for S2 charge
    minS2Width, maxS2Width: minimum and maximum for S2 time width
    minS2StartTime, maxS2StartTime: minimum and maximum for S2 start time

    Returns:
    --------
    selInclusiveAlphas: boolean np.array, one entry per data row
    inputEntries: input number of entries
    outputEntries: output number of entries
    """

    # First: select KDST entries where the S1 and S2 peaks are both alpha-like
    selS1 = s1Filter(data,
                     minS1Charge,    maxS1Charge,
                     minS1Width,     maxS1Width,
                     minS1StartTime, maxS1StartTime)[0]
    selS2 = s2Filter(data,
                     minS2Charge,    maxS2Charge,
                     minS2Width,     maxS2Width,
                     minS2StartTime, maxS2StartTime)[0]

    selAlphaLikePeaks = np.logical_and(selS1, selS2)

    # Second: only for entries where S1 and S2 peaks are both alpha-like, select ones where a given event number appears only once. This means there is a unique alpha candidate in that event
    selUniqueAlpha = np.logical_not(data[selAlphaLikePeaks].event.duplicated(keep=False))

    # Third: make an AND between the alpha-like peaks selection and the unique alpha selection
    # First fill an array with zeros, with as many elements as the inoput data frame
    selInclusiveAlphas=np.zeros_like(selAlphaLikePeaks)
    # Only for those entries satisfying the alpha-like peak selection, set them to true if they also satisfy the unique alpha selection
    selInclusiveAlphas[selAlphaLikePeaks]=selUniqueAlpha
    inputEntries=len(selInclusiveAlphas)
    outputEntries=sum(selInclusiveAlphas)
    
    return selInclusiveAlphas, inputEntries, outputEntries

          
