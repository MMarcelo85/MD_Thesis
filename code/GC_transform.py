def MoCa_to_MMSE(x, method='Van Steenoven'):
    """
    Description
    ------------
    Function to convert MoCA total scores to MMSE total scores.
    Based on the following publication:
    Kim, Ryul & Kim, Han-Joon & Kim, Aryun & Jang, Mi-Hee & Kim, Hyun & Jeon, Beomseok. (2018).
    https://www.researchgate.net/publication/322403155_Validation_of_the_Conversion_between_the_Mini-Mental_State_Examination_and_Montreal_Cognitive_assessment_in_Korean_Patients_with_Parkinson%27s_Disease

    Parameters
    ----------
    x: array or list like of MoCA total scores.
    method (str): Options are Van Steenoven (default) or Lawton.

    Returns
    -------
    Array of converted scores
    """
    import numpy as np
    # Equivalences
    x = np.array(x)
    minimum = np.sort(np.unique(x))[0]
    maximum = np.sort(np.unique(x))[-1]
    # VanS = [6, 9, 11 ,12, 13, 14, 15, 15, 16, 17, 18, 18, 19, 20, 21, 22, 22, 23, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 30]
    # Lawt = [1, 2, 4, 10, 13, 14, 15, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30]

    # Check that array is Ok
    if minimum < 0:
        return (f"Check your data, values < 0 in array\Minimun value = {minimum}")
    elif maximum > 30:
        return (f"Check your data, values > 30 in array\nMaximum value = {maximum}")
    else:
        # print("Min >=0 & Max <=30")
        pass

    converted = []
    # Convert to MMSE Van Steenoven
    if method=='Van Steenoven':
        for i in x:
            if np.isnan(i) == True:
                converted.append(np.nan)
            elif ( i == 0):
                converted.append(0)
            elif (i>=1) & (i<2):
                converted.append(6)
            elif (i>=2) & (i<3):
                converted.append(9)
            elif (i>=3) & (i<4):
                converted.append(11)
            elif (i>=4) & (i<5):
                converted.append(12)
            elif (i>=5) & (i<6):
                converted.append(13)
            elif (i>=6) & (i<7):
                converted.append(14)
            elif (i>=7) & (i<9):
                converted.append(15)
            elif (i>=9) & (i<10):
                converted.append(16)
            elif (i>=10) & (i<11):
                converted.append(17)
            elif (i>=11) & (i<13):
                converted.append(18)
            elif (i>=13) & (i<14):
                converted.append(19)
            elif (i>=14) & (i<15):
                converted.append(20)
            elif (i>=15) & (i<16):
                converted.append(21)
            elif (i>=16) & (i<18):
                converted.append(22)
            elif (i>=18) & (i<19):
                converted.append(23)
            elif (i>=19) & (i<20):
                converted.append(24)
            elif (i>=20) & (i<21):
                converted.append(25)
            elif (i>=21) & (i<23):
                converted.append(26)
            elif (i>=23) & (i<24):
                converted.append(27)
            elif (i>=24) & (i<26):
                converted.append(28)
            elif (i>=26) & (i<28):
                converted.append(29)
            elif (i>=28) & (i<=30):
                converted.append(30)
            else:
                converted.append(-1)
                print("Warning. Something happend: -1 appended")

    elif method=='Lawton':
        for i in x:
            if np.isnan(i) == True:
                converted.append(np.nan)
            elif (i == 0):
                converted.append(0)
            elif (i>=1) & (i<2):
                converted.append(1)
            elif (i>=2) & (i<3):
                converted.append(2)
            elif (i>=3) & (i<4):
                converted.append(4)
            elif (i>=4) & (i<5):
                converted.append(10)
            elif (i>=5) & (i<6):
                converted.append(13)
            elif (i>=6) & (i<7):
                converted.append(14)
            elif (i>=7) & (i<8):
                converted.append(15)
            elif (i>=8) & (i<9):
                converted.append(16)
            elif (i>=9) & (i<10):
                converted.append(17)
            elif (i>=10) & (i<12):
                converted.append(18)
            elif (i>=12) & (i<13):
                converted.append(19)
            elif (i>=13) & (i<15):
                converted.append(20)
            elif (i>=15) & (i<16):
                converted.append(21)
            elif (i>=16) & (i<18):
                converted.append(22)
            elif (i>=18) & (i<19):
                converted.append(23)
            elif (i>=19) & (i<21):
                converted.append(24)
            elif (i>=21) & (i<22):
                converted.append(25)
            elif (i>=22) & (i<24):
                converted.append(26)
            elif (i>=24) & (i<25):
                converted.append(27)
            elif (i>=25) & (i<27):
                converted.append(28)
            elif (i>=27) & (i<29):
                converted.append(29)
            elif (i>=29) & (i<=30):
                converted.append(30)
            else:
                converted.append(-1)
                print("Warning. Something happend: -1 appended")
    else:
        return "Valid methods are 'Van Steenoven' and 'Lawton'"

    return np.array(converted)

def MMSE_to_MoCA(x, method='Van Steenoven'):
    """
    Description
    ------------
    Function to convert MMSE total scores to MoCA total scores.
    Based on the following publication:
    Kim, Ryul & Kim, Han-Joon & Kim, Aryun & Jang, Mi-Hee & Kim, Hyun & Jeon, Beomseok. (2018).
    https://www.researchgate.net/publication/322403155_Validation_of_the_Conversion_between_the_Mini-Mental_State_Examination_and_Montreal_Cognitive_assessment_in_Korean_Patients_with_Parkinson%27s_Disease

    Parameters
    ----------
    x: array or list like of MMSE total scores.
    method (str): Options are Van Steenoven (default) or Lawton.

    Returns
    -------
    Array of converted scores
    """
    import numpy as np
    # Check that array is Ok
    if x.min() < 0:
        return  "Check your data, values < 0 in array"
    elif x.max() > 30:
        return "Check your data, values > 30 in array"
    else:
        pass

    converted = []
    # Convert to MoCA Van Steenoven
    if method=='Van Steenoven':
        for i in x:
            if np.isnan(i) == True:
                converted.append(np.nan)
            elif (i >= 0) & (i<7):
                converted.append(1)
            elif (i >= 7) & (i<11):
                converted.append(2)
            elif (i >= 11) & (i<12):
                converted.append(3)
            elif (i >= 12) & (i<13):
                converted.append(4)
            elif (i >= 13) & (i<14):
                converted.append(5)
            elif (i >= 14) & (i<15):
                converted.append(6)
            elif (i >= 15) & (i<16):
                converted.append(7.5)
            elif (i >= 16) & (i<17):
                converted.append(9)
            elif (i >= 17) & (i<18):
                converted.append(10)
            elif (i >= 18) & (i<19):
                converted.append(11.5)
            elif (i >= 19) & (i<20):
                converted.append(13)
            elif (i >= 20) & (i<21):
                converted.append(14)
            elif (i >= 21) & (i<22):
                converted.append(15)
            elif (i >= 22) & (i<23):
                converted.append(16.5)
            elif (i >= 23) & (i<24):
                converted.append(18)
            elif (i >= 24) & (i<25):
                converted.append(19)
            elif (i >= 25) & (i<26):
                converted.append(20)
            elif (i >= 26) & (i<27):
                converted.append(21.5)
            elif (i >= 27) & (i<28):
                converted.append(23)
            elif (i >= 28) & (i<29):
                converted.append(24.5)
            elif (i >= 29) & (i<30):
                converted.append(26.5)
            elif (i >= 30) & (i<31):
                converted.append(29)
            else:
                converted.append(-1)
                print("Warning. Something happend: -1 appended")

    elif method=='Lawton':
        for i in x:
            if np.isnan(i) == True:
                converted.append(np.nan)
            elif (i >= 0) & (i<2):
                converted.append(1)
            elif (i >= 2) & (i<3):
                converted.append(2)
            elif (i >= 3) & (i<5):
                converted.append(3)
            elif (i >= 5) & (i<11):
                converted.append(4)
            elif (i >= 11) & (i<14):
                converted.append(5)
            elif (i >= 14) & (i<15):
                converted.append(6)
            elif (i >= 15) & (i<16):
                converted.append(7)
            elif (i >= 16) & (i<17):
                converted.append(8)
            elif (i >= 17) & (i<18):
                converted.append(9)
            elif (i >= 18) & (i<19):
                converted.append(10.5)
            elif (i >= 19) & (i<20):
                converted.append(12)
            elif (i >= 20) & (i<21):
                converted.append(13.5)
            elif (i >= 21) & (i<22):
                converted.append(15)
            elif (i >= 22) & (i<23):
                converted.append(16.5)
            elif (i >= 23) & (i<24):
                converted.append(18)
            elif (i >= 24) & (i<25):
                converted.append(19.5)
            elif (i >= 25) & (i<26):
                converted.append(21)
            elif (i >= 26) & (i<27):
                converted.append(22.5)
            elif (i >= 27) & (i<28):
                converted.append(24)
            elif (i >= 28) & (i<29):
                converted.append(25.5)
            elif (i >= 29) & (i<30):
                converted.append(27.5)
            elif (i >= 30) & (i<31):
                converted.append(29.5)
            else:
                converted.append(-1)
                print("Warning. Something happend: -1 appended")
    else:
        return "Valid methods are 'Van Steenoven' and 'Lawton'"

    return np.array(converted)

def ACEIII_to_MMSE(x):
    """
    Description
    ------------
    Function to convert ACE-III total scores to MMSE total scores.

    Based on the following publication:

    Matías-Guiu, J. A., Pytel, V., Cortés-Martínez, A., Valles-Salgado, M.,
    Rognoni, T., Moreno-Ramos, T., & Matías-Guiu, J. (2017).
    Conversion between Addenbrooke’s Cognitive Examination III and Mini-Mental State Examination.
    International Psychogeriatrics, 1–7. doi:10.1017/s104161021700268x

    Parameters
    ----------
    x: array or list like of ACE-III total scores.

    Returns
    -------
    Array of converted scores
    """
    import numpy as np
    x = np.array(x)
    # Check that array is Ok
    if x.min() < 0:
        return  "Check your data, values < 0 in array"
    elif x.max() > 100:
        return "Check your data, values > 100 in array"
    else:
        pass

    converted = []
    # Convert to MMSE

    for i in x:
        if np.isnan(i) == True:
            converted.append(np.nan)
        elif i<=7:
            converted.append(0)
        elif (i>7) & (i <= 10):
            converted.append(1)
        elif (i>10) & (i <= 12):
            converted.append(2)
        elif (i>12) & (i <= 14):
            converted.append(3)
        elif (i>14) & (i <= 16):
            converted.append(4)
        elif (i > 16) & (i<=17):
            converted.append(5)
        elif (i>17) & (i <= 19):
            converted.append(6)
        elif (i>19) & (i <= 21):
            converted.append(7)
        elif (i >= 22) & (i<23):
            converted.append(8)
        elif (i>=23) & (i <= 24):
            converted.append(9)
        elif (i>24) & (i <= 26):
            converted.append(10)
        elif (i>26) & (i <= 28):
            converted.append(11)
        elif (i>28) & (i <= 31):
            converted.append(12)
        elif (i>31) & (i <= 33):
            converted.append(13)
        elif (i>33) & (i <= 36):
            converted.append(14)
        elif (i>36) & (i <= 38):
            converted.append(15)
        elif (i>38) & (i <= 41):
            converted.append(16)
        elif (i>41) & (i <= 44):
            converted.append(17)
        elif (i>44) & (i <= 48):
            converted.append(18)
        elif (i>48) & (i <= 51):
            converted.append(19)
        elif (i>51) & (i <= 55):
            converted.append(20)
        elif (i>55) & (i <=59 ):
            converted.append(21)
        elif (i>59) & (i <=63):
            converted.append(22)
        elif (i>63) & (i <=68):
            converted.append(23)
        elif (i>68) & (i <=73):
            converted.append(24)
        elif (i>73) & (i <=78):
            converted.append(25)
        elif (i>78) & (i <=83):
            converted.append(26)
        elif (i>83) & (i <=88):
            converted.append(27)
        elif (i>88) & (i <=93):
            converted.append(28)
        elif (i>93) & (i <=98):
            converted.append(29)
        elif (i>98) & (i <=100):
            converted.append(30)
        else:
            return ('WARNING: Check function!')
    return np.array(converted)

def MMSE_to_ACEIII(x):
    """
    Description
    ------------
    Function to convert MMSE total scores to ACE-III total scores.

    Based on the following publication:

    Matías-Guiu, J. A., Pytel, V., Cortés-Martínez, A., Valles-Salgado, M.,
    Rognoni, T., Moreno-Ramos, T., & Matías-Guiu, J. (2017).
    Conversion between Addenbrooke’s Cognitive Examination III and Mini-Mental State Examination.
    International Psychogeriatrics, 1–7. doi:10.1017/s104161021700268x

    Parameters
    ----------
    x: array or list like of MMSE total scores.

    Returns
    -------
    Array of converted scores
    """
    import numpy as np
    x = np.array(x)
    # Check that array is Ok
    if x.min() < 0:
        return  "Check your data, values < 0 in array"
    elif x.max() > 100:
        return "Check your data, values > 100 in array"
    else:
        pass

    converted = []
    # Convert to ACE-III

    for i in x:
        if np.isnan(i) == True:
            converted.append(np.nan)
        elif (i==0):
            converted.append(4)
        elif i == 1:
            converted.append(9)
        elif i == 2:
            converted.append(11.5)
        elif i == 3:
            converted.append(13.5)
        elif i == 4:
            converted.append(15.5)
        elif i == 5:
            converted.append(17)
        elif i == 6:
            converted.append(18.5)
        elif i == 7:
            converted.append(20.5)
        elif i == 8:
            converted.append(22)
        elif i == 9:
            converted.append(23.5)
        elif i == 10:
            converted.append(25.5)
        elif i == 11:
            converted.append(27.5)
        elif i == 12:
            converted.append(30)
        elif i == 13:
            converted.append(32.5)
        elif i == 14:
            converted.append(35)
        elif i == 15:
            converted.append(37.5)
        elif i == 16:
            converted.append(40)
        elif i == 17:
            converted.append(43)
        elif i == 18:
            converted.append(46.5)
        elif i == 19:
            converted.append(50)
        elif i == 20:
            converted.append(53.5)
        elif i == 21:
            converted.append(57.5)
        elif i == 22:
            converted.append(61.5)
        elif i == 23:
            converted.append(66)
        elif i == 24:
            converted.append(71)
        elif i == 25:
            converted.append(76)
        elif i == 26:
            converted.append(81)
        elif i == 27:
            converted.append(86)
        elif i == 28:
            converted.append(91)
        elif i == 29:
            converted.append(96)
        elif i == 30:
            converted.append(99.5)
        elif np.isnan(i) == True:
            converted.append(np.nan)
        else:
            return ('WARNING: Check function!')
    return np.array(converted)
