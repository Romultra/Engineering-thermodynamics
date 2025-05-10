import numpy as np
from apdx_functions import get_apdx_7, get_apdx_1
from solver_helper_functions import known, unknown, equalize, count_nans

# Otto cycle solver
def define_empty_variables():
    """
    Initializes the variables for the Otto cycle simulation.
    
    Returns:
    list: A list of dictionaries containing the initialized variables.
    """
    # Initialize all variables to NaN
    variables = [
        {
        'r': np.nan,  # Compression ratio (dimensionless)
        'R': np.nan,  # Specific gas constant (kJ/(kg·K))
        'gamma': np.nan,  # Heat capacity ratio (dimensionless)
        'Qh': np.nan,  # Heat input (kJ/kg)
        'Qc': np.nan,  # Heat rejection (kJ/kg)
        'Wi': np.nan,  # Work input (kJ/kg)
        'Wo': np.nan,  # Work output (kJ/kg)
        'W': np.nan,  # Work (kJ/kg)
        'n': np.nan,  # Efficiency (dimensionless)
        },
        {
            'T1': np.nan,  # K
            'P1': np.nan,  # kPa
            'V1': np.nan,  # m³/kg
            'S1': np.nan,  # kJ/kg·K
            'h1': np.nan,  # kJ/kg
            'u1': np.nan  # kJ/kg
        },
        {
            'T2': np.nan,  # K
            'P2': np.nan,  # kPa
            'V2': np.nan,  # m³/kg
            'S2': np.nan,  # kJ/kg·K
            'h2': np.nan,  # kJ/kg
            'u2': np.nan  # kJ/kg
        },
        {
            'T3': np.nan,  # K
            'P3': np.nan,  # kPa
            'V3': np.nan,  # m³/kg
            'S3': np.nan,  # kJ/kg·K
            'h3': np.nan,  # kJ/kg
            'u3': np.nan  # kJ/kg
        },
        {
            'T4': np.nan,  # K
            'P4': np.nan,  # kPa
            'V4': np.nan,  # m³/kg
            'S4': np.nan,  # kJ/kg·K
            'h4': np.nan,  # kJ/kg
            'u4': np.nan  # kJ/kg
        }
    ]

    return variables

def system_variables(vars, cold_air_standard=False):
    """
    Calculates all the system-level variables based on the known variables.
    
    Parameters:
    vars (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    # ----------------------------------------------------
    # A) r RELATIONS
    # ----------------------------------------------------
    if unknown(vars[0]['r']):
        # 1) If V1, V2 known => r = V1 / V2
        if known(vars[1]['V1']) and known(vars[2]['V2']):
            vars[0]['r'] = vars[1]['V1'] / vars[2]['V2']

        # 2) If P1, P2 known => r = (P2 / P1)^(1/gamma)
        elif known(vars[1]['P1']) and known(vars[2]['P2']):
            vars[0]['r'] = (vars[2]['P2'] / vars[1]['P1'])**(1 / vars[0]['gamma'])

        # 3) If T1, T2 known => r = (T2 / T1)^(1 / (gamma - 1))
        elif known(vars[1]['T1']) and known(vars[2]['T2']):
            vars[0]['r'] = (vars[2]['T2'] / vars[1]['T1'])**(1 / (vars[0]['gamma'] - 1))
    
    # ----------------------------------------------------
    # C) Qh and Qc RELATIONS
    # ----------------------------------------------------
    if unknown(vars[0]['Qh']):
        # 1) If h3, h2 known => Qh = h3 - h2
        if known(vars[3]['h3']) and (known(vars[2]['h2']) and not cold_air_standard):
            vars[0]['Qh'] = vars[3]['h3'] - vars[2]['h2']
    
    if unknown(vars[0]['Qc']):        
        # 1) If u4, u1 known => Qc = u4 - u1
        if known(vars[4]['u4']) and (known(vars[1]['u1']) and not cold_air_standard):
            vars[0]['Qc'] = vars[4]['u4'] - vars[1]['u1']
    
    # ----------------------------------------------------
    # D) Wi and Wo RELATIONS
    # ----------------------------------------------------
    if unknown(vars[0]['Wi']):
        # 1) If u2, u1 known => Wi = u1 - u2
        if known(vars[1]['u1']) and known(vars[2]['u2']):
            vars[0]['Wi'] = vars[1]['u1'] - vars[2]['u2']
    
    if unknown(vars[0]['Wo']):
        # 1) If P2, V2, V3, u3, u4 known => Wo = P2(V3-V2) + (u3 - u4)
        if (known(vars[2]['P2']) and 
            known(vars[2]['V2']) and 
            known(vars[3]['V3']) and 
            known(vars[3]['u3']) and 
            known(vars[4]['u4'])):
            vars[0]['Wo'] = vars[2]['P2'] * (vars[3]['V3'] - vars[2]['V2']) + (vars[3]['u3'] - vars[4]['u4'])
    
    # ----------------------------------------------------
    # E) W and n (efficiency) RELATIONS
    # ----------------------------------------------------
    if unknown(vars[0]['W']):
        # 1) If Qh, Qc known => W = Qh - Qc
        if known(vars[0]['Qh']) and known(vars[0]['Qc']):
            vars[0]['W'] = vars[0]['Qh'] - vars[0]['Qc']

        # 2) If Wo, Wi known => W = Wo - Wi
        elif known(vars[0]['Wo']) and known(vars[0]['Wi']):
            vars[0]['W'] = vars[0]['Wo'] - vars[0]['Wi']
    
    if unknown(vars[0]['n']):
        # 1) If W, Qh known => n = Wo - Wi / Qh = W / Qh
        if known(vars[0]['W']) and known(vars[0]['Qh']):
            vars[0]['n'] = vars[0]['W'] / vars[0]['Qh']

    return vars

def step_1(vars):
    """
    Calculates all the variables at step 1 that are possible to get from the currently known variables.
    
    Parameters:
    vars (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    r     = vars[0]['r']           # compression ratio
    Rg    = vars[0]['R']           # specific gas constant (kJ/kg.K)
    gamma = vars[0]['gamma']       # heat capacity ratio
    # other cycle-level variables as needed

    T1 = vars[1]['T1']
    P1 = vars[1]['P1']
    V1 = vars[1]['V1']
    S1 = vars[1]['S1']
    h1 = vars[1]['h1']
    u1 = vars[1]['u1']

    T2 = vars[2]['T2']
    P2 = vars[2]['P2']
    V2 = vars[2]['V2']
    S2 = vars[2]['S2']
    # h2, u2 if needed

    T4 = vars[4]['T4']
    P4 = vars[4]['P4']
    V4 = vars[4]['V4']
    S4 = vars[4]['S4']
    # h4, u4 if needed

    # Step 1: Calculate variables based on known values

    # ----------------------------------------------------
    # A) V1 RELATIONS
    # ----------------------------------------------------
    if unknown(V1):
        # 1) If volume at state 4 is known and process 4->1 is isochoric => V1 = V4
        if known(V4):
            V1 = V4

        # 2) If V2 is known and we know compression ratio r => V2 = V1 / r => V1 = r * V2
        elif known(V2) and known(r):
            V1 = r * V2

        # 3) If T1, P1, R known => V1 = (R * T1) / P1
        elif known(T1) and known(P1) and known(Rg):
            V1 = (Rg * T1) / P1

        vars[1]['V1'] = V1

    # ----------------------------------------------------
    # B) T1 RELATIONS
    # ----------------------------------------------------
    if unknown(T1):
        # 1) Isentropic from 1->2 => T2 = T1 * (r)^(gamma - 1) => T1 = T2 / r^(gamma - 1)
        if known(T2) and known(r) and known(gamma):
            T1 = T2 / (r**(gamma - 1))

        # 2) If we know T4,P4,P1 and 4->1 is isochoric => P1/P4 = T1/T4 => T1 = T4 * (P1/P4)
        if (unknown(T1) and 
            known(T4) and
            known(P4) and
            known(P1)):
            T1 = T4 * (P1 / P4)

        # 3) If we know internal energy u1 => invert via table or ideal‐gas formula
        if (unknown(T1) and known(u1)):
            T1 = get_apdx_7('u', u1, 'T')

        # 4) If we know enthalpy h1 => invert via table or formula
        if (unknown(T1) and known(h1)):
            T1 = get_apdx_7('h', h1, 'T')

        #  5) If we know s1, P1 => T1 from s,P
        # if (unknown(T1) and known(S1) and known(P1)):
        #     T_test = T_from_sP(S1, P1)
        #     if known(T_test):
        #         T1 = T_test
    
    # ----------------------------------------------------
    # C) P1 RELATIONS
    # ----------------------------------------------------
    if unknown(P1):
        # 1) From isentropic 1->2 => P2 = P1 * r^gamma => P1 = P2 / r^gamma
        if known(P2) and known(r) and known(gamma):
            P1 = P2 / (r**gamma)

        # 2) If T1, V1, R known => P1 = R*T1 / V1
        if (unknown(P1) and 
            known(T1) and 
            known(V1) and 
            known(Rg)):
            P1 = (Rg * T1) / V1

        # 3) If T4,P4,T1 known and 4->1 is isochoric => P1 / T1 = P4 / T4 => P1 = (T1/T4)*P4
        if (unknown(P1) and 
            known(T1) and
            known(T4) and
            known(P4)):
            P1 = (T1 / T4)*P4

        # 4) Possibly from s1, T1 => P_from_sT(...) if you have a table approach
        #    or from h1, v1, etc. 
        vars[1]['P1'] = P1

    # ----------------------------------------------------
    # E) h1 RELATIONS
    # ----------------------------------------------------
    if unknown(h1):
        # 1) If T1 is known => h1 = h(T1)
        if known(T1):
            h1 = get_apdx_7('T', T1, 'h')
        vars[1]['h1'] = h1

    # ----------------------------------------------------
    # F) u1 RELATIONS
    # ----------------------------------------------------
    if unknown(u1):
        # 1) If T1 is known => u1 = u(T1)
        if known(T1):
            u1 = get_apdx_7('T', T1, 'u')
        vars[1]['u1'] = u1
    
    return vars

def step_2(vars):
    """
    Calculates all the variables at state 2 that are possible to get from 
    the currently known variables in a Otto cycle.
    
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
    list: Updated list of dictionaries with calculated variables for state 2.
    """

    # Unpack cycle-level variables
    r     = vars[0]['r']     # compression ratio = V1 / V2
    Rg    = vars[0]['R']     # specific gas constant, kJ/(kg·K)
    gamma = vars[0]['gamma']

    # Unpack state 1 (start of compression)
    T1 = vars[1]['T1']
    P1 = vars[1]['P1']
    V1 = vars[1]['V1']
    S1 = vars[1]['S1']
    h1 = vars[1]['h1']
    u1 = vars[1]['u1']

    # Unpack state 2 (end of compression)
    T2 = vars[2]['T2']
    P2 = vars[2]['P2']
    V2 = vars[2]['V2']
    S2 = vars[2]['S2']
    h2 = vars[2]['h2']
    u2 = vars[2]['u2']

    # Unpack state 3 (end of heat addition)
    T3 = vars[3]['T3']
    P3 = vars[3]['P3']
    V3 = vars[3]['V3']
    S3 = vars[3]['S3']
    h3 = vars[3]['h3']
    u3 = vars[3]['u3']

    # ----------------------------------------------------
    # A) V2 RELATIONS
    # ----------------------------------------------------
    if unknown(V2):
        # From isentropic 1->2 => V2 = V1 / r
        if known(V1) and known(r):
            V2 = V1 / r
        
        # From constant-volume 2->3 => V2 = V3
        elif known(V3):
            V2 = V3

        # If T2, P2, R known => V2 = (R*T2)/P2 (ideal-gas assumption)
        elif (known(T2) and 
            known(P2) and 
            known(Rg)):
            V2 = (Rg * T2) / P2

        vars[2]['V2'] = V2

    # ----------------------------------------------------
    # B) T2 RELATIONS
    # ----------------------------------------------------
    if unknown(T2):
        # 1) Isentropic 1->2 => T2 = T1 * (r)^(gamma - 1)
        if known(T1) and known(r) and known(gamma):
            T2 = T1 * (r ** (gamma - 1))

        # 2) If we know T3, P2, P3 and 2->3 is constant V => for an ideal gas, T3/T2 = P3/P2
        # => T2 = T3 * P2 / P3
        elif known(T3) and known(P2) and known(P3):
            T2 = T3 * (P2 / P3)

        # 3) If we know internal energy u2 => invert via table
        elif known(u2):
            T2 = get_apdx_7('u', u2, 'T')

        # 4) If we know enthalpy h2 => invert via table
        elif known(h2):
            T2 = get_apdx_7('h', h2, 'T')

        vars[2]['T2'] = T2

    # ----------------------------------------------------
    # C) P2 RELATIONS
    # ----------------------------------------------------
    if unknown(P2):
        # 1) Isentropic 1->2 => P2 = P1 * (r^gamma)
        if known(P1) and known(r) and known(gamma):
            P2 = P1 * (r ** gamma)

        # 2) If we know T2, T3, P3, and 2->3 is constant V => for an ideal gas, T3/T2 = P3/P2
        # => P2 = P3 * (T2 / T3)
        elif known(T2) and known(T3) and known(P3):
            P2 = P3 * (T2 / T3)

        # 3) If T2, V2, R known => P2 = (R*T2)/V2
        elif (known(T2) and
            known(V2) and
            known(Rg)):
            P2 = (Rg * T2) / V2

        vars[2]['P2'] = P2

    # ----------------------------------------------------
    # D) S2 RELATIONS
    # ----------------------------------------------------
    if unknown(S2):
        # 1) Isentropic 1->2 => S2 = S1
        if known(S1):
            S2 = S1
        """
        # 2) If T2,P2 known => S2 = get_apdx_7(...) or from s(T2,P2)
        if (unknown(S2) and 
            known(T2) and 
            known(P2)):
            # Example using APDX 7 with T for interpolation:
            # Then if you have P2 as well, you might do a more advanced approach. 
            # For demonstration, we'll just get s from T2 alone:
            S2 = get_apdx_7('T', T2, 's0')  
            # In many ideal-gas tables, 's0' is measured from a reference pressure, 
            # so you might need s2 = s0(T2) + R ln(Pref/P2). 
            # Adjust as needed for your setup.
        """
        vars[2]['S2'] = S2

    # ----------------------------------------------------
    # E) h2 RELATIONS
    # ----------------------------------------------------
    if unknown(h2):
        # 1) If T2 known => h2 = get_apdx_7('T', T2, 'h')
        if known(T2):
            h2 = get_apdx_7('T', T2, 'h')

        vars[2]['h2'] = h2

    # ----------------------------------------------------
    # F) u2 RELATIONS
    # ----------------------------------------------------
    if unknown(u2):
        # 1) If T2 known => u2 = get_apdx_7('T', T2, 'u')
        if known(T2):
            u2 = get_apdx_7('T', T2, 'u')

        vars[2]['u2'] = u2

    return vars

def step_3(vars):
    """
    Calculates all the variables at state 3 that are possible to get
    from the currently known variables (states 2 and 4) in a Otto cycle.
    
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
      list: updated list of dictionaries with the newly computed state‐3 variables.
    """

    # Unpack cycle-level data
    rc    = vars[0]['rc']    # cutoff ratio = V3 / V2
    Rg    = vars[0]['R']     # gas constant (kJ/(kg·K))
    gamma = vars[0]['gamma']

    # State 2
    T2 = vars[2]['T2']
    P2 = vars[2]['P2']
    V2 = vars[2]['V2']
    S2 = vars[2]['S2']
    h2 = vars[2]['h2']
    u2 = vars[2]['u2']

    # State 3
    T3 = vars[3]['T3']
    P3 = vars[3]['P3']
    V3 = vars[3]['V3']
    S3 = vars[3]['S3']
    h3 = vars[3]['h3']
    u3 = vars[3]['u3']

    # State 4
    T4 = vars[4]['T4']
    P4 = vars[4]['P4']
    V4 = vars[4]['V4']
    S4 = vars[4]['S4']
    h4 = vars[4]['h4']
    u4 = vars[4]['u4']

    # ----------------------------------------------------
    # A) V3 RELATIONS
    # ----------------------------------------------------
    if unknown(V3):
        # 1) If 2->3 is constant volume => V3 = V2
        if known(V2):
            V3 = V2

        # 2) If T3, P3, and R are known (ideal gas) => V3 = (R * T3) / P3
        elif (unknown(V3) and 
            known(T3) and
            known(P3) and
            known(Rg)):
            V3 = (Rg * T3) / P3

        # 3) If we know T4,P4, and the isentropic expansion 3->4:
        #    P4 = P3*(V3/V4)^(-gamma) or T4 = T3*(V3/V4)^(1-gamma).
        #    Usually you solve for T4 from T3, but you could invert for V3
        #    if T3 is known. That might be more complicated. 
        #    We won't do the full inversion here unless needed.
        vars[3]['V3'] = V3

    # ----------------------------------------------------
    # B) T3 RELATIONS
    # ----------------------------------------------------
    if unknown(T3):
        # 1) If 2->3 is constant pressure => P3 = P2
        #    and for an ideal gas, T3/T2 = V3/V2 = rc if cp is ~ constant
        #    => T3 = T2 * rc
        if known(T2) and known(rc):
            T3 = T2 * rc  # common Otto assumption with constant cp
        
        # 1) If we know T2, P2, P3 and 2->3 is constant V => for an ideal gas, T3/T2 = P3/P2
        # => T3 = T2 * (P3 / P2)
        elif known(T2) and known(P2) and known(P3):
            T3 = T2 * (P3 / P2)

        # 2) If we know T4 and the isentropic relation 3->4 => 
        #    T4 = T3 * (V3/V4)^(1-gamma) => T3 = T4*(V4/V3)^(gamma-1)
        if (unknown(T3) and 
            known(T4) and
            known(V4) and
            known(V3) and
            known(gamma)):
            T3 = T4 * ((V4 / V3) ** (gamma - 1))

        # 3) If we have u3 => invert via table
        if (unknown(T3) and known(u3)):
            T3 = get_apdx_7('u', u3, 'T')

        # 4) If we have h3 => invert via table
        if (unknown(T3) and known(h3)):
            T3 = get_apdx_7('h', h3, 'T')

        vars[3]['T3'] = T3

    # ----------------------------------------------------
    # C) P3 RELATIONS
    # ----------------------------------------------------
    if unknown(P3):
        # 1) If we know T2, T3, P2, and 2->3 is constant V => for an ideal gas, T3/T2 = P3/P2
        # => P3 = P2 * (T3 / T2)
        if known(T2) and known(T3) and known(P3):
            P3 = P2 * (T3 / T2)

        # 2) If 3->4 is isentropic => P4 = P3*(V3/V4)^gamma => 
        #    => P3 = P4 / (V3/V4)^gamma = P4*(V4/V3)^gamma
        elif (unknown(P3) and 
            known(P4) and
            known(V4) and
            known(V3) and
            known(gamma)):
            P3 = P4 * ((V4 / V3) ** gamma)

        # 3) If T3, V3, and R known => P3 = R*T3 / V3
        elif (unknown(P3) and
            known(T3) and
            known(V3) and
            known(Rg)):
            P3 = (Rg * T3) / V3

        vars[3]['P3'] = P3

    # ----------------------------------------------------
    # D) S3 RELATIONS
    # ----------------------------------------------------
    if unknown(S3):
        # 1) If 3->4 is isentropic => S3 = S4
        if known(S4):
            S3 = S4
        """
        # 2) If T3,P3 known => S3 = s(T3,P3) from table or formula
        if (unknown(S3) and
            known(T3) and
            known(P3)):
            # Example usage with APDX 7. Many tables store s0 for an arbitrary reference pressure:
            # S3 = s0(T3) + R ln(P_ref / P3) or similar. 
            # For illustration, we'll just do:
            S3 = get_apdx_7('T', T3, 's0')  
            # (You might adjust for actual P3 if your table requires it.)
        """
        vars[3]['S3'] = S3
        
    # ----------------------------------------------------
    # E) h3 RELATIONS
    # ----------------------------------------------------
    if unknown(h3):
        # 1) If T3 is known => h3 = get_apdx_7('T', T3, 'h')
        if known(T3):
            h3 = get_apdx_7('T', T3, 'h')

        vars[3]['h3'] = h3

    # ----------------------------------------------------
    # F) u3 RELATIONS
    # ----------------------------------------------------
    if unknown(u3):
        # 1) If T3 is known => u3 = get_apdx_7('T', T3, 'u')
        if known(T3):
            u3 = get_apdx_7('T', T3, 'u')

        vars[3]['u3'] = u3

    return vars

def step_4(vars):
    """
    Calculates all the variables at state 4 that are possible to get
    from the currently known variables (states 3 and 1) in a Otto cycle.
   
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
      list: updated vars with newly computed state-4 variables.
    """

    # Unpack cycle-level data
    Rg    = vars[0]['R']        # kJ/(kg·K)
    gamma = vars[0]['gamma']

    # State 1 (after 4, in the cycle loop)
    T1 = vars[1]['T1']
    P1 = vars[1]['P1']
    V1 = vars[1]['V1']
    # S1, h1, u1 if needed

    # State 3 (before 4 in the cycle)
    T3 = vars[3]['T3']
    P3 = vars[3]['P3']
    V3 = vars[3]['V3']
    S3 = vars[3]['S3']
    h3 = vars[3]['h3']
    u3 = vars[3]['u3']

    # State 4
    T4 = vars[4]['T4']
    P4 = vars[4]['P4']
    V4 = vars[4]['V4']
    S4 = vars[4]['S4']
    h4 = vars[4]['h4']
    u4 = vars[4]['u4']

    # ----------------------------------------------------
    # A) V4 RELATIONS
    # ----------------------------------------------------
    if unknown(V4):
        # 1) If process 4->1 is constant volume => V4 = V1
        if known(V1):
            V4 = V1

        # 2) If T4, P4, and R known => V4 = (R * T4) / P4 (ideal gas)
        if (unknown(V4) and
            known(T4) and
            known(P4) and
            known(Rg)):
            V4 = (Rg * T4) / P4

        # 3) If 3->4 is isentropic => P4 = P3*(V3/V4)^gamma  or T4= T3*(V3/V4)^(gamma-1)
        #    you could solve for V4 if T4 or P4 is known, but that's a bit more involved. 
        #    Example (only if T3,T4 are known):
        if (unknown(V4) and 
            known(T3) and
            known(T4) and
            known(V3) and
            known(gamma)):
            # T4 = T3*(V3/V4)^(gamma-1) => V4 = V3 * (T3 / T4)^(1/(gamma-1))
            ratio_exp = 1.0/(gamma - 1.0)
            V4 = V3 * ((T3 / T4) ** ratio_exp)

        vars[4]['V4'] = V4

    # ----------------------------------------------------
    # B) T4 RELATIONS
    # ----------------------------------------------------
    if unknown(T4):
        # 1) Isentropic from 3->4 => T4 = T3*(V3/V4)^(gamma - 1)
        if (known(T3) and 
            known(V3) and
            known(V4) and
            known(gamma)):
            T4 = T3 * ((V3 / V4) ** (gamma - 1))

        # 2) If 4->1 is isochoric => (P4/T4) = (P1/T1) => T4 = (P4/P1)*T1
        if (unknown(T4) and
            known(P4) and
            known(P1) and
            known(T1)):
            T4 = (P4 / P1) * T1

        # 3) If we have u4 => T4 = invert via table, get_apdx_7('u', u4, 'T')
        if (unknown(T4) and known(u4)):
            T4 = get_apdx_7('u', u4, 'T')

        # 4) If we have h4 => T4 = invert via table, get_apdx_7('h', h4, 'T')
        if (unknown(T4) and known(h4)):
            T4 = get_apdx_7('h', h4, 'T')

        vars[4]['T4'] = T4

    # ----------------------------------------------------
    # C) P4 RELATIONS
    # ----------------------------------------------------
    if unknown(P4):
        # 1) Isentropic 3->4 => P4 = P3 * (V3 / V4)^gamma
        if (known(P3) and 
            known(V3) and
            known(V4) and
            known(gamma)):
            P4 = P3 * ((V3 / V4) ** gamma)

        # 2) If T4, V4, R known => P4 = (R * T4) / V4
        if (unknown(P4) and
            known(T4) and
            known(V4) and
            known(Rg)):
            P4 = (Rg * T4) / V4

        # 3) If 4->1 is isochoric => P4 / T4 = P1 / T1 => P4 = (T4 / T1)* P1
        if (unknown(P4) and
            known(T4) and
            known(T1) and
            known(P1)):
            P4 = (T4 / T1)* P1

        vars[4]['P4'] = P4

    # ----------------------------------------------------
    # D) S4 RELATIONS
    # ----------------------------------------------------
    if unknown(S4):
        # 1) Isentropic 3->4 => S4 = S3
        if known(S3):
            S4 = S3
        """
        # 2) If T4,P4 known => S4 = s(T4,P4) from table or formula
        if (unknown(S4) and
            known(T4) and
            known(P4)):
            # For an ideal-gas table, we might do s0(T4) + R ln(Pref/P4), etc.
            # For demonstration, we just call get_apdx_7('T', T4, 's0'):
            S4 = get_apdx_7('T', T4, 's0')
        """
        vars[4]['S4'] = S4

    # ----------------------------------------------------
    # E) h4 RELATIONS
    # ----------------------------------------------------
    if unknown(h4):
        # 1) If T4 known => h4 = h(T4)
        if known(T4):
            h4 = get_apdx_7('T', T4, 'h')

        vars[4]['h4'] = h4

    # ----------------------------------------------------
    # F) u4 RELATIONS
    # ----------------------------------------------------
    if unknown(u4):
        # 1) If T4 known => u4 = u(T4)
        if known(T4):
            u4 = get_apdx_7('T', T4, 'u')

        vars[4]['u4'] = u4

    return vars

def solve_otto_cycle(variables, cold_air_standard=False, verbose=False):
    """
    Solves the Otto cycle using the known variables and calculates the unknown variables.
    
    Parameters:
    variables (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    if unknown(variables[0]['gamma']):
        variables[0]['gamma'] = get_apdx_1('Air', 'gamma')
    if unknown(variables[0]['R']):
        variables[0]['R'] = get_apdx_1('Air', 'R')
    
    previous_nan_count = count_nans(variables)
    counter = 0

    if verbose:
        print(f"Starting with {previous_nan_count} unknowns")
    
    while True:
        counter += 1

        # Calculate system-level variables such as compression ratio (r), cutoff ratio (rc), 
        # heat input (Qh), heat rejection (Qc), and work-related variables (Wi, Wo, W):
        variables = system_variables(variables, cold_air_standard)
        
        # Step 1:
        variables = step_1(variables)
        
        # Step 2:
        variables = step_2(variables)

        # Step 3:
        variables = step_3(variables)

        # Step 4:
        variables = step_4(variables)
        
        current_nan_count = count_nans(variables)

        if verbose:
            print(f"Iteration {counter}: {current_nan_count} unknowns")

        if current_nan_count == 0 or \
        current_nan_count >= previous_nan_count:
            break

        previous_nan_count = current_nan_count
    
    return variables