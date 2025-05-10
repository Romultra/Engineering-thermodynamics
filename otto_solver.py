import numpy as np
from apdx_functions import get_apdx_7, get_apdx_1, get_apdx_4
from solver_helper_functions import known, unknown, equalize, count_nans, display_tables

# Otto cycle solver
def define_empty_variables():
    """
    Initializes the variables for the Otto cycle simulation.
    
    Returns:
    list: A list of dictionaries containing the initialized variables.
    """
    # Initialize all variables to NaN
    variables = {
        'r': np.nan,  # Compression ratio (dimensionless)
        'R': np.nan,  # Specific gas constant (kJ/(kg·K))
        'cv': np.nan,  # Specific heat at constant volume (kJ/(kg·K))
        'gamma': np.nan,  # Heat capacity ratio (dimensionless)
        'Qh': np.nan,  # Heat input (kJ/kg)
        'Qc': np.nan,  # Heat rejection (kJ/kg)
        'Wi': np.nan,  # Work input (kJ/kg)
        'Wo': np.nan,  # Work output (kJ/kg)
        'W': np.nan,  # Work (kJ/kg)
        'n': np.nan,  # Efficiency (dimensionless)

        '1':{
            'T': np.nan,  # K
            'P': np.nan,  # kPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan  # kJ/kg
        },

        '2':{
            'T': np.nan,  # K
            'P': np.nan,  # kPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan  # kJ/kg
        },

        '3':{
            'T': np.nan,  # K
            'P': np.nan,  # kPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan  # kJ/kg
        },

        '4':{
            'T': np.nan,  # K
            'P': np.nan,  # kPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan  # kJ/kg
        }
    }

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
    if unknown(vars['r']):
        # 1) If v1, v2 known => r = v1 / v2
        if known(vars['1']['v']) and known(vars['2']['v']):
            vars['r'] = vars['1']['v'] / vars['2']['v']

        # 2) If P1, P2 known => r = (P2 / P1)^(1/gamma)
        elif known(vars['1']['P']) and known(vars['2']['P']):
            vars['r'] = (vars['2']['P'] / vars['1']['P'])**(1 / vars['gamma'])

        # 3) If T1, T2 known => r = (T2 / T1)^(1 / (gamma - 1))
        elif known(vars['1']['T']) and known(vars['2']['T']):
            vars['r'] = (vars['2']['T'] / vars['1']['T'])**(1 / (vars['gamma'] - 1))
    
    # ----------------------------------------------------
    # C) Qh and Qc RELATIONS
    # ----------------------------------------------------
    if cold_air_standard:
        if unknown(vars['Qh']):
            # 1) If cv, T3, T2 known => Qh = cv * (T3 - T2)
            if known(vars['cv']) and known(vars['3']['T']) and known(vars['2']['T']):
                vars['Qh'] = vars['cv'] * (vars['3']['T'] - vars['2']['T'])
            
        if unknown(vars['Qc']):        
            # 1) If cv, T4, T1 known => Qc = cv * (T4 - T1)
            if known(vars['cv']) and known(vars['4']['T']) and known(vars['1']['T']):
                vars['Qc'] = vars['cv'] * (vars['4']['T'] - vars['1']['T'])
    else:
        if unknown(vars['Qh']):
            # 1) If h3, h2 known => Qh = u3 - u2
            if known(vars['3']['u']) and known(vars['2']['u']):
                vars['Qh'] = vars['3']['u'] - vars['2']['u']
        
        if unknown(vars['Qc']):        
            # 1) If u4, u1 known => Qc = u4 - u1
            if known(vars['4']['u']) and known(vars['1']['u']):
                vars['Qc'] = vars['4']['u'] - vars['1']['u']
    
    # ----------------------------------------------------
    # D) Wi and Wo RELATIONS
    # ----------------------------------------------------
    if cold_air_standard:
        if unknown(vars['Wi']):
            # 1) If cv, T2, T1 known => Wi = cv * (T2 - T1)
            if known(vars['cv']) and known(vars['2']['T']) and known(vars['1']['T']):
                vars['Wi'] = vars['cv'] * (vars['2']['T'] - vars['1']['T'])
        
        if unknown(vars['Wo']):
            # 1) If cv, T3, T4 known => Wo = cv * (T3 - T4)
            if known(vars['cv']) and known(vars['3']['T']) and known(vars['4']['T']):
                vars['Wo'] = vars['cv'] * (vars['3']['T'] - vars['4']['T'])        

    else:
        if unknown(vars['Wi']):
            # 1) If h1, h2 known => Wi = h2 - h1
            if known(vars['1']['h']) and known(vars['2']['h']) :
                vars['Wi'] = vars['2']['h'] - vars['1']['h']

        if unknown(vars['Wo']):
            # 1) If h3, h4 known => Wo = h3 - h4
            if known(vars['3']['h']) and known(vars['4']['h']):
                vars['Wo'] = vars['3']['h'] - vars['4']['h']

    
    # ----------------------------------------------------
    # E) W and n (efficiency) RELATIONS
    # ----------------------------------------------------
    if unknown(vars['W']):
        # 1) If Qh, Qc known => W = Qh - Qc
        if known(vars['Qh']) and known(vars['Qc']):
            vars['W'] = vars['Qh'] - vars['Qc']

        # 2) If Wo, Wi known => W = Wo - Wi
        elif known(vars['Wo']) and known(vars['Wi']):
            vars['W'] = vars['Wo'] - vars['Wi']
    
    if unknown(vars['n']):
        # 1) If W, Qh known => n = Wo - Wi / Qh = W / Qh
        if known(vars['W']) and known(vars['Qh']):
            vars['n'] = vars['W'] / vars['Qh']

    return vars

def step_1(vars, cold_air_standard=False):
    """
    Calculates all the variables at step 1 that are possible to get from the currently known variables.
    
    Parameters:
    vars (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    r     = vars['r']           # compression ratio
    Rg    = vars['R']           # specific gas constant (kJ/kg.K)
    gamma = vars['gamma']       # heat capacity ratio
    # other cycle-level variables as needed

    T1 = vars['1']['T']
    P1 = vars['1']['P']
    v1 = vars['1']['v']
    s1 = vars['1']['s']
    h1 = vars['1']['h']
    u1 = vars['1']['u']

    T2 = vars['2']['T']
    P2 = vars['2']['P']
    v2 = vars['2']['v']
    s2 = vars['2']['s']
    # h2, u2 if needed

    T4 = vars['4']['T']
    P4 = vars['4']['P']
    v4 = vars['4']['v']
    s4 = vars['4']['s']
    # h4, u4 if needed

    # Step 1: Calculate variables based on known values

    # ----------------------------------------------------
    # A) v1 RELATIONS
    # ----------------------------------------------------
    if unknown(v1):
        # 1) If volume at state 4 is known and process 4->1 is isochoric => v1 = v4
        if known(v4):
            v1 = v4

        # 2) If v2 is known and we know compression ratio r => v2 = v1 / r => v1 = r * v2
        elif known(v2) and known(r):
            v1 = r * v2

        # 3) If T1, P1, R known => v1 = (R * T1) / P1
        elif known(T1) and known(P1) and known(Rg):
            v1 = (Rg * T1) / P1

        vars['1']['v'] = v1

    # ----------------------------------------------------
    # B) T1 RELATIONS
    # ----------------------------------------------------
    if unknown(T1):
        # 1) Isentropic from 1->2 => T2 = T1 * (r)^(gamma - 1) => T1 = T2 / r^(gamma - 1)
        if known(T2) and known(r) and known(gamma):
            T1 = T2 / (r**(gamma - 1))

        # 2) If we know T4,P4,P1 and 4->1 is isochoric => P1/P4 = T1/T4 => T1 = T4 * (P1/P4)
        elif (unknown(T1) and 
            known(T4) and
            known(P4) and
            known(P1)):
            T1 = T4 * (P1 / P4)

        # 3) If we know internal energy u1 => invert via table or ideal‐gas formula
        elif (unknown(T1) and known(u1)):
            T1 = get_apdx_7('u', u1, 'T')

        # 4) If we know enthalpy h1 => invert via table or formula
        elif (unknown(T1) and known(h1)):
            T1 = get_apdx_7('h', h1, 'T')
        
        if cold_air_standard:
            # 5) If we know Qc, cv, T4 => Qc = cv * (T4 - T1) => T1 = T4 - Qc / cv
            if known(vars['Qc']) and known(vars['cv']) and known(T4):
                T1 = T4 - (vars['Qc'] / vars['cv'])
            
            # 6) If we know Wi, cv, T2 => Wi = cv * (T2 - T1) => T1 = T2 - Wi / cv
            elif known(vars['Wi']) and known(vars['cv']) and known(T2):
                T1 = T2 - (vars['Wi'] / vars['cv'])

        #  5) If we know s1, P1 => T1 from s,P
        # if (unknown(T1) and known(s1) and known(P1)):
        #     T_test = T_from_sP(s1, P1)
        #     if known(T_test):
        #         T1 = T_test
    
    # ----------------------------------------------------
    # C) P1 RELATIONS
    # ----------------------------------------------------
    if unknown(P1):
        # 1) From isentropic 1->2 => P2 = P1 * r^gamma => P1 = P2 / r^gamma
        if known(P2) and known(r) and known(gamma):
            P1 = P2 / (r**gamma)

        # 2) If T1, v1, R known => P1 = R*T1 / v1
        elif (unknown(P1) and 
            known(T1) and 
            known(v1) and 
            known(Rg)):
            P1 = (Rg * T1) / v1

        # 3) If T4,P4,T1 known and 4->1 is isochoric => P1 / T1 = P4 / T4 => P1 = (T1/T4)*P4
        elif (unknown(P1) and 
            known(T1) and
            known(T4) and
            known(P4)):
            P1 = (T1 / T4)*P4

        # 4) Possibly from s1, T1 => P_from_sT(...) if you have a table approach
        #    or from h1, v1, etc. 
        vars['1']['P'] = P1

    # ----------------------------------------------------
    # E) h1 RELATIONS
    # ----------------------------------------------------
    if unknown(h1):
        # 1) If T1 is known => h1 = h(T1)
        if known(T1):
            h1 = get_apdx_7('T', T1, 'h')
        vars['1']['h'] = h1

    # ----------------------------------------------------
    # F) u1 RELATIONS
    # ----------------------------------------------------
    if unknown(u1):
        # 1) If T1 is known => u1 = u(T1)
        if known(T1):
            u1 = get_apdx_7('T', T1, 'u')
        vars['1']['u'] = u1
    
    return vars

def step_2(vars, cold_air_standard=False):
    """
    Calculates all the variables at state 2 that are possible to get from 
    the currently known variables in a Otto cycle.
    
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
    list: Updated list of dictionaries with calculated variables for state 2.
    """

    # Unpack cycle-level variables
    r     = vars['r']     # compression ratio = v1 / v2
    Rg    = vars['R']     # specific gas constant, kJ/(kg·K)
    gamma = vars['gamma']

    # Unpack state 1 (start of compression)
    T1 = vars['1']['T']
    P1 = vars['1']['P']
    v1 = vars['1']['v']
    s1 = vars['1']['s']
    h1 = vars['1']['h']
    u1 = vars['1']['u']

    # Unpack state 2 (end of compression)
    T2 = vars['2']['T']
    P2 = vars['2']['P']
    v2 = vars['2']['v']
    s2 = vars['2']['s']
    h2 = vars['2']['h']
    u2 = vars['2']['u']

    # Unpack state 3 (end of heat addition)
    T3 = vars['3']['T']
    P3 = vars['3']['P']
    v3 = vars['3']['v']
    s3 = vars['3']['s']
    h3 = vars['3']['h']
    u3 = vars['3']['u']

    # ----------------------------------------------------
    # A) v2 RELATIONS
    # ----------------------------------------------------
    if unknown(v2):
        # From isentropic 1->2 => v2 = v1 / r
        if known(v1) and known(r):
            v2 = v1 / r
        
        # From constant-volume 2->3 => v2 = v3
        elif known(v3):
            v2 = v3

        # If T2, P2, R known => v2 = (R*T2)/P2 (ideal-gas assumption)
        elif (known(T2) and 
            known(P2) and 
            known(Rg)):
            v2 = (Rg * T2) / P2

        vars['2']['v'] = v2

    # ----------------------------------------------------
    # B) T2 RELATIONS
    # ----------------------------------------------------
    if unknown(T2):
        # 1) Isentropic 1->2 => T2 = T1 * (r)^(gamma - 1)
        if known(T1) and known(r) and known(gamma):
            T2 = T1 * (r ** (gamma - 1))

        # 2) If we know T3, P2, P3 and 2->3 is constant v => for an ideal gas, T3/T2 = P3/P2
        # => T2 = T3 * P2 / P3
        elif known(T3) and known(P2) and known(P3):
            T2 = T3 * (P2 / P3)

        # 3) If we know internal energy u2 => invert via table
        elif known(u2):
            T2 = get_apdx_7('u', u2, 'T')

        # 4) If we know enthalpy h2 => invert via table
        elif known(h2):
            T2 = get_apdx_7('h', h2, 'T')
        
        if cold_air_standard:
            # 5) If we know Qh, cv, T3 => Qh = cv * (T3 - T2) => T2 = T3 - Qh / cv
            if known(vars['Qh']) and known(vars['cv']) and known(T3):
                T2 = T3 - (vars['Qh'] / vars['cv'])
            
            # 6) If we know Wi, cv, T1 => Wi = cv * (T2 - T1) => T2 = Wi / cv + T1
            elif known(vars['Wi']) and known(vars['cv']) and known(T1):
                T2 = (vars['Wi'] / vars['cv']) + T1

        vars['2']['T'] = T2

    # ----------------------------------------------------
    # C) P2 RELATIONS
    # ----------------------------------------------------
    if unknown(P2):
        # 1) Isentropic 1->2 => P2 = P1 * (r^gamma)
        if known(P1) and known(r) and known(gamma):
            P2 = P1 * (r ** gamma)

        # 2) If we know T2, T3, P3, and 2->3 is constant v => for an ideal gas, T3/T2 = P3/P2
        # => P2 = P3 * (T2 / T3)
        elif known(T2) and known(T3) and known(P3):
            P2 = P3 * (T2 / T3)

        # 3) If T2, v2, R known => P2 = (R*T2)/v2
        elif (known(T2) and
            known(v2) and
            known(Rg)):
            P2 = (Rg * T2) / v2

        vars['2']['P'] = P2

    # ----------------------------------------------------
    # D) s2 RELATIONS
    # ----------------------------------------------------
    if unknown(s2):
        # 1) Isentropic 1->2 => s2 = s1
        if known(s1):
            s2 = s1
        """
        # 2) If T2,P2 known => s2 = get_apdx_7(...) or from s(T2,P2)
        elif (unknown(s2) and 
            known(T2) and 
            known(P2)):
            # Example using APDX 7 with T for interpolation:
            # Then if you have P2 as well, you might do a more advanced approach. 
            # For demonstration, we'll just get s from T2 alone:
            s2 = get_apdx_7('T', T2, 's0')  
            # In many ideal-gas tables, 's0' is measured from a reference pressure, 
            # so you might need s2 = s0(T2) + R ln(Pref/P2). 
            # Adjust as needed for your setup.
        """
        vars['2']['s'] = s2

    # ----------------------------------------------------
    # E) h2 RELATIONS
    # ----------------------------------------------------
    if unknown(h2):
        # 1) If T2 known => h2 = get_apdx_7('T', T2, 'h')
        if known(T2):
            h2 = get_apdx_7('T', T2, 'h')

        vars['2']['h'] = h2

    # ----------------------------------------------------
    # F) u2 RELATIONS
    # ----------------------------------------------------
    if unknown(u2):
        # 1) If T2 known => u2 = get_apdx_7('T', T2, 'u')
        if known(T2):
            u2 = get_apdx_7('T', T2, 'u')

        vars['2']['u'] = u2

    return vars

def step_3(vars, cold_air_standard=False):
    """
    Calculates all the variables at state 3 that are possible to get
    from the currently known variables (states 2 and 4) in a Otto cycle.
    
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
      list: updated list of dictionaries with the newly computed state‐3 variables.
    """

    # Unpack cycle-level data
    Rg    = vars['R']     # gas constant (kJ/(kg·K))
    gamma = vars['gamma']

    # State 2
    T2 = vars['2']['T']
    P2 = vars['2']['P']
    v2 = vars['2']['v']
    s2 = vars['2']['s']
    h2 = vars['2']['h']
    u2 = vars['2']['u']

    # State 3
    T3 = vars['3']['T']
    P3 = vars['3']['P']
    v3 = vars['3']['v']
    s3 = vars['3']['s']
    h3 = vars['3']['h']
    u3 = vars['3']['u']

    # State 4
    T4 = vars['4']['T']
    P4 = vars['4']['P']
    v4 = vars['4']['v']
    s4 = vars['4']['s']
    h4 = vars['4']['h']
    u4 = vars['4']['u']

    # ----------------------------------------------------
    # A) v3 RELATIONS
    # ----------------------------------------------------
    if unknown(v3):
        # 1) If 2->3 is constant volume => v3 = v2
        if known(v2):
            v3 = v2

        # 2) If T3, P3, and R are known (ideal gas) => v3 = (R * T3) / P3
        elif (unknown(v3) and 
            known(T3) and
            known(P3) and
            known(Rg)):
            v3 = (Rg * T3) / P3

        # 3) If we know T4,P4, and the isentropic expansion 3->4:
        #    P4 = P3*(v3/v4)^(-gamma) or T4 = T3*(v3/v4)^(1-gamma).
        #    Usually you solve for T4 from T3, but you could invert for v3
        #    if T3 is known. That might be more complicated. 
        #    We won't do the full inversion here unless needed.
        vars['3']['v'] = v3

    # ----------------------------------------------------
    # B) T3 RELATIONS
    # ----------------------------------------------------
    if unknown(T3):        
        # 1) If we know T2, P2, P3 and 2->3 is constant v => for an ideal gas, T3/T2 = P3/P2
        # => T3 = T2 * (P3 / P2)
        if known(T2) and known(P2) and known(P3):
            T3 = T2 * (P3 / P2)

        # 2) If we know T4 and the isentropic relation 3->4 => 
        #    T4 = T3 * (v3/v4)^(1-gamma) => T3 = T4*(v4/v3)^(gamma-1)
        elif (unknown(T3) and 
            known(T4) and
            known(v4) and
            known(v3) and
            known(gamma)):
            T3 = T4 * ((v4 / v3) ** (gamma - 1))

        # 3) If we have u3 => invert via table
        elif (unknown(T3) and known(u3)):
            T3 = get_apdx_7('u', u3, 'T')

        # 4) If we have h3 => invert via table
        elif (unknown(T3) and known(h3)):
            T3 = get_apdx_7('h', h3, 'T')
        
        if cold_air_standard:
            # 5) If we know Qh, cv, T2 => Qh = cv * (T3 - T2) => T3 = Qh / cv + T2
            if known(vars['Qh']) and known(vars['cv']) and known(T2):
                T3 = (vars['Qh'] / vars['cv']) + T2
            
            # 6) If we know Wo, cv, T4 => Wo = cv * (T3 - T4) => T3 = Wo / cv + T4
            elif known(vars['Wo']) and known(vars['cv']) and known(T4):
                T3 = (vars['Wo'] / vars['cv']) + T4

        vars['3']['T'] = T3

    # ----------------------------------------------------
    # C) P3 RELATIONS
    # ----------------------------------------------------
    if unknown(P3):
        # 1) If we know T2, T3, P2, and 2->3 is constant v => for an ideal gas, T3/T2 = P3/P2
        # => P3 = P2 * (T3 / T2)
        if known(T2) and known(T3) and known(P3):
            P3 = P2 * (T3 / T2)

        # 2) If 3->4 is isentropic => P4 = P3*(v3/v4)^gamma => 
        #    => P3 = P4 / (v3/v4)^gamma = P4*(v4/v3)^gamma
        elif (unknown(P3) and 
            known(P4) and
            known(v4) and
            known(v3) and
            known(gamma)):
            P3 = P4 * ((v4 / v3) ** gamma)

        # 3) If T3, v3, and R known => P3 = R*T3 / v3
        elif (unknown(P3) and
            known(T3) and
            known(v3) and
            known(Rg)):
            P3 = (Rg * T3) / v3

        vars['3']['P'] = P3

    # ----------------------------------------------------
    # D) s3 RELATIONS
    # ----------------------------------------------------
    if unknown(s3):
        # 1) If 3->4 is isentropic => s3 = s4
        if known(s4):
            s3 = s4
        """
        # 2) If T3,P3 known => s3 = s(T3,P3) from table or formula
        elif (unknown(s3) and
            known(T3) and
            known(P3)):
            # Example usage with APDX 7. Many tables store s0 for an arbitrary reference pressure:
            # s3 = s0(T3) + R ln(P_ref / P3) or similar. 
            # For illustration, we'll just do:
            s3 = get_apdx_7('T', T3, 's0')  
            # (You might adjust for actual P3 if your table requires it.)
        """
        vars['3']['s'] = s3
        
    # ----------------------------------------------------
    # E) h3 RELATIONS
    # ----------------------------------------------------
    if unknown(h3):
        # 1) If T3 is known => h3 = get_apdx_7('T', T3, 'h')
        if known(T3):
            h3 = get_apdx_7('T', T3, 'h')

        vars['3']['h'] = h3

    # ----------------------------------------------------
    # F) u3 RELATIONS
    # ----------------------------------------------------
    if unknown(u3):
        # 1) If T3 is known => u3 = get_apdx_7('T', T3, 'u')
        if known(T3):
            u3 = get_apdx_7('T', T3, 'u')

        vars['3']['u'] = u3

    return vars

def step_4(vars, cold_air_standard=False):
    """
    Calculates all the variables at state 4 that are possible to get
    from the currently known variables (states 3 and 1) in a Otto cycle.
   
    Parameters:
    vars (list): List of dictionaries containing the variables for states 0..4.

    Returns:
      list: updated vars with newly computed state-4 variables.
    """

    # Unpack cycle-level data
    Rg    = vars['R']        # kJ/(kg·K)
    gamma = vars['gamma']

    # State 1 (after 4, in the cycle loop)
    T1 = vars['1']['T']
    P1 = vars['1']['P']
    v1 = vars['1']['v']
    # s1, h1, u1 if needed

    # State 3 (before 4 in the cycle)
    T3 = vars['3']['T']
    P3 = vars['3']['P']
    v3 = vars['3']['v']
    s3 = vars['3']['s']
    h3 = vars['3']['h']
    u3 = vars['3']['u']

    # State 4
    T4 = vars['4']['T']
    P4 = vars['4']['P']
    v4 = vars['4']['v']
    s4 = vars['4']['s']
    h4 = vars['4']['h']
    u4 = vars['4']['u']

    # ----------------------------------------------------
    # A) v4 RELATIONS
    # ----------------------------------------------------
    if unknown(v4):
        # 1) If process 4->1 is constant volume => v4 = v1
        if known(v1):
            v4 = v1

        # 2) If T4, P4, and R known => v4 = (R * T4) / P4 (ideal gas)
        if (unknown(v4) and
            known(T4) and
            known(P4) and
            known(Rg)):
            v4 = (Rg * T4) / P4

        # 3) If 3->4 is isentropic => P4 = P3*(v3/v4)^gamma  or T4= T3*(v3/v4)^(gamma-1)
        #    you could solve for v4 if T4 or P4 is known, but that's a bit more involved. 
        #    Example (only if T3,T4 are known):
        elif (unknown(v4) and 
            known(T3) and
            known(T4) and
            known(v3) and
            known(gamma)):
            # T4 = T3*(v3/v4)^(gamma-1) => v4 = v3 * (T3 / T4)^(1/(gamma-1))
            ratio_exp = 1.0/(gamma - 1.0)
            v4 = v3 * ((T3 / T4) ** ratio_exp)

        vars['4']['v'] = v4

    # ----------------------------------------------------
    # B) T4 RELATIONS
    # ----------------------------------------------------
    if unknown(T4):
        # 1) Isentropic from 3->4 => T4 = T3*(v3/v4)^(gamma - 1)
        if (known(T3) and 
            known(v3) and
            known(v4) and
            known(gamma)):
            T4 = T3 * ((v3 / v4) ** (gamma - 1))

        # 2) If 4->1 is isochoric => (P4/T4) = (P1/T1) => T4 = (P4/P1)*T1
        elif (unknown(T4) and
            known(P4) and
            known(P1) and
            known(T1)):
            T4 = (P4 / P1) * T1

        # 3) If we have u4 => T4 = invert via table, get_apdx_7('u', u4, 'T')
        elif (unknown(T4) and known(u4)):
            T4 = get_apdx_7('u', u4, 'T')

        # 4) If we have h4 => T4 = invert via table, get_apdx_7('h', h4, 'T')
        elif (unknown(T4) and known(h4)):
            T4 = get_apdx_7('h', h4, 'T')
        
        if cold_air_standard:
            # 5) If we know Qc, cv, T1 => Qc = cv * (T4 - T1) => T4 = Qc / cv + T1
            if known(vars['Qc']) and known(vars['cv']) and known(T1):
                T4 = (vars['Qc'] / vars['cv']) + T1
            
            # 6) If we know Wo, cv, T3 => Wo = cv * (T3 - T4) => T4 = T3 - Wo / cv
            elif known(vars['Wo']) and known(vars['cv']) and known(T3):
                T4 = T3 - (vars['Wo'] / vars['cv'])

        vars['4']['T'] = T4

    # ----------------------------------------------------
    # C) P4 RELATIONS
    # ----------------------------------------------------
    if unknown(P4):
        # 1) Isentropic 3->4 => P4 = P3 * (v3 / v4)^gamma
        if (known(P3) and 
            known(v3) and
            known(v4) and
            known(gamma)):
            P4 = P3 * ((v3 / v4) ** gamma)

        # 2) If T4, v4, R known => P4 = (R * T4) / v4
        elif (unknown(P4) and
            known(T4) and
            known(v4) and
            known(Rg)):
            P4 = (Rg * T4) / v4

        # 3) If 4->1 is isochoric => P4 / T4 = P1 / T1 => P4 = (T4 / T1)* P1
        elif (unknown(P4) and
            known(T4) and
            known(T1) and
            known(P1)):
            P4 = (T4 / T1)* P1

        vars['4']['P'] = P4

    # ----------------------------------------------------
    # D) s4 RELATIONS
    # ----------------------------------------------------
    if unknown(s4):
        # 1) Isentropic 3->4 => s4 = s3
        if known(s3):
            s4 = s3
        """
        # 2) If T4,P4 known => s4 = s(T4,P4) from table or formula
        elif (unknown(s4) and
            known(T4) and
            known(P4)):
            # For an ideal-gas table, we might do s0(T4) + R ln(Pref/P4), etc.
            # For demonstration, we just call get_apdx_7('T', T4, 's0'):
            s4 = get_apdx_7('T', T4, 's0')
        """
        vars['4']['s'] = s4

    # ----------------------------------------------------
    # E) h4 RELATIONS
    # ----------------------------------------------------
    if unknown(h4):
        # 1) If T4 known => h4 = h(T4)
        if known(T4):
            h4 = get_apdx_7('T', T4, 'h')

        vars['4']['h'] = h4

    # ----------------------------------------------------
    # F) u4 RELATIONS
    # ----------------------------------------------------
    if unknown(u4):
        # 1) If T4 known => u4 = u(T4)
        if known(T4):
            u4 = get_apdx_7('T', T4, 'u')

        vars['4']['u'] = u4

    return vars

def solve_otto_cycle(variables, cold_air_standard=False, verbose=False):
    """
    Solves the Otto cycle using the known variables and calculates the unknown variables.
    
    Parameters:
    variables (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    if unknown(variables['gamma']) and known(variables['1']['T']):
        variables['gamma'] = get_apdx_4('Air', 'T', variables['1']['T'], 'gamma')

    elif unknown(variables['gamma']):
        variables['gamma'] = get_apdx_1('Air', 'gamma')

    if unknown(variables['R']):
        variables['R'] = get_apdx_1('Air', 'R')

    if unknown(variables['cv']) and known(variables['1']['T']) and cold_air_standard:
        variables['cv'] = get_apdx_4('Air', 'T', variables['1']['T'], 'cv')
    
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
        variables = step_1(variables, cold_air_standard=cold_air_standard)
        
        # Step 2:
        variables = step_2(variables, cold_air_standard=cold_air_standard)

        # Step 3:
        variables = step_3(variables, cold_air_standard=cold_air_standard)

        # Step 4:
        variables = step_4(variables, cold_air_standard=cold_air_standard)
        
        current_nan_count = count_nans(variables)

        if verbose:
            print(f"Iteration {counter}: {current_nan_count} unknowns")

        if current_nan_count == 0 or \
        current_nan_count >= previous_nan_count:
            break

        previous_nan_count = current_nan_count
    
    return variables

def otto_display_tables(variables, sig_figs=4):
    display_tables(variables, 'Otto', sig_figs=sig_figs)