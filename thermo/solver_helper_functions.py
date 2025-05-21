import numpy as np
from thermo.apdx_functions import *
from IPython.display import display

def known(var):
    """
    Checks if a variable is defined (not NaN).
    
    Parameters:
    var (float): The variable to check.
    
    Returns:
    bool: True if the variable is defined, False otherwise.
    """
    return not np.isnan(var)

def unknown(var):
    """
    Checks if a variable is undefined (NaN).
    
    Parameters:
    var (float): The variable to check.
    
    Returns:
    bool: True if the variable is undefined, False otherwise.
    """
    return np.isnan(var)

def equalize(var1, var2):
    """
    Sets two variables equal to each other if one is known and the other is unknown.
    
    Parameters:
    var1 (float): The first variable.
    var2 (float): The second variable.
    
    Returns:
    tuple: A tuple containing the updated variables.
    """
    if unknown(var1) and known(var2):
        var1 = var2
    elif unknown(var2) and known(var1):
        var2 = var1
    
    return var1, var2

def count_nans(variables):
    return sum(
        unknown(item) 
        for key, val in variables.items()       # top level
        for item in (val.values() if isinstance(val, dict) else [val])
    )

def get_cycle_units(cycle):
    """
    Returns the system and state variable units based on the solver type.
    
    Parameters:
    cycle (str): The type of cycle ('Rankine', 'Diesel', 'Otto').

    returns:
    tuple: (state_units, system_units) A tuple containing 
    the system and state variable dict with units inside.
    """
    if cycle == 'Rankine':
        system_units = {
            'm_dot': 'kg/s',
            'Qh_dot': 'kW',
            'qh': 'kJ/kg',
            'Qc_dot': 'kW',
            'qc': 'kJ/kg',
            'Wc_dot': 'kW',
            'wc': 'kJ/kg',
            'n': '-',
            'COP_hp': '-'
        }
        state_units = {
            'T': '°C',
            'P': 'MPa',
            'v': 'm³/kg',
            's': 'kJ/kg·K',
            'h': 'kJ/kg',
            'u': 'kJ/kg',
            'x': '-'
        }
    elif cycle == "Diesel":
        system_units = {
            # Define the system variable units for Diesel cycle here
        }
        state_units = {
            # Define the state variable units for Diesel cycle here
        }
    elif cycle == "Otto":
        system_units = {
            'r': '-',
            'R': 'kJ/(kg·K)',
            'cv': 'kJ/(kg·K)',
            'gamma': '-',
            'Qh': 'kJ/kg',
            'Qc': 'kJ/kg',
            'Wi': 'kJ/kg',
            'Wo': 'kJ/kg',
            'W': 'kJ/kg',
            'n': '-',
        }
        state_units = {
            'T': 'K',  # K
            'P': 'kPa',  # kPa
            'V': 'm³/kg',  # m³/kg
            'S': 'kJ/kg·K',  # kJ/kg·K
            'h': 'kJ/kg',  # kJ/kg
            'u': 'kJ/kg'  # kJ/kg
        }
    else:
        raise ValueError("Invalid cycle type. Choose from ['Rankine', 'Diesel', 'Otto'].")
    
    return state_units, system_units

def display_tables(variables, cycle, sig_figs=4):
    # Separate system variables and state variables
    state_vars = {}
    system_vars = {}

    state_units, system_units = get_cycle_units(cycle)
    
    # Format numbers to a given number of significant figures, keeping significant zeros
    def format_value(val, sig_figs=sig_figs):
        if isinstance(val, (int, float, np.number)):
            # Convert to string with sig_figs significant figures
            fmt_str = f"{{:.{sig_figs}g}}"
            formatted = fmt_str.format(float(val))
            # Add trailing zeros if needed to maintain sig_figs significant figures
            if '.' not in formatted and len(formatted.replace('-', '')) < sig_figs:
                # For integers, add decimal point and zeros
                formatted += '.' + '0' * (sig_figs - len(formatted.replace('-', '')))
            elif '.' in formatted:
                significant_digits = len(formatted.replace('.', '').replace('-', '').lstrip('0'))
                if significant_digits < sig_figs:
                    formatted += '0' * (sig_figs - significant_digits)
            return formatted
        return val

    # Process variables with the specified number of significant figures
    for key, value in variables.items():
        if isinstance(value, dict):
            state_vars[key] = {k: format_value(v, sig_figs) for k, v in value.items()}
        else:
            system_vars[key] = format_value(value, sig_figs)

    # Create state variables table with units
    state_df = pd.DataFrame(state_vars).T
    state_df.columns = [f"{col} ({state_units[col]})" for col in state_df.columns]
    print("\n=== State Variables ===")
    display(state_df)
    
    # Create system variables table with units
    system_df = pd.DataFrame([(f"{k} ({system_units[k]})", v) for k, v in system_vars.items()], 
                           columns=['Variable', 'Value'])
    system_df.set_index('Variable', inplace=True)
    print("\n=== System Variables ===")
    display(system_df)