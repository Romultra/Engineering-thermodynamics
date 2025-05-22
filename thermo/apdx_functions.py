import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.interpolate import griddata
from importlib.resources import files

def get_apdx_1(gas, request, use_chem_formula=False):
    """
    Retrieves the APDX 1 data for a given gas and request type.
    Appendix 1 contains these properties for various gases, including air:
    - M: Molar mass (kg/kmol)
    - R: Specific gas constant (kJ/(kg*K))
    - cp: Specific heat at constant pressure (kJ/(kg*K))
    - cv: Specific heat at constant volume (kJ/(kg*K))
    - gamma: Specific heat ratio (dimensionless)

    The specific heats are evaluated at 25°C and 100kPa.
    
    Parameters:
    gas (str): The type of gas.
    request (str): The type of request ('R' or 'cp').
    use_chem_formula (bool): Whether to use the chemical formula or the gas name. (Air doesn't have a chemical formula).
    
    Returns:
    float: The requested property of the gas.
    """
    data_path = files("thermo.Appendix-data").joinpath("1-Properties-of-gases.csv")
    data = pd.read_csv(data_path, header=1)
    if use_chem_formula:
        return data[request][data['Chemical formula'] == gas].values[0]
    else:
        return data[request][data['Gas'] == gas].values[0]
    
def get_apdx_4(gas, relative_to, input, request):
    """
    Retrieves the APDX 4 data for a given gas and request type.
    Appendix 4 contains the specific heat data for various gases:
    - T: Temperature (K)
    - cp: Specific heat at constant pressure (kJ/(kg*K))
    - cv: Specific heat at constant volume (kJ/(kg*K))
    - gamma: Specific heat ratio (dimensionless)

    the available gases are:
    Air, CO2, CO, H2, N2, O2

    Parameters:
    gas (str): The type of gas from the list above.
    relative_to (str): The variable to interpolate against ('T', 'cp').
    input (array-like or number): The input values for interpolation.
    request (str): The variable to retrieve ('cp', 'cv', or 'gamma').

    Returns:
    array: The interpolated output values.
    """
    data_path = files("thermo.Appendix-data").joinpath("4-Specific-Heats-of-Gases.csv")
    data = pd.read_csv(data_path, header=1)
    
    data = data[data['Gas'] == gas]  # Filter for the selected gas

    # Convert columns to numeric
    data[relative_to] = pd.to_numeric(data[relative_to])
    data[request] = pd.to_numeric(data[request])

    if type(input) != list or type(input) != np.ndarray:
        input = np.array([input])
    
    output_value = np.interp(input, data[relative_to].to_numpy(), data[request].to_numpy())
    return output_value[0] if np.ndim(output_value)==1 else output_value

def get_apdx_7(relative_to, input, request):
    """
    Retrieves the APDX 7 data for a given relative_to, input, and request.
    This appendix contains the Ideal Gas Tables for Air, including:
    - T: Temperature (K)
    - h: Enthalpy (kJ/kg)
    - Pr: Relative Pressure (Dimensionless)
    - u: Internal Energy (kJ/kg)
    - vr: Relative Specific Volume (Dimensionless)
    - s0: Specific Entropy (kJ/kg·K)

    Parameters:
    relative_to (str): The variable to interpolate against ('T' or 'Pr').
    input (array-like or number): The input values for interpolation.
    request (str): The variable to retrieve ('h', 'u', 'vr', or 's0').

    Returns:
    array: The interpolated output values.
    """
    data_path = files("thermo.Appendix-data").joinpath("7-Ideal-Gas-Tables-for-Air.csv")
    data = pd.read_csv(data_path, header=1)
    
    # Convert columns to numeric
    data[relative_to] = pd.to_numeric(data[relative_to])
    data[request] = pd.to_numeric(data[request])

    if type(input) != list or type(input) != np.ndarray:
        input = np.array([input])
    
    output_value = np.interp(input, data[relative_to].to_numpy(), data[request].to_numpy())
    return output_value[0] if np.ndim(output_value)==1 else output_value

# Useful lamda macros
def h_air(T):
    return get_apdx_7('T', T, 'h')
def u_air(T):
    return get_apdx_7('T', T, 'u')

def get_apdx_8ab(table_base, relative_to, input, request):
    """
    Appendix 8: Properties of saturated Water.

    Retrieves and interpolates the APDX data for a given table base, relative to, input, and request.
    This appendix is either based on pressure or temperature.
    Appendix 8 includes these properties for saturated Water:
    - Tsat: Saturation temperature (°C)
    - P: Pressure (MPa)
    - vf: Specific volume (liquid) (m³/kg)
    - vg: Specific volume (vapor) (m³/kg)
    - uf: Internal energy (liquid) (kJ/kg)
    - ug: Internal energy (vapor) (kJ/kg)
    - hf: Enthalpy (liquid) (kJ/kg)
    - hg: Enthalpy (vapor) (kJ/kg)
    - sf: Entropy (liquid) (kJ/(kg*K))
    - sg: Entropy (vapor) (kJ/(kg*K))  

    Parameters:
    table_base (str): The base of the table ('Pressure' or 'Temperature').
    relative_to (str): The variable to interpolate against ('T' or 'P').
    input (array-like or number) : The input values for interpolation.
    request (str): The variable to retrieve ('sf' or 'sg').

    Returns:
    array: The interpolated output values.
    """
    if table_base == 'Pressure':
        data_path = files("thermo.Appendix-data").joinpath("8b-Saturated-Water-Pressure.csv")
        data = pd.read_csv(data_path, header=1)
    elif table_base == 'Temperature':
        data_path = files("thermo.Appendix-data").joinpath("8a-Saturated-Water-Temperature.csv")
        data = pd.read_csv(data_path, header=1)
    else:
        raise ValueError("Invalid table_base. Choose 'Pressure' or 'Temperature'.")
    
    if relative_to == 'T':
        relative_to = 'Tsat'
    if request == 'T':
        request = 'Tsat'

    data[relative_to] = pd.to_numeric(data[relative_to])  # Convert to numeric
    data[request] = pd.to_numeric(data[request])  # Convert to numeric

    if type(input) == float or type(input) == int:
        input = np.array([input])

    output_value = np.interp(input, data[relative_to].to_numpy(), data[request].to_numpy())
    return output_value[0] if np.ndim(output_value)==1 else output_value

def get_apdx_8c(relative_to: tuple, input: tuple, request: str):
    """
    Appendix 8c: Properties of superheated Water.

    Retrieves and interpolates the APDX data for a given relative_to, input, and request.
    This appendix is based on pressure and temperature.
    Appendix 8c includes these properties for superheated Water:
    - T: Saturation temperature (°C)
    - P: Pressure (MPa)
    - v: Specific volume (m³/kg)
    - u: Internal energy (kJ/kg)
    - h: Enthalpy (kJ/kg)
    - s: Entropy (kJ/(kg*K))

    Parameters:
    relative_to (tuple): The variables to interpolate against ('P', 'T').
    input (tuple): The input values for interpolation.
    request (str): The variable to retrieve ('s').

    Returns:
    array: The interpolated output values.
    """
    # Load the new uploaded CSV
    data_path = files("thermo.Appendix-data").joinpath("8c-Superheated-Water.csv")
    data = pd.read_csv(data_path, header=1)

    # Convert Pressure and Temperature columns to numeric
    data[relative_to[0]] = pd.to_numeric(data[relative_to[0]], errors='coerce')
    data[relative_to[1]] = pd.to_numeric(data[relative_to[1]], errors='coerce')
    data[request] = pd.to_numeric(data[request], errors='coerce')

    # Remove rows where any of the relevant columns contain NaN
    clean_data = data.dropna(subset=[relative_to[0], relative_to[1], request])
    
    # Check if input pressure or temperature matches exactly with the data
    # If so, use 1D interpolation for better accuracy
    unique_values_0 = clean_data[relative_to[0]].unique()
    unique_values_1 = clean_data[relative_to[1]].unique()
    
    # Check if either coordinate matches exactly with available data
    if input[0] in unique_values_0:
        # Filter data for the exact first coordinate and do 1D interpolation on second coordinate
        filtered_data = clean_data[clean_data[relative_to[0]] == input[0]]
        return np.interp(input[1], filtered_data[relative_to[1]], filtered_data[request])
    
    # Check if second coordinate matches exactly
    elif input[1] in unique_values_1:
        # Filter data for the exact second coordinate and do 1D interpolation on first coordinate
        filtered_data = clean_data[clean_data[relative_to[1]] == input[1]]
        return np.interp(input[0], filtered_data[relative_to[0]], filtered_data[request])
    
    # If no exact match, use 2D interpolation
    else:
        # Prepare points and values from cleaned data
        points = np.column_stack((clean_data[relative_to[0]], clean_data[relative_to[1]]))
        output_value = griddata(points, clean_data[request], (input[0], input[1]), method='linear')
        return np.float64(output_value) if np.ndim(output_value)==0 else output_value[0] if np.ndim(output_value)==1 else output_value

def get_apdx_9ab(table_base, relative_to, input, request):
    """
    Appendix 9: Properties of saturated R134a.

    Retrieves and interpolates the APDX data for a given table base, relative to, input, and request.
    This appendix is either based on pressure or temperature.
    Appendix 9 includes these properties for saturated R134a:
    - Tsat: Saturation temperature (°C)
    - P: Pressure (MPa)
    - vf: Specific volume (liquid) (m³/kg)
    - vg: Specific volume (vapor) (m³/kg)
    - uf: Internal energy (liquid) (kJ/kg)
    - ug: Internal energy (vapor) (kJ/kg)
    - hf: Enthalpy (liquid) (kJ/kg)
    - hg: Enthalpy (vapor) (kJ/kg)
    - sf: Entropy (liquid) (kJ/(kg*K))
    - sg: Entropy (vapor) (kJ/(kg*K))  

    Parameters:
    table_base (str): The base of the table ('Pressure' or 'Temperature').
    relative_to (str): The variable to interpolate against ('T' or 'P').
    input (array-like or number): The input values for interpolation.
    request (str): The variable to retrieve ('sf' or 'sg').

    Returns:
    array: The interpolated output values.
    """
    if table_base == 'Pressure':
        data_path = files("thermo.Appendix-data").joinpath("9b-Saturated-R134a-Pressure.csv")
        data = pd.read_csv(data_path, header=1)
    elif table_base == 'Temperature':
        data_path = files("thermo.Appendix-data").joinpath("9a-Saturated-R134a-Temperature.csv")
        data = pd.read_csv(data_path, header=1)
    else:
        raise ValueError("Invalid table_base. Choose 'Pressure' or 'Temperature'.")
    
    if relative_to == 'T':
        relative_to = 'Tsat'
    if request == 'T':
        request = 'Tsat'

    data[relative_to] = pd.to_numeric(data[relative_to])  # Convert to numeric
    data[request] = pd.to_numeric(data[request])  # Convert to numeric
    if type(input) == float or type(input) == int:
        input = np.array([input])
    
    output_value = np.interp(input, data[relative_to].to_numpy(), data[request].to_numpy())
    return output_value[0] if np.ndim(output_value)==1 else output_value

def get_apdx_9c(relative_to: tuple, input: tuple, request: str):
    """
    Appendix 9c: Properties of superheated R134a.

    Retrieves and interpolates the APDX data for a given relative_to, input, and request.
    This appendix is based on pressure and temperature.
    Appendix 9c includes these properties for superheated R134a:
    - T: Saturation temperature (°C)
    - P: Pressure (MPa)
    - v: Specific volume (m³/kg)
    - u: Internal energy (kJ/kg)
    - h: Enthalpy (kJ/kg)
    - s: Entropy (kJ/(kg*K))

    Parameters:
    relative_to (tuple): The variables to interpolate against ('P', 'T').
    input (tuple): The input values for interpolation.
    request (str): The variable to retrieve ('s').

    Returns:
    array: The interpolated output values.
    """
    # Load the new uploaded CSV
    data_path = files("thermo.Appendix-data").joinpath("9c-Superheated-R134a.csv")
    data = pd.read_csv(data_path, header=1)

    # Convert Pressure and Temperature columns to numeric
    data[relative_to[0]] = pd.to_numeric(data[relative_to[0]], errors='coerce')
    data[relative_to[1]] = pd.to_numeric(data[relative_to[1]], errors='coerce')
    data[request] = pd.to_numeric(data[request], errors='coerce')

    # Remove rows where any of the relevant columns contain NaN
    clean_data = data.dropna(subset=[relative_to[0], relative_to[1], request])
    
    # Check if input pressure or temperature matches exactly with the data
    # If so, use 1D interpolation for better accuracy
    unique_values_0 = clean_data[relative_to[0]].unique()
    unique_values_1 = clean_data[relative_to[1]].unique()
    
    # Check if either coordinate matches exactly with available data
    if input[0] in unique_values_0:
        # Filter data for the exact first coordinate and do 1D interpolation on second coordinate
        filtered_data = clean_data[clean_data[relative_to[0]] == input[0]]
        return np.interp(input[1], filtered_data[relative_to[1]], filtered_data[request])
    
    # Check if second coordinate matches exactly
    elif input[1] in unique_values_1:
        # Filter data for the exact second coordinate and do 1D interpolation on first coordinate
        filtered_data = clean_data[clean_data[relative_to[1]] == input[1]]
        return np.interp(input[0], filtered_data[relative_to[0]], filtered_data[request])
    
    # If no exact match, use 2D interpolation
    else:
        # Prepare points and values from cleaned data
        points = np.column_stack((clean_data[relative_to[0]], clean_data[relative_to[1]]))
        output_value = griddata(points, clean_data[request], (input[0], input[1]), method='linear')
        return np.float64(output_value) if np.ndim(output_value)==0 else output_value[0] if np.ndim(output_value)==1 else output_value


def format_value(val, sig_figs=4):
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

def x_from_PT_and_var(apdx, pt_type, pt_value, known_var, known_var_value):
    """
    Compute the quality (x) based on one quality-dependent variable (h, s, u, or v)
    and either pressure (P) or temperature (T).
    
    Parameters:
    apdx (int): The appendix number to use (8 for water, 9 for R134a)
    pt_type (str): The type of state variable provided ('P' for pressure or 'T' for temperature)
    pt_value (float): The value of the pressure or temperature
    known_var (str): The quality-dependent property that is known ('v', 'h', 's', or 'u')
    known_var_value (float): The value of the known quality-dependent property
    
    Returns:
    float: The quality (x) of the mixture
    """
    # Determine the table base based on the provided state variable
    if pt_type == 'P':
        table_base = 'Pressure'
    elif pt_type == 'T':
        table_base = 'Temperature'
    else:
        raise ValueError("pt_type must be either 'P' or 'T'")
    
    # Get the saturated liquid and vapor properties
    if apdx == 8:
        var_f = get_apdx_8ab(table_base, pt_type, pt_value, known_var + 'f')
        var_g = get_apdx_8ab(table_base, pt_type, pt_value, known_var + 'g')
    elif apdx == 9:
        var_f = get_apdx_9ab(table_base, pt_type, pt_value, known_var + 'f')
        var_g = get_apdx_9ab(table_base, pt_type, pt_value, known_var + 'g')
    else:
        raise ValueError("apdx must be either 8 (water) or 9 (R134a)")

    # Calculate quality using the lever rule
    x = (known_var_value - var_f) / (var_g - var_f)
    
    return x

def vars_from_x_and_PT(apdx, pt_type, pt_value, quality, print_vars=True):
    """
    Calculate all thermodynamic state variables for a two-phase mixture based on 
    the quality (x) and either pressure (P) or temperature (T).
    
    Parameters:
    apdx (int): The appendix number to use (8 for water, 9 for R134a)
    pt_type (str): The type of state variable provided ('P' for pressure or 'T' for temperature)
    pt_value (float): The value of the pressure or temperature
    quality (float): The quality (x) of the mixture (between 0 and 1)
    print_vars (bool): If True, prints all calculated variables with formatted values
    
    Returns:
    dict: Dictionary containing all state variables (P, T, v, u, h, s)
    """
    vars = {}
    apdx_func = get_apdx_8ab if apdx == 8 else get_apdx_9ab

    x = quality
    if pt_type == 'P':
        table_base = 'Pressure'
        vars['T'] = apdx_func(table_base, pt_type, pt_value, 'T')
    
    elif pt_type == 'T':
        table_base = 'Temperature'
        vars['P'] = apdx_func(table_base, pt_type, pt_value, 'P')
    
    for var in ['v', 'h', 's', 'u']:
        var_f = apdx_func(table_base, pt_type, pt_value, var + 'f')
        var_g = apdx_func(table_base, pt_type, pt_value, var + 'g')
        vars[var] = (1 - x) * var_f + x * var_g
    
    if print_vars:
        for var, value in vars.items():
            print(f"{var}: {format_value(value)}")
        return vars

    return vars

def vars_from_x_and_quality_var(apdx, quality, known_var, known_var_value, print_vars=True, verbose=False):
    """
    Fill in all state variables based on known quality (x) and one quality-dependent variable (h, s, or u).

    Parameters:
    vars (dict): Dictionary containing state variables, including quality 'x'
    known_var (str): The quality-dependent property that is known ('v', 'h', 's', or 'u').
    verbose (bool): If True, prints debugging information.
    """
    assert known_var in ['v', 'h', 's', 'u'], "Known variable must be a quality-dependent property."

    vars = {}
    apdx_func = get_apdx_8ab if apdx == 8 else get_apdx_9ab
    x = quality
    table_base = 'Temperature'  # We'll search over T to match the known property

    if verbose:
        print(f"Starting vars_from_x_and_quality_var for '{known_var}' with target value: {known_var_value} and quality x: {x}")

    if x == 0 or x == 1:
        # If x is 0 or 1, we can directly use the saturated properties
        T_sat = apdx_func(table_base, known_var + ('f' if x == 0 else 'g'), known_var_value, 'T')
        if verbose:
            print(f"Quality is {x}, using direct lookup: T_sat = {T_sat}")
    else:
        # If x is not 0 or 1, we need to find T such that the property matches
        def objective(T):
            var_f = apdx_func(table_base, 'T', T, known_var + 'f')
            var_g = apdx_func(table_base, 'T', T, known_var + 'g')
            interpolated = (1 - x) * var_f + x * var_g
            if verbose:
                print(f"At T = {T:.3f}: var_f = {var_f}, var_g = {var_g}, interpolated = {interpolated}, objective = {interpolated - known_var_value}")
            return interpolated - known_var_value

        # Root-finding bounds for T [°C] — Range of saturation temperature for R134a in the tables
        if apdx == 8:
            T_min, T_max = 0.01, 374.14
        elif apdx == 9:
            T_min, T_max = -24, 100
        if verbose:
            print(f"Finding T_sat between {T_min} and {T_max}")

        if objective(T_min) * objective(T_max) >= 0:
            raise ValueError("f(a) and f(b) must have different signs for root_scalar to work.")

        sol = root_scalar(objective, bracket=[T_min, T_max], method='brentq')
        T_sat = sol.root
        if verbose:
            print(f"Root solver converged: T_sat = {T_sat} "
                  f"(iterations: {sol.iterations}, function calls: {sol.function_calls})")

    vars['T'] = T_sat
    
    # Now get the corresponding saturation pressure
    vars['P'] = apdx_func(table_base, 'T', T_sat, 'P')
    if verbose:
        print(f"Calculated saturation pressure: P = {vars['P']} at T_sat = {T_sat}")

    # Fill in other state variables using interpolation
    unknown_vars = ['v', 'h', 's', 'u']

    for var in unknown_vars:
        if var == known_var:
            vars[var] = known_var_value
            continue
        var_f = apdx_func(table_base, 'T', T_sat, var + 'f')
        var_g = apdx_func(table_base, 'T', T_sat, var + 'g')
        vars[var] = (1 - x) * var_f + x * var_g
        if verbose:
            print(f"Interpolated {var}: var_f = {var_f}, var_g = {var_g}, value = {vars[var]}")
    
    if print_vars:
        for var, value in vars.items():
            print(f"{var}: {format_value(value)}")

    return vars