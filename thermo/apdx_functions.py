import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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
    data = pd.read_csv('Appendix-data/1-Properties-of-gases.csv', header=1)
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
    data = pd.read_csv('Appendix-data/4-Specific-Heats-of-Gases.csv', header=1)
    
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
    data = pd.read_csv('Appendix-data/7-Ideal-Gas-Tables-for-Air.csv', header=1)
    
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
        data = pd.read_csv('Appendix-data/8b-Saturated-Water-Pressure.csv', header=1)
    elif table_base == 'Temperature':
        data = pd.read_csv('Appendix-data/8a-Saturated-Water-Temperature.csv', header=1)
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
    file_path_new = 'Appendix-data/8c-Superheated-Water.csv'
    data = pd.read_csv(file_path_new, header=1)

    # Convert Pressure and Temperature columns to numeric
    data[relative_to[0]] = pd.to_numeric(data[relative_to[0]], errors='coerce')
    data[relative_to[1]] = pd.to_numeric(data[relative_to[1]], errors='coerce')
    data[request] = pd.to_numeric(data[request], errors='coerce')

    # Prepare points and values
    points = np.column_stack((data[relative_to[0]], data[relative_to[1]]))

    output_value = griddata(points, data[request], (input[0], input[1]), method='linear')

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
        data = pd.read_csv('Appendix-data/9b-Saturated-R134a-Pressure.csv', header=1)
    elif table_base == 'Temperature':
        data = pd.read_csv('Appendix-data/9a-Saturated-R134a-Temperature.csv', header=1)
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
    file_path_new = 'Appendix-data/9c-Superheated-R134a.csv'
    data = pd.read_csv(file_path_new, header=1)

    # Convert Pressure and Temperature columns to numeric
    data[relative_to[0]] = pd.to_numeric(data[relative_to[0]], errors='coerce')
    data[relative_to[1]] = pd.to_numeric(data[relative_to[1]], errors='coerce')
    data[request] = pd.to_numeric(data[request], errors='coerce')

    # Prepare points and values
    points = np.column_stack((data[relative_to[0]], data[relative_to[1]]))

    output_value = griddata(points, data[request], (input[0], input[1]), method='linear')

    return np.float64(output_value) if np.ndim(output_value)==0 else output_value[0] if np.ndim(output_value)==1 else output_value