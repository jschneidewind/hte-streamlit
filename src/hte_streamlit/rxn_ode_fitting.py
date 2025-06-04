# code based on https://github.com/jschneidewind/Water-Splitting/blob/c615ae6a371363bd48091d46684c17c6fc815b62/Data_Analysis/Reaction_ODE_Fitting.py
# see https://www.rsc.org/suppdata/d1/ee/d1ee01053k/d1ee01053k1.pdf for a description
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import differential_evolution
from scipy.integrate import odeint


def plot_func(
    data,
    t,
    plot_type,
    labels=["A", "B", "C", "D", "E", "F"],
    ax=plt,
    show_labels=False,
    transpose=False,
    markersize=2,
):
    """Function to plot results of ODE fitting"""

    colors = ["blue", "orange", "green", "red", "black", "grey"]

    if transpose is True:
        data = data.T

    if show_labels is True:
        for counter, i in enumerate(data):
            ax.plot(
                t,
                i,
                "%s" % plot_type,
                color=colors[counter],
                markersize=markersize,
                label=labels[counter],
            )
    else:
        for counter, i in enumerate(data):
            ax.plot(
                t, i, "%s" % plot_type, color=colors[counter], markersize=markersize
            )


def convert_letters_to_numbers(string):
    """Converting letters in reaction string to numbers"""

    string = str.split(string, "+")

    output = []

    for item in string:
        character = item.strip(" ")
        value = ord(character) - 65
        output.append(value)

    return np.asarray(output)
    #return output


def correct_idx(idx, shapes):

    values = []

    for counter, i in enumerate(shapes):
        value = np.ones(i) * counter
        values.append(value)

    values = np.concatenate(np.asarray(values))
    idx_corrected = []

    for i in idx:
        idx_corrected.append(values[i])

    return np.asarray(idx_corrected, dtype="int")


def interpret_single_string(string):
    reaction_and_non_reactant_part = str.split(string, ",")
    reaction = reaction_and_non_reactant_part[0]
    non_reactant_part = reaction_and_non_reactant_part[1:]

    reactants, products = str.split(reaction, ">")

    non_reactant_components = []

    for item in non_reactant_part:
        item = item.strip(" ")

        component_type = re.sub(r"\d+", "", item)
        index = int(re.findall(r"\d+", item)[0]) - 1  # converting from 1 to 0 indexing

        non_reactant_components.append([component_type, index])

    return [
        convert_letters_to_numbers(reactants),
        convert_letters_to_numbers(products),
    ], non_reactant_components


def ode_string_interpreter_complete(string):
    reactions = []
    non_reactant_components = {}

    for counter, i in enumerate(string):
        reaction, non_reactant_component = interpret_single_string(i)
        reactions.append(reaction)

        non_reactant_components[counter] = non_reactant_component


    
    return np.asarray(reactions), non_reactant_components
    #return reactions, non_reactant_components


def consumption_production(
    matrix,
    non_reactant_components,
    mode,
    reaction_list_provided=False,
    reaction_list=None,
):
    number_reactants = np.amax(
        np.r_[np.concatenate(matrix[:, 0]), np.concatenate(matrix[:, 1])]
    )

    if reaction_list_provided is False:
        reactions = []
        for i in range(number_reactants + 1):
            reactions.append([])
    else:
        reactions = reaction_list

    if mode == "consume":
        ix = 0
        k = "-k"

    if mode == "produce":
        ix = 1
        k = "k"

    components = np.unique(np.concatenate(matrix[:, ix]))

    idx = []
    for _ in components:
        idx.append([])

    length = []
    for i in matrix[:, ix]:
        length.append(len(i))
    length = np.asarray(length)

    for counter, i in enumerate(components):
        idx_temp = np.where(np.concatenate(matrix[:, ix]) == i)[0]
        idx_temp = correct_idx(idx_temp, length)
        idx[counter].append(idx_temp)

    for counter, z in enumerate(idx):
        idx[counter] = [np.unique(z)]

    reax = dict(zip(components, idx))

    mapping = {"y": 0, "k": 1, "-k": 2, "hv": 3, "sigma": 4}

    for i in components:
        for j in reax[i][0]:
            temp = []

            for component_type, component_index in non_reactant_components[j]:
                if component_type == "k":
                    component_type = k

                part = [
                    mapping[component_type],
                    component_index,
                ]  # component_types are converted to integers based on mapping dict
                temp.append(part)

            for l in matrix[j][0]:
                y_part = [0, l]  # component type y is assigned integer 0
                temp.append(y_part)

            reactions[i].append(temp)

    return reactions


def ode_interpreter(matrix, non_reactant_components):
    reactions = consumption_production(matrix, non_reactant_components, "consume")
    reactions = consumption_production(
        matrix, non_reactant_components, "produce", "True", reactions
    )

    return reactions


def reaction_string_to_matrix(string):
    """Convert reaction string to matrix

    For example, interpret reaction strings like "A+B>C, k1" to create a matrix representation of the reaction system.

    Args:
        string (str): Reaction string

    Returns:
        matrix (list): Matrix representation of the reaction system
        k_number (int): Number of rate constants
        reactant_number (int): Number of reactants
    """
    reax, non_reactant_components = ode_string_interpreter_complete(string)

    reactant_number = len(
        np.unique(np.r_[np.concatenate(reax[:, 0]), np.concatenate(reax[:, 1])])
    )

    k_indices = []

    for _key, value in non_reactant_components.items():
        for item in value:
            if "k" in item:
                k_indices.append(item[1])

    k_number = len(np.unique(np.asarray(k_indices)))

    matrix = ode_interpreter(reax, non_reactant_components)

    return matrix, k_number, reactant_number


def ode_generator(y, t, output, matrix, k, flux, cross_section):
    """Generation of ODE system based on provided matrix (list)"""
    par = {0: y, 1: k, 2: (-1) * k, 3: flux, 4: cross_section}

    for counter, component in enumerate(matrix):
        r_component = 0

        for component_reaction in component:
            r_temp = 1.0

            for term in component_reaction:
                r_temp *= par[term[0]][term[1]]

            r_component += r_temp

        output[counter] = r_component

    return output


def ode_matrix_fit_func(
    k,
    initial_state,
    t,
    matrix,
    ravel=True,
    flux=np.array([1]),
    cross_section=np.array([1]),
    idx=None,
):
    """Numerically solving ODE system generated by ODE_generator using SciPy's odeint function."""

    ODE_function = ode_generator

    output = np.empty(len(initial_state))

    sol = odeint(
        ODE_function, initial_state, t, args=(output, matrix, k, flux, cross_section)
    )

    if idx is not None:
        sol = sol[:, idx]
    elif ravel is True:
        sol = np.ravel(sol)

    return sol


def residual_ode(P, initial, t, y, matrix, idx=None):
    """Computes residual from data to be fitted and fit"""

    y_fit = ode_matrix_fit_func(P, initial, t, matrix, idx=idx)
    res = np.sum((y - y_fit) ** 2)

    return res / 1000.0


def ode_fitting(
    data, t, reaction_string, idx=None, upper_bound=1.0, bounds_arr=None, workers=1, 
    initial_state_multiplier = 4
):
    """Fitting of ODE function to data. idx indicates which column of the generated results is fit to the data.

    Use SciPy's differential_evolution to optimize the rate constants
    by minimizing the residual between the ODE solution and the experimental data. initial_state_multiplier 
    adjusts initial concentration of reactant A to be high enough for accurate kinetic modelling.
    """

    matrix, k_number, reactant_number = reaction_string_to_matrix(reaction_string)

    initial_state = np.zeros(reactant_number)
    initial_state[0] = np.amax(data) * initial_state_multiplier

    if bounds_arr is None:
        bounds_arr = []  # bounds for rate constants in optimization
        for i in range(k_number):
            bounds_arr.append([0.0, upper_bound])

    p = differential_evolution(
        func=residual_ode,
        bounds=bounds_arr,
        args=(initial_state, t, data, matrix, idx),
        workers=workers,
    )
    p_ode = p.x
    residual = p.fun

    return p_ode, matrix, initial_state, residual

if __name__ == "__main__":

    data = np.arange(0, 100, 1) * 3
    time = np.arange(0, 100, 1)

    reaction_string = ["A + C > B, k1",
                       'B > C + E, k2',
                       'C > D, k3']

    p_ode, matrix, initial_state, residual = ode_fitting(data, time, reaction_string, idx = 1)


    