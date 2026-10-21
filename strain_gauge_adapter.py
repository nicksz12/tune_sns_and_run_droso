def return_empty_strain_gauge_adapter():
    sga = {"torque_sensor_index": 1, "sensory_neuron_index": None, "a": 1, "b": 0, "min_strain": -1000,
           "current_at_min_strain": 0, "max_strain": 1000, "current_at_max_strain": 0}
    return sga


def get_reading_from_mj(sgp, sgd, mj_data):
    """
    :param sgp: strain_gauge_props dictionary
    :param sgd: strain_gauge_data dictionary
    :param mj_data: mujoco data structure from which to extract the torque information
    """
    r = mj_data.sensordata[sgp["torque_sensor_index"]]
    sgd["reading"] = r

    return r


def write_state_to_sensory(sgp, sgd, sns_input=None):
    """
    :param sgp: strain_gauge_props dictionary
    :param sgd: strain_gauge_data dictionary
    :param sns_input: numpy array into which calculated current will be saved.
    """
    a = sgp["a"]
    b = sgp["b"]

    r = sgd["reading"]

    i = a * r + b

    if i > sgp["max_strain"]:
        i = sgp["current_at_max_strain"]
    elif i < sgp["min_strain"]:
        i = sgp["current_at_min_strain"]

    if sns_input is not None:
        sns_input[sgp["sensory_neuron_index"]] = i

    return i
