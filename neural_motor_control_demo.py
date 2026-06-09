import mujoco
import mujoco.viewer
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import datetime
from joint_adapter import JointAdapter
from pathlib import Path
from plot_data import plot_data

from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.renderer import render
from sns_toolbox.neurons import NonSpikingNeuronWithPersistentSodiumChannel


def main():
    '''

    :return: dict: Neural states and mechanical states
    :return: path: Path of .json file storing neural states and mechanical states
    '''
    dt = datetime.datetime

    st_sw_ratio = 4
    sw_duration = 1.25
    max_speed = 0.5
    servo_kp = 8.5
    num_loops = 8
    tmax = 2 * sw_duration * num_loops

    r_net = 20

    # Load robot model
    mech_model_file_path = 'mjcf/Drosophibot_floating_custom_actuators20260606.xml'
    drosophibot = mujoco.MjModel.from_xml_path(filename=mech_model_file_path)

    # Get robot state
    state_struct = mujoco.MjData(drosophibot)

    # Load starting pose and control inputs
    mujoco.mj_resetData(drosophibot, state_struct)

    # Set the physics step
    sns_dt = 0.1  # ms
    mj_dt = 0.001  # s
    drosophibot.opt.timestep = mj_dt

    # Configure how SNS and physics steps will interact
    sns_dt_in_seconds = sns_dt * 1e-3

    if sns_dt_in_seconds < mj_dt:
        #  sns timesteps are shorter, so we need to run multiple sns steps before updating Mujoco
        sns_shorter = True
        min_dt = sns_dt_in_seconds
        update_ratio = int(np.round(mj_dt / sns_dt_in_seconds))
    elif sns_dt_in_seconds > mj_dt:
        #  Mujoco timesteps are shorter, so we need to run multiple Mujoco steps before updating sns
        sns_shorter = False
        min_dt = mj_dt
        raise "sns_dt must be less than mj_dt when measured in seconds"
    else:
        #  Timesteps are equal and the situation is straightforward
        sns_shorter = None
        min_dt = sns_dt_in_seconds

    tvec = np.arange(start=0, step=min_dt, stop=tmax)
    sns_tvec = np.arange(start=0, step=sns_dt_in_seconds, stop=tmax)
    mj_tvec = np.arange(start=0, step=mj_dt, stop=tmax)
    num_steps = len(tvec)
    num_sns_steps = len(sns_tvec)
    num_mj_steps = len(mj_tvec)

    g_ion_new = sp.optimize.bisect(calc_rg_equilibrium, 0, 5)
    print("Ca conductance is " + str(g_ion_new))
    print("Equilibrium voltage should be " + str(r_net + calc_rg_equilibrium(g_ion_new, to_plot=False)))

    net = Network(name="droso1leg")

    mx_28_no_load_speed_rpm = 55  # At 12V power supply
    mx_28_no_load_speed = mx_28_no_load_speed_rpm * 2 * np.pi / 60
    mx_64_no_load_speed_rpm = 63  # At 12V power supply
    mx_64_no_load_speed = mx_64_no_load_speed_rpm * 2 * np.pi / 60

    all_names = {}
    body_name = "droso"
    #  Critically, k_step_len and k_rotate will be set based on inverse kinematics. This is the purpose of Ava's work. I
    #  will integrate this in the future.
    generic_names = build_body_control_net(net, r_net=r_net, body_name=body_name)
    all_names[body_name] = generic_names

    #  Build
    joint_props = {}
    j_a = {}

    joint_props["LF_ThC1"] = {"ctrl_index": None, "qpos_index": None, "is_ctr": False, "ext_pep_rest_volt": -5,
                              "flx_pep_rest_volt": -5}
    joint_props["LF_ThC3"] = {"ctrl_index": None, "qpos_index": None, "is_ctr": False, "ext_pep_rest_volt": -5,
                              "flx_pep_rest_volt": -5}
    joint_props["LF_CTr"] = {"ctrl_index": None, "qpos_index": None, "is_ctr": True, "ext_pep_rest_volt": -5,
                             "flx_pep_rest_volt": -5}
    joint_props["LF_TrF"] = {"ctrl_index": None, "qpos_index": None, "is_ctr": False, "ext_pep_rest_volt": -5,
                             "flx_pep_rest_volt": -5}
    joint_props["LF_FTi"] = {"ctrl_index": None, "qpos_index": None, "is_ctr": False, "ext_pep_rest_volt": -5,
                             "flx_pep_rest_volt": -5}

    for key, value in joint_props.items():
        joint_props[key]["ctrl_index"] = mujoco.mj_name2id(drosophibot, mujoco.mjtObj.mjOBJ_ACTUATOR, name=key)
        joint_props[key]["qpos_index"] = mujoco.mj_name2id(drosophibot, mujoco.mjtObj.mjOBJ_JOINT, name=key)

        generic_names = build_joint_control_net(net, r_net=r_net, is_ctr=joint_props[key]["is_ctr"], joint_name=key,
                                                ext_pep_rest=joint_props[key]["ext_pep_rest_volt"],
                                                flx_pep_rest=joint_props[key]["flx_pep_rest_volt"], gmax_ion=g_ion_new,
                                                tau_slow=1400, st_sw_ratio=st_sw_ratio)
        all_names[key] = generic_names

        generic_names = build_descending_pathways(net, r_net=r_net, k_step_len=0.9, k_rotate=0, joint_name=key,
                                                  ext_in_stance=True, body_name=body_name)
        all_names[key].update(generic_names)  # generic_names is currently empty, so it just goofs up all_names.
        # replace ctrl_index, etc. with joint_props dict

        j_a[key] = JointAdapter(name=key, joint_props=value,
                                mn_indices=[all_names[key]["Ext_MN"], all_names[key]["Flx_MN"]],
                                sens_neur_indices=[all_names[key]["Ext_Pos_SN"], all_names[key]["Flx_Pos_SN"],
                                                   all_names[key]["Ext_Vel_SN"], all_names[key]["Flx_Vel_SN"]],
                                resting_posture_angle=0, min_angle=-0.5, max_angle=0.5, swing_duration=sw_duration,
                                max_stance_duration=st_sw_ratio*sw_duration, servo_max_speed=mx_28_no_load_speed,
                                servo_kp=servo_kp)

        # j_a[key].verify_motor_maps()

    sns_model = net.compile(dt=sns_dt, backend='numpy', device='cpu', debug=False)
    sns_inputs = np.zeros([num_sns_steps, net.get_num_inputs()])
    #  To step with a constant step length:
    sns_inputs[np.arange(0, np.floor(4 * num_sns_steps / 5), dtype=int), net.get_input_index(
        'droso_step_length_input')] = 0
    #  To step forward then reverse direction:
    sns_inputs[:, net.get_input_index('droso_step_length_input')] = -10 * sp.special.erf((sns_tvec - 2 * tmax / 3))
    #  sns_inputs[:, net.get_input_index('body_step_length_input')] = 10*sp.special.erf((sns_tvec - 2*tmax/3))

    for key, value in joint_props.items():
        rg = key + "_Ext_RG_input"
        sns_inputs[0, net.get_input_index(rg)] = 1
        # sns_inputs[np.arange(0, np.floor(num_sns_steps/5), dtype=int), net.get_input_index(rg)] = -10

    sns_outputs = np.zeros([num_sns_steps, net.get_num_outputs()])
    joint_pos = np.zeros([num_mj_steps, len(j_a)])
    joint_pos_command = np.zeros([num_mj_steps, len(j_a)])
    joint_vel = np.zeros([num_mj_steps, len(j_a)])
    joint_vel_command = np.zeros([num_mj_steps, len(j_a)])
    joint_torque = np.zeros([num_mj_steps, len(j_a)])

    i = int(0)
    i_mj = int(0)
    t = 0

    render(net)

    # Initialize viewer
    viewer = mujoco.viewer.launch_passive(drosophibot, state_struct, key_callback=key_callback)
    step_mj(viewer, drosophibot, state_struct)

    # Main loop
    while i < num_steps:

        if np.mod(i, update_ratio) == 0:
            #  Inform the joint adapter what the MN voltages are
            for key in joint_props.keys():
                #  j_a[key].mn_to_servo(v_all=sns_outputs[i-1, j_a[key].mn_indices])
                j_a[key].mn_to_servo(v_all=sns_outputs[i - 1, :])

                #  Read current joint state and calculated commanded values for the servo
                [current_pos, current_vel, desired_pos, desired_vel] = j_a[key].set_servo_commands(state_struct)

                #  Write the servo commands to the MuJoCo model. This could be modified/replaced to communicate with a robot.
                j_a[key].write_servo_commands(state_struct)

                #  Save values for logging.
                #  Gotta expand size of joint_pos and joint_vel to account for all the actuated joints.
                joint_pos[i_mj, j_a[key].qpos_index] = current_pos
                joint_pos_command[i_mj, j_a[key].ctrl_index] = desired_pos
                joint_vel[i_mj, j_a[key].qpos_index] = current_vel
                joint_vel_command[i_mj, j_a[key].ctrl_index] = desired_vel
                joint_torque[i_mj, j_a[key].ctrl_index] = state_struct.actuator_force[j_a[key].ctrl_index]

            #  Step the MuJuCo model forward. This could be modified/replaced to communicate with a robot.
            #  np.mod(i, 1000) == 0 sets whether to update the video or not.
            # step_mj(viewer, drosophibot, state_struct, to_sync=False)
            step_mj(viewer, drosophibot, state_struct, to_sync=np.mod(i, 1000) == 0)

            i_mj += 1

        #  This writes the stored mechanical states to neural currents. Even if mechanics/robot did not update in this
        #  loop, this method acts like "memory" to give the neural system useful information, so we need it in every
        #  loop.
        for key in joint_props.keys():
            j_a[key].write_state_to_sensory(sns_inputs[i, :])

        sns_outputs[i, :] = sns_model(sns_inputs[i, :])
        # print(state.qvel[index])
        t += min_dt
        i = i + 1

    #  Pack up data
    data_pack = {}
    #  store time
    if sns_shorter:
        data_pack["time"] = mj_tvec.tolist()
    else:
        data_pack["time"] = tvec.tolist()
    #  store joint angles
    for key in joint_props.keys():
        data_pack[str(key) + "_pos"] = joint_pos[:, joint_props[key]["qpos_index"]].tolist()
        data_pack[str(key) + "_pos_comm"] = joint_pos_command[:, joint_props[key]["ctrl_index"]].tolist()
        data_pack[str(key) + "_vel"] = joint_vel[:, joint_props[key]["qpos_index"]].tolist()
        data_pack[str(key) + "_vel_comm"] = joint_vel_command[:, joint_props[key]["ctrl_index"]].tolist()
        data_pack[str(key) + "_torque"] = joint_pos[:, joint_props[key]["qpos_index"]].tolist()
    #  store neural states
    if sns_shorter:
        #  Downsample neural data to the physics timestep. THIS MAY CAUSE PROBLEMS WITH SPIKING DATA; SPIKES WILL BE DROPPED
        for net_key in all_names.keys():
            for neur_key, neur_val in all_names[net_key].items():
                data_pack[str(net_key) + "_" + str(neur_key) + "_volt"] = np.interp(mj_tvec, tvec,
                                                                                    sns_outputs[:,
                                                                                    neur_val]).tolist()
                data_pack[str(net_key) + "_" + str(neur_key) + "_input"] = np.interp(mj_tvec, tvec,
                                                                                     sns_inputs[:,
                                                                                     neur_val]).tolist()
    else:
        for net_key in all_names.keys():
            for neur_key, neur_val in all_names[net_key].items():
                data_pack[str(net_key) + "_" + str(neur_key) + "_volt"] = sns_outputs[:, neur_val].tolist()
                data_pack[str(net_key) + "_" + str(neur_key) + "_input"] = sns_inputs[:, neur_val].tolist()
    #  save data_pack in a time- and dated .json file
    data_file_str = "droso_sns_" + dt.today().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(data_file_str, "w", encoding="utf-8") as file:
        json.dump(data_pack, file, indent=4)

    data_file_path = Path(data_file_str)

    prev_file_size = -1
    new_file_size = 1
    while new_file_size != prev_file_size:
        prev_file_size = new_file_size
        new_file_size = data_file_path.stat().st_size
        print(f"File size: {new_file_size} bytes.")
        #  Kill time, wait for it to be published.

    plot_data(data_file_path)

    return data_pack, data_file_path




def key_callback(keycode):
    if chr(keycode) == ' ':
        global max_speed
        max_speed = float(input('Enter max speed in rad/s'))
        print(max_speed)


def step_mj(viewer, model, state, to_sync=True):
    """
    Simulate the model for one step and update the viewer
    :param viewer: Mujoco viewer
    :param model: Model to simulate
    :param state: State to apply to the model
    :param to_sync: Boolean of whether to show state changes in the viewer (i.e., whether to animate)
    :return:
    """
    mujoco.mj_step(model, state)
    if to_sync:
        viewer.sync()


def calc_rg_equilibrium(g_ion, r=20, mh_slope=1/20, tau_slow=200, to_plot=False):
    """

    :param g_ion: max ionic conductance, which will drive equilibrium voltage to r.
    :param r: max expected neuron voltage + synaptic conductance saturation value
    :param mh_slope: dm/dV and -dh/dV, determines how switch-like ionic current conductance is
    :param tau_slow: time constant of h, the deactivation variable. Determines relaxation rate to equilibrium.
    :param to_plot: boolean of whether to plot response.
    :return: (float) difference between equilibrium voltage and r.
    """
    rg_neuron = NonSpikingNeuronWithPersistentSodiumChannel(g_ion=[g_ion], slope_m=[mh_slope],
                                                            k_h=[0.5], slope_h=[-mh_slope], tau_max_h=[tau_slow])
    net = Network("single_rg")
    net.add_neuron(rg_neuron, name="rg")
    net.add_output(source="rg", name="rg_out")
    model = net.compile(dt=0.1)

    v_prev = -1
    v_new = 1
    v_tol = 1e-6
    i = int(0)
    i_max = int(1e4)
    v = np.zeros([i_max])
    while abs(v_prev - v_new) > v_tol and i < i_max:
        v_prev = v_new
        v_new = model()
        v[i] = v_new
        i = i + 1

    if to_plot:
        plt.figure()
        plt.plot(v, label='voltage over time')
        plt.draw()
        plt.show()

    if i == i_max:
        return None
    else:
        return v_new[0] - r


def zinf(U, Kz, Sz, Ez):
    return 1/(1+Kz*np.exp(Sz*(Ez-U)))


def tauz(U, tauzmax, Kz, Sz, Ez):
    """

    :param U: Instantaneous membrane voltage
    :param tauzmax: Maximum ionic current gating variable (z) time constant
    :param Kz: Scaling factor
    :param Sz: Maximum ionic current slope dz/dU
    :param Ez: Membrane voltage at which gating variable is 50% activated
    :return: (float) Instantaneous gating variable time constant
    """
    return tauzmax*zinf(U, Kz, Sz, Ez)*np.sqrt(Kz*np.exp(Sz*(Ez-U)))


def build_body_control_net(network, r_net, body_name):
    """

    :param network: SNS-Toolbox network dictionary to append to
    :param r_net: max expected neuron voltage + synaptic conductance saturation value
    :param body_name: string label for neurons at the "body" level
    :return: (dict) neuron names and their index within the network
    """
    #  These are neurons that set overall body state (e.g., walking, standing).
    fast_neuron = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5)
    fast_neuron_half_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, bias=r_net / 2)
    fast_neuron_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, bias=r_net)

    e_in = -100
    g_in = g_syn_from_k(k=-2, rev_pot=e_in, volt_range=r_net)
    print('g_in in walking logic net = ' + str(g_in))
    hyp_k_neg_2 = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_in, reversal_potential=e_in)
    hyp_binary_k_neg_1 = NonSpikingSynapse(e_lo=r_net / 2, e_hi=r_net / 2 + 0.5, max_conductance=g_in,
                                           reversal_potential=e_in)

    generic_names = {}

    #  Step length neurons
    step_length = "step_length"
    generic_names[step_length] = network.get_num_neurons()
    step_length_full_name = add_neuron_input_output(network, fast_neuron_half_tonic, step_length, body_name)
    #  Full name is required to create synapses.

    #  Backward step length, which is tonically active but inhibited by step length neurons.
    negative_step_length = "negative_step_length"
    generic_names[negative_step_length] = network.get_num_neurons()
    negative_step_length_full_name = add_neuron_input_output(network, fast_neuron_tonic,
                                                             negative_step_length, body_name)

    #  Not-walking neuron, which is inhibited by boh step length and negative step length.
    not_walking = "not_walking"
    generic_names[not_walking] = network.get_num_neurons()
    not_walking_full_name = add_neuron_input_output(network, fast_neuron_tonic, not_walking, body_name)

    #  Step rotation (body rotation per stance phase) neurons will also be added in the future.

    network.add_connection(hyp_k_neg_2, source=step_length_full_name, destination=negative_step_length_full_name)
    network.add_connection(hyp_binary_k_neg_1, source=step_length_full_name, destination=not_walking_full_name)
    network.add_connection(hyp_binary_k_neg_1, source=negative_step_length_full_name, destination=not_walking_full_name)

    return generic_names


def build_leg_control_net(network, r_net, leg_name):
    """

    :param network: SNS-Toolbox network dictionary to append to
    :param r_net: max expected neuron voltage + synaptic conductance saturation value
    :param leg_name: string label for neurons within the "leg" level network
    :return: (dict) neuron names and their index within the network
    """
    #  These are neurons that all joints in the leg can receive signals from (e.g., walking state) and send signals to
    #  (e.g., leg is moving).
    fast_neuron = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5)
    fast_neuron_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, bias=r_net)

    generic_names = {}

    #  "Moving in stance" neurons. These reinforce stance phase pattern generator activity.
    moving_in_stance = "moving_in_stance"
    generic_names[moving_in_stance] = network.get_num_neurons()
    moving_in_stance_full_name = add_neuron_input_output(network, fast_neuron, moving_in_stance, leg_name)
    #  Full name is required to create synapses.

    #  "AEP reached" neurons. These trigger leg depression.
    aep_reached = "aep_reached"
    generic_names[aep_reached] = network.get_num_neurons()
    aep_reached_full_name = add_neuron_input_output(network, fast_neuron, aep_reached, leg_name)

    #  "PEP reached" neurons. These trigger leg elevation.
    pep_reached = "pep_reached"
    generic_names[pep_reached] = network.get_num_neurons()
    pep_reached_full_name = add_neuron_input_output(network, fast_neuron, pep_reached, leg_name)


def build_descending_pathways(network, r_net, k_step_len, k_rotate, joint_name, ext_in_stance, body_name):
    """

    :param network: SNS-Toolbox network dictionary to append to
    :param r_net: max expected neuron voltage + synaptic conductance saturation value
    :param k_step_len: gain between neural representation of body translation per step and of this joint's range of
                       motion
    :param k_rotate: gain between neural representation of body rotation per step and of this joint's range of motion
    :param joint_name: string label for neurons within the "joint" level network
    :param ext_in_stance:
    :param body_name: string label for neurons at the "body" level
    :return: (dict) neuron names and their index within the network
    """
    #  This will be called for each joint.
    e_ex = 220
    generic_names = {}

    g_step_len = g_syn_from_k(k=k_step_len, rev_pot=e_ex, volt_range=r_net)
    g_rotate = g_syn_from_k(k=k_rotate, rev_pot=e_ex, volt_range=r_net)

    syn_type_step_len = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=e_ex, max_conductance=g_step_len)
    syn_type_rotate = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=e_ex, max_conductance=g_rotate)

    fast_neuron_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, resting_potential=r_net)

    step_length_full_name = body_name + "_step_length"
    negative_step_length_full_name = body_name + "_negative_step_length"
    step_rotation_full_name = ""

    pep_plus_full_name = joint_name + "_PEP+"
    pep_minus_full_name = joint_name + "_PEP-"
    if ext_in_stance:
        network.add_connection(syn_type_step_len, source=step_length_full_name, destination=pep_plus_full_name)
        network.add_connection(syn_type_step_len, source=negative_step_length_full_name,
                               destination=pep_minus_full_name)
    else:
        network.add_connection(syn_type_step_len, source=step_length_full_name, destination=pep_minus_full_name)
        network.add_connection(syn_type_step_len, source=negative_step_length_full_name, destination=pep_plus_full_name)

    not_walking_IN = "not_walking_IN"
    generic_names[not_walking_IN] = network.get_num_neurons()
    not_walking_IN_full_name = add_neuron_input_output(network, fast_neuron_tonic, not_walking_IN, joint_name)

    e_mult = -1
    g_mult = -r_net / e_mult
    hyp_mult = NonSpikingSynapse(max_conductance=g_mult, reversal_potential=e_mult, e_lo=0, e_hi=r_net)

    not_walking_full_name = body_name + "_not_walking"
    network.add_connection(hyp_mult, source=not_walking_full_name, destination=not_walking_IN_full_name)

    exting_dend_full_name = joint_name + "_Exting_Dendrite"
    flxing_dend_full_name = joint_name + "_Flxing_Dendrite"

    network.add_connection(hyp_mult, source=not_walking_IN_full_name, destination=exting_dend_full_name)
    network.add_connection(hyp_mult, source=not_walking_IN_full_name, destination=flxing_dend_full_name)

    return generic_names


def build_joint_control_net(network, r_net, is_ctr, joint_name, ext_pep_rest, flx_pep_rest, gmax_ion=1.049,
                            mh_slope=1/20, tau_slow=2000, st_sw_ratio=4):
    """

    :param network: SNS-Toolbox network dictionary to append to
    :param r_net: max expected neuron voltage + synaptic conductance saturation value
    :param is_ctr: boolean of whether this joint is the coxa-trochanter joint, which has special neural control
    :param joint_name: string label for neurons within the "joint" level network
    :param ext_pep_rest: TBD
    :param flx_pep_rest: TBD
    :param gmax_ion: max ionic conductance
    :param mh_slope: dm/dV and -dh/dV, determines how switch-like ionic current conductance is
    :param tau_slow: time constant of h, the deactivation variable. Determines relaxation rate to equilibrium.
    :param st_sw_ratio: ratio of maximum stance duration to (constant) swing duration. Effectively range of speeds.
    :return: (dict) neuron names and their index within the network
    """
    #  Neuron type definitions:
    fast_neuron = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5)
    fast_neuron_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, resting_potential=r_net)
    rg_neuron = NonSpikingNeuronWithPersistentSodiumChannel(g_ion=[gmax_ion], e_ion=[110], k_m=[1], slope_m=[mh_slope],
                                                            e_m=[r_net], k_h=[0.5], slope_h=[-mh_slope], e_h=[0],
                                                            tau_max_h=[tau_slow])

    #  ext_is_pep and flx_is_pep neurons require joint-specific property values to control motion.
    if is_ctr:
        i_bias_ext = 5 - ext_pep_rest
    else:
        i_bias_ext = 0

    i_bias_flx = 0
    fast_neuron_ext_pep = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, bias=i_bias_ext,
                                           resting_potential=ext_pep_rest)
    fast_neuron_flx_pep = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5, bias=i_bias_flx,
                                           resting_potential=flx_pep_rest)

    #  Synapse type definitions:
    e_ex = 220
    g_k_1 = g_syn_from_k(k=1, rev_pot=e_ex, volt_range=r_net)
    dep_k_1 = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_k_1, reversal_potential=e_ex)

    e_in = -100
    e_in2 = -40  # Should be set by analysis
    hyp_bistable = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=0.2, reversal_potential=e_in)
    e_gain = -1
    g_gain = -r_net / e_gain
    g_cpg = 0.95*0.517  # Should be set by analysis
    hyp_gain = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_gain, reversal_potential=e_gain)
    hyp_gain_strong = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_gain, reversal_potential=e_gain)
    hyp_cpg = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_cpg, reversal_potential=e_in2)

    g_k_neg_1 = g_syn_from_k(k=-1, rev_pot=e_in, volt_range=r_net)
    hyp_k_neg_1 = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_k_neg_1, reversal_potential=e_in)
    print('hyp_k_neg_1, g = ' + str(g_k_neg_1))

    generic_names = {}

    #  Every joint has PEP+ and PEP- neurons.
    ext_is_pep = "PEP+"
    generic_names[ext_is_pep] = network.get_num_neurons()
    ext_is_pep_full_name = add_neuron_input_output(network, fast_neuron_ext_pep, ext_is_pep, joint_name)
    #  Full name is required to create synapses.

    flx_is_pep = "PEP-"
    generic_names[flx_is_pep] = network.get_num_neurons()
    flx_is_pep_full_name = add_neuron_input_output(network, fast_neuron_flx_pep, flx_is_pep, joint_name)

    #  PEP+ and PEP- feed into "Coact_MNs" neuron.
    coact = "Coact_MNs"
    generic_names[coact] = network.get_num_neurons()
    coact_full_name = add_neuron_input_output(network, fast_neuron, coact, joint_name)

    #  PEP+ and PEP- also feed into "Ext is Stance" and "Flx is Stance" neurons.
    ext_is_stance = "Ext_is_Stance"
    generic_names[ext_is_stance] = network.get_num_neurons()
    ext_is_stance_full_name = add_neuron_input_output(network, fast_neuron_tonic, ext_is_stance, joint_name)

    flx_is_stance = "Flx_is_Stance"
    generic_names[flx_is_stance] = network.get_num_neurons()
    flx_is_stance_full_name = add_neuron_input_output(network, fast_neuron_tonic, flx_is_stance, joint_name)

    #  Ext_RG and Flx_RG
    ext_rg = "Ext_RG"
    generic_names[ext_rg] = network.get_num_neurons()
    ext_rg_full_name = add_neuron_input_output(network, rg_neuron, ext_rg, joint_name)

    flx_rg = "Flx_RG"
    generic_names[flx_rg] = network.get_num_neurons()
    flx_rg_full_name = add_neuron_input_output(network, rg_neuron, flx_rg, joint_name)

    ext_mn = "Ext_MN"
    generic_names[ext_mn] = network.get_num_neurons()
    ext_mn_full_name = add_neuron_input_output(network, fast_neuron, ext_mn, joint_name)

    flx_mn = "Flx_MN"
    generic_names[flx_mn] = network.get_num_neurons()
    flx_mn_full_name = add_neuron_input_output(network, fast_neuron, flx_mn, joint_name)

    #  Add connections among these first 5 neurons
    network.add_connection(dep_k_1, source=ext_is_pep_full_name, destination=coact_full_name)
    network.add_connection(dep_k_1, source=flx_is_pep_full_name, destination=coact_full_name)
    network.add_connection(dep_k_1, source=ext_is_pep_full_name, destination=ext_is_stance_full_name)
    network.add_connection(dep_k_1, source=flx_is_pep_full_name, destination=flx_is_stance_full_name)

    k_ff = 1 / st_sw_ratio
    g_ff_pos = g_syn_from_k(k=k_ff, rev_pot=e_ex, volt_range=r_net)
    dep_nsi_ff = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=e_ex, max_conductance=g_ff_pos)

    network.add_connection(dep_nsi_ff, source=coact_full_name, destination=ext_mn_full_name)
    network.add_connection(dep_nsi_ff, source=coact_full_name, destination=flx_mn_full_name)

    network.add_connection(hyp_bistable, source=ext_is_stance_full_name, destination=flx_is_stance_full_name)
    network.add_connection(hyp_bistable, source=flx_is_stance_full_name, destination=ext_is_stance_full_name)

    #  Not a typo, these are deliberately "crossed" in this way
    network.add_connection(hyp_gain_strong, source=ext_rg_full_name, destination=flx_mn_full_name)
    network.add_connection(hyp_gain_strong, source=flx_rg_full_name, destination=ext_mn_full_name)

    network.add_connection(hyp_cpg, source=ext_rg_full_name, destination=flx_rg_full_name)
    network.add_connection(hyp_cpg, source=flx_rg_full_name, destination=ext_rg_full_name)

    #  Sensory neurons (sn) that process joint angle and velocity feedback
    ext_pos_sn = "Ext_Pos_SN"
    generic_names[ext_pos_sn] = network.get_num_neurons()
    ext_pos_sn_full_name = add_neuron_input_output(network, fast_neuron, ext_pos_sn, joint_name)

    flx_pos_sn = "Flx_Pos_SN"
    generic_names[flx_pos_sn] = network.get_num_neurons()
    flx_pos_sn_full_name = add_neuron_input_output(network, fast_neuron, flx_pos_sn, joint_name)

    ext_vel_sn = "Ext_Vel_SN"
    generic_names[ext_vel_sn] = network.get_num_neurons()
    ext_vel_sn_full_name = add_neuron_input_output(network, fast_neuron, ext_vel_sn, joint_name)

    flx_vel_sn = "Flx_Vel_SN"
    generic_names[flx_vel_sn] = network.get_num_neurons()
    flx_vel_sn_full_name = add_neuron_input_output(network, fast_neuron, flx_vel_sn, joint_name)

    too_exted = "Too_Exted"
    generic_names[too_exted] = network.get_num_neurons()
    too_exted_full_name = add_neuron_input_output(network, fast_neuron, too_exted, joint_name)

    not_exted = "Not_Exted"
    generic_names[not_exted] = network.get_num_neurons()
    not_exted_full_name = add_neuron_input_output(network, fast_neuron, not_exted, joint_name)

    too_flxed = "Too_Flxed"
    generic_names[too_flxed] = network.get_num_neurons()
    too_flxed_full_name = add_neuron_input_output(network, fast_neuron, too_flxed, joint_name)

    not_flxed = "Not_Flxed"
    generic_names[not_flxed] = network.get_num_neurons()
    not_flxed_full_name = add_neuron_input_output(network, fast_neuron, not_flxed, joint_name)

    network.add_connection(dep_k_1, source=ext_pos_sn_full_name, destination=too_exted_full_name)
    network.add_connection(dep_k_1, source=flx_pos_sn_full_name, destination=too_flxed_full_name)
    network.add_connection(dep_k_1, source=coact_full_name, destination=not_exted_full_name)
    network.add_connection(dep_k_1, source=coact_full_name, destination=not_flxed_full_name)

    network.add_connection(hyp_k_neg_1, source=coact_full_name, destination=too_exted_full_name)
    network.add_connection(hyp_k_neg_1, source=coact_full_name, destination=too_flxed_full_name)
    network.add_connection(hyp_k_neg_1, source=ext_pos_sn_full_name, destination=not_exted_full_name)
    network.add_connection(hyp_k_neg_1, source=flx_pos_sn_full_name, destination=not_flxed_full_name)

    network.add_connection(hyp_gain, source=ext_rg_full_name, destination=not_flxed_full_name)
    network.add_connection(hyp_gain, source=ext_rg_full_name, destination=too_flxed_full_name)
    network.add_connection(hyp_gain, source=flx_rg_full_name, destination=not_exted_full_name)
    network.add_connection(hyp_gain, source=flx_rg_full_name, destination=too_exted_full_name)

    #  Limit MN activity if joint is too extended
    k_position_fb = st_sw_ratio / (st_sw_ratio + 1)  # n / (n + 1)
    g_fb_negative = g_syn_from_k(k=-k_position_fb, rev_pot=e_in, volt_range=r_net)

    k_vel_fb = 1.286 - 0.2863206847
    g_fb_positive = g_syn_from_k(k=k_vel_fb, rev_pot=e_ex, volt_range=r_net)

    dep_nsi_fb = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=e_ex, max_conductance=g_fb_positive)
    hyp_nsi_fb = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=e_in, max_conductance=g_fb_negative)

    network.add_connection(hyp_nsi_fb, source=too_exted_full_name, destination=ext_mn_full_name)
    network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=too_flxed_full_name)

    network.add_connection(hyp_nsi_fb, source=too_flxed_full_name, destination=flx_mn_full_name)
    network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=too_exted_full_name)

    # Set up negative velocity FB to the MNs, which is disabled during locomotion.
    exting_dend = "Exting_Dendrite"
    generic_names[exting_dend] = network.get_num_neurons()
    exting_dend_full_name = add_neuron_input_output(network, fast_neuron, exting_dend, joint_name)

    flxing_dend = "Flxing_Dendrite"
    generic_names[flxing_dend] = network.get_num_neurons()
    flxing_dend_full_name = add_neuron_input_output(network, fast_neuron, flxing_dend, joint_name)

    network.add_connection(dep_k_1, source=ext_vel_sn_full_name, destination=exting_dend_full_name)
    network.add_connection(dep_k_1, source=flx_vel_sn_full_name, destination=flxing_dend_full_name)

    network.add_connection(dep_k_1, source=exting_dend_full_name, destination=flx_mn_full_name)
    network.add_connection(dep_k_1, source=flxing_dend_full_name, destination=ext_mn_full_name)

    if not is_ctr:
        # Add more neurons that control RG activation and report position to other joints

        #  "Ext is Stance" and "Flx is Stance" neurons feed into "Flx is Stance Load" and "Ext is Stance Load".
        ext_stance_load = "Ext_is_Stance_Load"
        generic_names[ext_stance_load] = network.get_num_neurons()
        ext_stance_load_full_name = add_neuron_input_output(network, fast_neuron, ext_stance_load, joint_name)

        flx_stance_load = "Flx_is_Stance_Load"
        generic_names[flx_stance_load] = network.get_num_neurons()
        flx_stance_load_full_name = add_neuron_input_output(network, fast_neuron, flx_stance_load, joint_name)

        network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=flx_stance_load_full_name)
        network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=ext_stance_load_full_name)

        exting = "Exting_in_Stance"
        generic_names[exting] = network.get_num_neurons()
        exting_full_name = add_neuron_input_output(network, fast_neuron, exting, joint_name)

        flxing = "Flxing_in_Stance"
        generic_names[flxing] = network.get_num_neurons()
        flxing_full_name = add_neuron_input_output(network, fast_neuron, flxing, joint_name)

        network.add_connection(dep_k_1, source=ext_vel_sn_full_name, destination=exting_full_name)
        network.add_connection(dep_k_1, source=flx_vel_sn_full_name, destination=flxing_full_name)

        # State-dependent silencing of positive velocity FB to CPGs.
        network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=flxing_full_name)
        network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=exting_full_name)

        # Positive feedback to CPG.
        network.add_connection(dep_nsi_fb, source=exting_full_name, destination=ext_rg_full_name)
        network.add_connection(dep_nsi_fb, source=flxing_full_name, destination=flx_rg_full_name)

        # Positive feedback to MNs; still hashing this out.
        # network.add_connection(dep_nsi_fb, source=exting_full_name, destination=ext_mn_full_name)
        # network.add_connection(dep_nsi_fb, source=flxing_full_name, destination=flx_mn_full_name)

        # Joint-level determination of whether AEP or PEP have been reached. This will ultimately be replaced by
        # leg-level network that determines where the foot is in space.
        ext_is_stance_AEP = "Ext_is_Stance_AEP"
        generic_names[ext_is_stance_AEP] = network.get_num_neurons()
        ext_is_stance_AEP_full_name = add_neuron_input_output(network, fast_neuron, ext_is_stance_AEP, joint_name)

        flx_is_stance_AEP = "Flx_is_Stance_AEP"
        generic_names[flx_is_stance_AEP] = network.get_num_neurons()
        flx_is_stance_AEP_full_name = add_neuron_input_output(network, fast_neuron, flx_is_stance_AEP, joint_name)

        ext_is_stance_PEP = "Ext_is_Stance_PEP"
        generic_names[ext_is_stance_PEP] = network.get_num_neurons()
        ext_is_stance_PEP_full_name = add_neuron_input_output(network, fast_neuron, ext_is_stance_PEP, joint_name)

        flx_is_stance_PEP = "Flx_is_Stance_PEP"
        generic_names[flx_is_stance_PEP] = network.get_num_neurons()
        flx_is_stance_PEP_full_name = add_neuron_input_output(network, fast_neuron, flx_is_stance_PEP, joint_name)

        network.add_connection(dep_k_1, source=not_flxed_full_name, destination=ext_is_stance_AEP_full_name)
        network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=ext_is_stance_AEP_full_name)
        network.add_connection(dep_k_1, source=not_exted_full_name, destination=flx_is_stance_AEP_full_name)
        network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=flx_is_stance_AEP_full_name)

        network.add_connection(dep_k_1, source=not_exted_full_name, destination=ext_is_stance_PEP_full_name)
        network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=ext_is_stance_PEP_full_name)
        network.add_connection(dep_k_1, source=not_flxed_full_name, destination=flx_is_stance_PEP_full_name)
        network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=flx_is_stance_PEP_full_name)

    else:
        #  if this is the CTr joint
        placeholder = True

    return generic_names


def add_neuron_input_output(network, neuron_type, neuron_name, joint_name):
    """Instantiates neuron, adds it to the network, adds inputs and outputs, give a formulaic name.

    :param network: SNS-Toolbox network dictionary to append to
    :param neuron_type: SNS-Toolbox neuron object
    :param neuron_name: "generic name" of the neuron, i.e., its function in the network
    :param joint_name: joint (or leg or body) to which this neuron belongs
    :return:
    """
    name_string = joint_name + "_" + neuron_name
    network.add_neuron(neuron_type, name=name_string)
    network.add_input(name_string, name=name_string + "_input")
    network.add_output(name_string, name=name_string + "_output")
    return name_string


def g_syn_from_k(k, rev_pot, volt_range):
    """Compute synaptic conductance given:

    :param k: synaptic gain (ratio of postsynaptic voltage to presynaptic voltage)
    :param rev_pot: reversal potential of the synapse
    :param volt_range: expected range of neuron voltage
    :return: (float) neuron conductance; may be None if computed conductance is negative.
    """
    #  rev_pot is delta_E, reversal potential of the synapse
    #  volt_range is R, expected range of voltage fluctuation
    g = k*volt_range/(rev_pot - k*volt_range)
    if g < 0:
        g = None
    return g


if __name__ == "__main__":
    main()
