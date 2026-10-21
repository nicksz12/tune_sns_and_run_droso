import mujoco
import mujoco.viewer
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import datetime
import strain_gauge_adapter as sga
import sys
import copy
import time
import os

from joint_adapter import JointAdapter
from pathlib import Path
from plot_data import plot_data

from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.renderer import render
from sns_toolbox.neurons import NonSpikingNeuronWithPersistentSodiumChannel


def main(mech_model_file_path='mjcf/Drosophibot_custom_actuators.xml', sns_control=False,
         joint_angle_file_path='data/IK_forward.json'):
    '''
    :return: dict: Neural states and mechanical states
    :return: path: Path of .json file storing neural states and mechanical states
    '''
    print("This is the sys: ")
    print(sys.executable)

    dt = datetime.datetime

    st_sw_ratio = 4
    sw_duration = 1.25
    servo_kp = 8.5
    num_loops = 12
    tmax = 2 * sw_duration * num_loops

    mx_28_no_load_speed_rpm = 55  # At 12V power supply
    mx_28_no_load_speed = mx_28_no_load_speed_rpm * 2 * np.pi / 60
    mx_64_no_load_speed_rpm = 63  # At 12V power supply
    mx_64_no_load_speed = mx_64_no_load_speed_rpm * 2 * np.pi / 60
    mx_28_max_torque = 1.5
    mx_64_max_torque = 4.5

    r_net = 20

    # Load robot model
    # model_droso = mujoco.MjModel.from_xml_path(filename=mech_model_file_path)
    #mjspec = mujoco.MjSpec.from_file(os.path.join(os.getcwd(), 'calculations/xml',
    #                                              "Drosophibot_custom_actuators.xml"))
    mjspec = mujoco.MjSpec.from_file(mech_model_file_path)

    model_droso = mjspec.compile()

    # Get robot state
    data_droso = mujoco.MjData(model_droso)

    # Load starting pose and control inputs
    mujoco.mj_resetData(model_droso, data_droso)

    # Set the physics step
    mj_dt = 0.001  # s
    model_droso.opt.timestep = mj_dt

    with open("droso_IK_20261020_171647.json", "r") as file:
        ik_data = json.load(file)

    #  Build joint_props and joint_adapters
    joint_props = {}
    jnt_adpt = {}
    joint_names = [model_droso.jnt(i).name for i in range(model_droso.njnt)]
    actuator_names = [model_droso.actuator(i).name for i in range(model_droso.nu)]
    qpos_inds = np.empty(model_droso.nu, dtype=np.int32)
    ctrl_inds = np.empty(model_droso.nu, dtype=np.int32)
    k_pos = np.zeros([model_droso.nu, model_droso.nu])

    for i in range(model_droso.nu):
        act_ind = mujoco.mj_name2id(model_droso, mujoco.mjtObj.mjOBJ_ACTUATOR, name=actuator_names[i])
        ctrl_inds[i] = np.int32(act_ind)
        jnt_ind = model_droso.actuator(act_ind).trnid[0]
        qpos_inds[i] = np.int32(model_droso.jnt_qposadr[jnt_ind])

        leg_key = actuator_names[i][:2]
        jt_key = actuator_names[i][3:]
        if leg_key not in joint_props:
            joint_props[leg_key] = {}
            jnt_adpt[leg_key] = {}
        joint_props[leg_key][jt_key] = {}
        joint_props[leg_key][jt_key]["ctrl_index"] = ctrl_inds[i]
        joint_props[leg_key][jt_key]["qpos_index"] = qpos_inds[i]
        joint_props[leg_key][jt_key]["ext_pep_rest_volt"] = -5
        joint_props[leg_key][jt_key]["flx_pep_rest_volt"] = -5
        joint_props[leg_key][jt_key]["motor_neuron_indices"] = None
        joint_props[leg_key][jt_key]["sensory_neuron_indices"] = None

        model_droso.actuator_ctrllimited[ctrl_inds[i]] = True
        model_droso.actuator_forcelimited[ctrl_inds[i]] = True
        if jt_key.lower() == "ctr":
            joint_props[leg_key][jt_key]["is_ctr"] = True
            model_droso.actuator_ctrlrange[ctrl_inds[i], :] = [-mx_64_no_load_speed, mx_64_no_load_speed]
            model_droso.actuator_forcerange[ctrl_inds[i], :] = [-mx_64_max_torque, mx_64_max_torque]
        else:
            joint_props[leg_key][jt_key]["is_ctr"] = False
            model_droso.actuator_ctrlrange[ctrl_inds[i], :] = [-mx_28_no_load_speed, mx_28_no_load_speed]
            model_droso.actuator_forcerange[ctrl_inds[i], :] = [-mx_28_max_torque, mx_28_max_torque]

        jnt_adpt[leg_key][jt_key] = JointAdapter(name=joint_names[i], joint_props=joint_props[leg_key][jt_key],
                                                 servo_max_speed=mx_28_no_load_speed, servo_kp=servo_kp)
        k_pos[i, i] = servo_kp


    model_droso.qpos0[qpos_inds] = ik_data['qpos0']

    num_legs = len(joint_props)

    strain_gauge_props = {}
    strain_gauge_data = {}
    CS_group_list = ["GP1_CS", "GP2_CS", "GP3_CS", "GP4_CS", "GP6A_CS", "GP6B_CS"]
    for leg_key in joint_props.keys():
        if leg_key not in strain_gauge_props:
            strain_gauge_props[leg_key] = {}
            strain_gauge_data[leg_key] = {}

        troch_index = mujoco.mj_name2id(model_droso, mujoco.mjtObj.mjOBJ_SITE, name=leg_key + "_Troch_CS")
        strain_gauge_props[leg_key]["GP1_CS"] = {"torque_sensor_index": 3 * troch_index + 1,
                                                 "sensory_neuron_index": None, "a": 1, "b": 0,
                                                 "min_strain": -1000, "current_at_min_strain": 0,
                                                 "max_strain": 1000, "current_at_max_strain": 0}
        strain_gauge_props[leg_key]["GP2_CS"] = {"torque_sensor_index": 3 * troch_index + 1,
                                                 "sensory_neuron_index": None, "a": -1, "b": 0,
                                                 "min_strain": -1000, "current_at_min_strain": 0,
                                                 "max_strain": 1000, "current_at_max_strain": 0}
        strain_gauge_props[leg_key]["GP3_CS"] = {"torque_sensor_index": 3 * troch_index + 2,
                                                 "sensory_neuron_index": None, "a": -10, "b": 0,
                                                 "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                                 "current_at_max_strain": 0}
        strain_gauge_props[leg_key]["GP4_CS"] = {"torque_sensor_index": 3 * troch_index + 2,
                                                 "sensory_neuron_index": None, "a": 10, "b": 0,
                                                 "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                                 "current_at_max_strain": 0}

        tib_index = mujoco.mj_name2id(model_droso, mujoco.mjtObj.mjOBJ_SITE, name=leg_key + "_Tibia_CS")
        strain_gauge_props[leg_key]["GP6A_CS"] = {"torque_sensor_index": 3 * tib_index + 2,
                                                  "sensory_neuron_index": None, "a": 1, "b": 0,
                                                  "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                                  "current_at_max_strain": 0}
        strain_gauge_props[leg_key]["GP6B_CS"] = {"torque_sensor_index": 3 * tib_index + 2,
                                                  "sensory_neuron_index": None, "a": -1, "b": 0,
                                                  "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                                  "current_at_max_strain": 0}

        strain_gauge_data[leg_key]["GP1_CS"] = {"reading": 0}
        strain_gauge_data[leg_key]["GP2_CS"] = {"reading": 0}
        strain_gauge_data[leg_key]["GP3_CS"] = {"reading": 0}
        strain_gauge_data[leg_key]["GP4_CS"] = {"reading": 0}
        strain_gauge_data[leg_key]["GP6A_CS"] = {"reading": 0}
        strain_gauge_data[leg_key]["GP6B_CS"] = {"reading": 0}



    if sns_control:
        # Configure how SNS and physics steps will interact
        sns_dt = 0.1  # ms
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
            update_ratio = 0
            raise "sns_dt must be less than mj_dt when measured in seconds"
        else:
            #  Timesteps are equal and the situation is straightforward
            sns_shorter = None
            min_dt = sns_dt_in_seconds
            update_ratio = 1

        tvec = np.arange(start=0, step=min_dt, stop=tmax)
        sns_tvec = np.arange(start=0, step=sns_dt_in_seconds, stop=tmax)
        mj_tvec = np.arange(start=0, step=mj_dt, stop=tmax)
        num_steps = len(tvec)
        num_sns_steps = len(sns_tvec)
        num_mj_steps = len(mj_tvec)

        #  By default, a capsule geom is extended along the z axis. This means that trochanter2, which is extended
        #  laterally, has (anterior-posterior x, vertical y, and) lateral z axis.
        #  LF_Trochanter2's geom has euler 0 90 0, so rotating 90 degrees about the y axis means the x-axis is lateral, the
        #  y-axis is vertical, and the z-axis is anterior-posterior. Thus, Gp1+2 are encoded in y-axis torque (2nd element,
        #  index 1) and Gp3+4 are encoded in z-axis torque (3rd element, index 2).
        """
        strain_gauge_props = {}
        strain_gauge_props["LF"] = {}
        strain_gauge_data = {}
        strain_gauge_data["LF"] = {}

        strain_gauge_props["LF"]["GP1_CS"] = {"torque_sensor_index": 1, "sensory_neuron_index": None, "a": 1, "b": 0,
                                              "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                              "current_at_max_strain": 0}
        strain_gauge_props["LF"]["GP2_CS"] = {"torque_sensor_index": 1, "sensory_neuron_index": None, "a": -1, "b": 0,
                                              "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                              "current_at_max_strain": 0}
        strain_gauge_props["LF"]["GP3_CS"] = {"torque_sensor_index": 2, "sensory_neuron_index": None, "a": -10, "b": 0,
                                              "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                              "current_at_max_strain": 0}
        strain_gauge_props["LF"]["GP4_CS"] = {"torque_sensor_index": 2, "sensory_neuron_index": None, "a": 10, "b": 0,
                                              "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                              "current_at_max_strain": 0}
        strain_gauge_props["LF"]["GP6A_CS"] = {"torque_sensor_index": 5, "sensory_neuron_index": None, "a": 1, "b": 0,
                                               "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                               "current_at_max_strain": 0}
        strain_gauge_props["LF"]["GP6B_CS"] = {"torque_sensor_index": 5, "sensory_neuron_index": None, "a": -1, "b": 0,
                                               "min_strain": -1000, "current_at_min_strain": 0, "max_strain": 1000,
                                               "current_at_max_strain": 0}

        strain_gauge_data["LF"]["GP1_CS"] = {"reading": 0}
        strain_gauge_data["LF"]["GP2_CS"] = {"reading": 0}
        strain_gauge_data["LF"]["GP3_CS"] = {"reading": 0}
        strain_gauge_data["LF"]["GP4_CS"] = {"reading": 0}
        strain_gauge_data["LF"]["GP6A_CS"] = {"reading": 0}
        strain_gauge_data["LF"]["GP6B_CS"] = {"reading": 0}
        """
        """ INSTANTIATE AND TUNE THE CONTROL NETWORK """
        g_ion_new = sp.optimize.bisect(calc_rg_equilibrium, 0, 5)
        print("Ca conductance is " + str(g_ion_new))
        print("Equilibrium voltage should be " + str(r_net + calc_rg_equilibrium(g_ion_new, to_plot=True)))

        net = Network(name="droso1leg")

        all_names = {}
        body_name = "droso"
        #  Critically, k_step_len and k_rotate will be set based on inverse kinematics. This is the purpose of Ava's work. I
        #  will integrate this in the future.
        generic_names = build_body_control_net(net, r_net=r_net, body_name=body_name)
        all_names[body_name] = generic_names

        max_joint_index = 0
        max_actuator_index = 0
        num_strain_gauges = 0
        #  For each joint (may include multiple legs)
        for leg_key, leg_value in joint_props.items():
            for jt_key, jt_value in joint_props[leg_key].items():
                key = leg_key + "_" + jt_key

                generic_names = build_joint_control_net(net, r_net=r_net, is_ctr=joint_props[leg_key][jt_key]["is_ctr"],
                                                        joint_name=key,
                                                        ext_pep_rest=joint_props[leg_key][jt_key]["ext_pep_rest_volt"],
                                                        flx_pep_rest=joint_props[leg_key][jt_key]["flx_pep_rest_volt"],
                                                        gmax_ion=g_ion_new, tau_slow=1400, st_sw_ratio=st_sw_ratio)

                all_names[key] = generic_names

                joint_props[leg_key][jt_key]["motor_neuron_indices"] = [all_names[key]["Ext_MN"], all_names[key]["Flx_MN"]]
                joint_props[leg_key][jt_key]["sensory_neuron_indices"] = [all_names[key]["Ext_Pos_SN"],
                                                                          all_names[key]["Flx_Pos_SN"],
                                                                          all_names[key]["Ext_Vel_SN"],
                                                                          all_names[key]["Flx_Vel_SN"]]

                ext_in_stance = False
                if key[1:] == "F_ThC3" or key[1:] == "F_CTr":
                    ext_in_stance = True
                    print(str(key) + " extends in stance phase!")

                generic_names = build_descending_pathways(net, r_net=r_net, k_step_len=0.9, k_rotate=0, joint_name=key,
                                                          ext_in_stance=ext_in_stance, body_name=body_name)

                all_names[key].update(generic_names)  # generic_names is currently empty, so it just goofs up all_names.
                # replace ctrl_index, etc. with joint_props dict

                jnt_adpt[leg_key][jt_key] = JointAdapter(name=key, joint_props=jt_value,
                                                    resting_posture_angle=0, min_angle=-0.5, max_angle=0.5,
                                                    swing_duration=sw_duration, max_stance_duration=st_sw_ratio*sw_duration,
                                                    servo_max_speed=mx_28_no_load_speed, servo_kp=servo_kp)
                jnt_adpt[leg_key][jt_key].resting_posture_angle = 0
                jnt_adpt[leg_key][jt_key].min_angle = -0.5
                jnt_adpt[leg_key][jt_key].max_angle = 0.5
                jnt_adpt[leg_key][jt_key].swing_duration=sw_duration
                jnt_adpt[leg_key][jt_key].max_stance_duration = st_sw_ratio*sw_duration

                max_joint_index = max(max_joint_index, joint_props[leg_key][jt_key]["qpos_index"])
                max_actuator_index = max(max_actuator_index, joint_props[leg_key][jt_key]["ctrl_index"])

            #  Still leg loop
            generic_names = build_leg_control_net(net, r_net=r_net, leg_name=leg_key, joint_props=joint_props)
            all_names[leg_key] = generic_names

            #  Loop through CS  # LF_GP1_CS_SN
            for cs_key, cs_value in strain_gauge_props[leg_key].items():
                key = cs_key + "_SN"
                strain_gauge_props[leg_key][cs_key]["sensory_neuron_index"] = [all_names[leg_key][key]]
                strain_gauge_data[leg_key][cs_key]["reading"] = 0
                num_strain_gauges = num_strain_gauges + 1

        #  Instantiate the SNS model and its inputs and outputs
        sns_model = net.compile(dt=sns_dt, backend='numpy', device='cpu', debug=False)
        sns_inputs = np.zeros([num_sns_steps, net.get_num_inputs()])
        #  To step with a constant step length:
        sns_inputs[np.arange(0, np.floor(4 * num_sns_steps / 5), dtype=int), net.get_input_index(
            'droso_step_length_input')] = 0

        #  To step forward then reverse direction:
        #  sns_inputs[:, net.get_input_index('droso_step_length_input')] = -10 * sp.special.erf((sns_tvec - 2 * tmax / 3))

        #  Give each joint's extensor RG neuron an excitatory impulse of current upon startup to break unstable equilibrium.
        for leg_key, leg_value in joint_props.items():
            for jt_key, jt_value in joint_props[leg_key].items():
                rg = leg_key + "_" + jt_key + "_Ext_RG_input"
                stim = 1
                if jt_key == "LF_ThC1" or jt_key == "LF_TrF" or jt_key == "LF_FTi":
                    stim = -1

                sns_inputs[0, net.get_input_index(rg)] = stim
                # sns_inputs[np.arange(0, np.floor(num_sns_steps/5), dtype=int), net.get_input_index(rg)] = -10

        sns_outputs = np.zeros([num_sns_steps, net.get_num_outputs()])

        render(net)
    else:
        sns_shorter = None
        update_ratio = 1
        
        tvec = np.arange(start=0, step=mj_dt, stop=tmax)
        mj_tvec = np.arange(start=0, step=mj_dt, stop=tmax)
        num_steps = len(tvec)
        num_mj_steps = len(mj_tvec)

        max_joint_index = 0
        max_actuator_index = 0
        num_strain_gauges = 0
        #  For each joint (may include multiple legs)
        for leg_key, leg_value in joint_props.items():
            for jt_key, jt_value in joint_props[leg_key].items():
                key = leg_key + "_" + jt_key
                jnt_adpt[leg_key][jt_key] = JointAdapter(name=key, joint_props=jt_value,
                                                    resting_posture_angle=0, min_angle=-0.5, max_angle=0.5,
                                                    swing_duration=sw_duration,
                                                    max_stance_duration=st_sw_ratio * sw_duration,
                                                    servo_max_speed=mx_28_no_load_speed, servo_kp=servo_kp)
                max_joint_index = max(max_joint_index, joint_props[leg_key][jt_key]["qpos_index"])
                max_actuator_index = max(max_actuator_index, joint_props[leg_key][jt_key]["ctrl_index"])

            #  Loop through CS  # LF_GP1_CS_SN
            for cs_key, cs_value in strain_gauge_props[leg_key].items():
                strain_gauge_props[leg_key][cs_key]["sensory_neuron_index"] = None
                strain_gauge_data[leg_key][cs_key]["reading"] = 0
                num_strain_gauges = num_strain_gauges + 1

    joint_pos = np.zeros([num_mj_steps, max_joint_index+1])
    joint_pos_command = np.zeros([num_mj_steps, max_actuator_index+1])
    joint_vel = np.zeros([num_mj_steps, max_joint_index])
    joint_vel_command = np.zeros([num_mj_steps, max_actuator_index+1])
    joint_torque = np.zeros([num_mj_steps, max_actuator_index+1])
    segment_moment = np.zeros([num_mj_steps, num_strain_gauges])

    i = int(0)
    i_mj = int(0)
    t = 0

    # Initialize viewer
    viewer = mujoco.viewer.launch_passive(model_droso, data_droso, key_callback=key_callback)
    step_mj(viewer, model_droso, data_droso)

    if sns_control:
        # Main loop
        while i < num_steps:

            if np.mod(i, update_ratio) == 0:
                #  Inform the joint adapter what the MN voltages are

                for leg_key in joint_props.keys():
                    for jt_key in joint_props[leg_key].keys():
                        #  jnt_adpt[key].mn_to_servo(v_all=sns_outputs[i-1, jnt_adpt[key].mn_indices])
                        jnt_adpt[leg_key][jt_key].mn_to_servo(v_all=sns_outputs[i - 1, :])

                        #  Read current joint state and calculated commanded values for the servo
                        [current_pos, current_vel, desired_pos, desired_vel] = (
                            jnt_adpt[leg_key][jt_key].set_servo_commands(data_droso))

                        #  Write the servo commands to the MuJoCo model. This could be modified/replaced to communicate with a robot.
                        jnt_adpt[leg_key][jt_key].write_servo_commands(data_droso)

                        #  Save values for logging.
                        #  Gotta expand size of joint_pos and joint_vel to account for all the actuated joints.
                        joint_pos[i_mj, jnt_adpt[leg_key][jt_key].qpos_index] = current_pos
                        joint_pos_command[i_mj, jnt_adpt[leg_key][jt_key].ctrl_index] = desired_pos
                        joint_vel[i_mj, jnt_adpt[leg_key][jt_key].qpos_index] = current_vel
                        joint_vel_command[i_mj, jnt_adpt[leg_key][jt_key].ctrl_index] = desired_vel
                        joint_torque[i_mj, jnt_adpt[leg_key][jt_key].ctrl_index] = (
                            data_droso.actuator_force)[jnt_adpt[leg_key][jt_key].ctrl_index]

                for leg_key in strain_gauge_props.keys():
                    for sg_key in strain_gauge_props[leg_key].keys():
                        segment_moment[i_mj, strain_gauge_props[leg_key][sg_key]["torque_sensor_index"]] = (
                            sga.get_reading_from_mj(strain_gauge_props[leg_key][sg_key], strain_gauge_data[leg_key][sg_key],
                                                    data_droso))

                #  Step the MuJuCo model forward. This could be modified/replaced to communicate with a robot.
                #  np.mod(i, 1000) == 0 sets whether to update the video or not.
                # step_mj(viewer, drosophibot, data_droso, to_sync=False)
                step_mj(viewer, model_droso, data_droso, to_sync=np.mod(i, 1000) == 0)

                i_mj += 1

            #  This writes the stored mechanical states to neural currents. Even if mechanics/robot did not update in this
            #  loop, this method acts like "memory" to give the neural system useful information, so we need it in every
            #  loop.

            for leg_key in joint_props.keys():
                for jt_key in joint_props[leg_key].keys():
                    jnt_adpt[leg_key][jt_key].write_state_to_sensory(sns_inputs[i, :])

            for leg_key in strain_gauge_props.keys():
                for sg_key in strain_gauge_props[leg_key].keys():
                    sga.write_state_to_sensory(strain_gauge_props[leg_key][sg_key], strain_gauge_data[leg_key][sg_key],
                                               sns_input=sns_inputs[i, :])

            sns_outputs[i, :] = sns_model(sns_inputs[i, :])
            t += min_dt
            i = i + 1
    else:
        #  not sns control, just running joint angle commands through
        #  load up a file with a struct of the IK solutions in it.

        # Walk straight ahead
        trans_ind = 6
        rot_ind = 5

        # Pivot around the hind legs
        #trans_ind = 3
        #rot_ind = 10

        #  then, for each row in the IK solution, find its actuator index, interpolate the data to this script's
        #  timestep, then write that to our 2D
        for jnt_name in ik_data["joint_names"]:
            ctrl_ind = mujoco.mj_name2id(model_droso, mujoco.mjtObj.mjOBJ_ACTUATOR, name=jnt_name)
            if ctrl_ind != -1:
                #  This actuator was found in the list
                f = sp.interpolate.PchipInterpolator(ik_data["time"], ik_data["joint_angles"][trans_ind][rot_ind][ctrl_ind])
                period = ik_data["time"][-1] + (ik_data["time"][1] - ik_data["time"][0])
                joint_pos_command[:, ctrl_ind] = f(mj_tvec % period)

        while i_mj < num_steps:

            data_droso.ctrl[ctrl_inds] = k_pos @ (joint_pos_command[i_mj, :].T - data_droso.qpos[qpos_inds])
            # data_droso.ctrl[ctrl_inds] = k_pos @ (0*joint_pos_command[i_mj, :].T - data_droso.qpos[qpos_inds])
            print(data_droso.qpos[qpos_inds])
            joint_pos[i_mj, :] = copy.deepcopy(data_droso.qpos)
            joint_vel[i_mj, :] = copy.deepcopy(data_droso.qvel)
            joint_vel_command[i_mj, ctrl_inds] = copy.deepcopy(data_droso.ctrl[ctrl_inds])
            joint_torque[i_mj, ctrl_inds] = copy.deepcopy(data_droso.actuator_force[ctrl_inds])
            """
            for leg_key in strain_gauge_props.keys():
                for sg_key in strain_gauge_props[leg_key].keys():
                    segment_moment[i_mj, strain_gauge_props[leg_key][sg_key]["torque_sensor_index"]] = (
                        sga.get_reading_from_mj(strain_gauge_props[leg_key][sg_key],
                                                strain_gauge_data[leg_key][sg_key],
                                                data_droso))
            """
            #  Step the MuJuCo model forward. This could be modified/replaced to communicate with a robot.
            #  np.mod(i, 1000) == 0 sets whether to update the video or not.
            # step_mj(viewer, model_droso, data_droso, to_sync=False)
            step_mj(viewer, model_droso, data_droso, to_sync=np.mod(i_mj, 1) == 0)

            i_mj += 1

            if np.mod(i_mj, 10) == 0:
                print(i_mj)
                1+1

    #  Pack up data
    data_pack = {}
    #  store time
    if sns_shorter:
        data_pack["time"] = mj_tvec.tolist()
    else:
        data_pack["time"] = tvec.tolist()
    #  store joint states
    for leg_key in joint_props.keys():
        for jt_key in joint_props[leg_key].keys():
            data_pack[str(leg_key) + "_" + str(jt_key) + "_pos"] = (
                joint_pos[:, joint_props[leg_key][jt_key]["qpos_index"]].tolist())
            data_pack[str(leg_key) + "_" + str(jt_key) + "_pos_comm"] = (
                joint_pos_command[:, joint_props[leg_key][jt_key]["ctrl_index"]].tolist())
            data_pack[str(leg_key) + "_" + str(jt_key) + "_vel"] = (
                joint_vel[:, joint_props[leg_key][jt_key]["qpos_index"]].tolist())
            data_pack[str(leg_key) + "_" + str(jt_key) + "_vel_comm"] = (
                joint_vel_command[:, joint_props[leg_key][jt_key]["ctrl_index"]].tolist())
            data_pack[str(leg_key) + "_" + str(jt_key) + "_torque"] = (
                joint_torque[:, joint_props[leg_key][jt_key]["ctrl_index"]].tolist())
    #  store torque/CS states
    for leg_key in strain_gauge_props.keys():
        for sg_key in strain_gauge_props[leg_key].keys():
            data_pack[str(leg_key) + "_" + str(sg_key) + "_torque"] = (
                segment_moment[:, strain_gauge_props[leg_key][sg_key]["torque_sensor_index"]].tolist())
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
        if sns_control:
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
        print(f"File size: {new_file_size*1e-6} MBs.")
        #  Kill time, wait for it to be published.

    plot_data(data_file_path)

    return data_pack, data_file_path


def key_callback(keycode):
    if chr(keycode) == ' ':
        max_speed = float(input('Enter floor z coordinate'))

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
        v_new = model()[0]
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
        return v_new - r


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


def build_leg_control_net(network, r_net, leg_name, joint_props):
    """

    :param network: SNS-Toolbox network dictionary to append to
    :param r_net: max expected neuron voltage + synaptic conductance saturation value
    :param leg_name: string label for neurons within the "leg" level network
    :return: (dict) neuron names and their index within the network
    """
    #  These are neurons that all joints in the leg can receive signals from (e.g., walking state) and send signals to
    #  (e.g., leg is moving).
    fast_neuron = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=5)
    slow_neuron = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=100)
    slow_neuron_tonic = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=100, resting_potential=r_net)
    slow_neuron_neg_bias = NonSpikingNeuron(membrane_conductance=1, membrane_capacitance=100, resting_potential=-r_net)

    generic_names = {}

    #  Add load sensory neurons, which belong to the leg, not any particular joint.
    cs_sens_neur_gp1 = "GP1_CS_SN"
    generic_names[cs_sens_neur_gp1] = network.get_num_neurons()
    cs_sens_neur_gp1_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp1, leg_name)
    #  Full name is required to create synapses.

    cs_sens_neur_gp2 = "GP2_CS_SN"
    generic_names[cs_sens_neur_gp2] = network.get_num_neurons()
    cs_sens_neur_gp2_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp2, leg_name)

    cs_sens_neur_gp3 = "GP3_CS_SN"
    generic_names[cs_sens_neur_gp3] = network.get_num_neurons()
    cs_sens_neur_gp3_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp3, leg_name)

    cs_sens_neur_gp4 = "GP4_CS_SN"
    generic_names[cs_sens_neur_gp4] = network.get_num_neurons()
    cs_sens_neur_gp4_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp4, leg_name)

    cs_sens_neur_gp6a = "GP6A_CS_SN"
    generic_names[cs_sens_neur_gp6a] = network.get_num_neurons()
    cs_sens_neur_gp6a_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp6a, leg_name)

    cs_sens_neur_gp6b = "GP6B_CS_SN"
    generic_names[cs_sens_neur_gp6b] = network.get_num_neurons()
    cs_sens_neur_gp6b_full_name = add_neuron_input_output(network, fast_neuron, cs_sens_neur_gp6b, leg_name)

    #  "AEP reached" neurons. These trigger leg depression.
    aep_reached = "aep_reached"
    generic_names[aep_reached] = network.get_num_neurons()
    #  aep_reached_full_name = add_neuron_input_output(network, slow_neuron_tonic, aep_reached, leg_name)
    aep_reached_full_name = add_neuron_input_output(network, slow_neuron, aep_reached, leg_name)

    #  "PEP reached" neurons. These trigger leg elevation.
    pep_reached = "pep_reached"
    generic_names[pep_reached] = network.get_num_neurons()
    pep_reached_full_name = add_neuron_input_output(network, slow_neuron, pep_reached, leg_name)

    e_ex = 220
    g_k_1 = g_syn_from_k(k=1, rev_pot=e_ex, volt_range=r_net)
    dep_k_1 = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_k_1, reversal_potential=e_ex)

    e_gain = -10
    g_gain = 2
    hyp_gain_binary = NonSpikingSynapse(e_lo=0, e_hi=0.1, max_conductance=g_gain, reversal_potential=e_gain)

    #  To start, Gp3 is the primary indicator of stance phase. This neuron excites pep_reached.
    network.add_connection(dep_k_1, source=cs_sens_neur_gp3_full_name, destination=pep_reached_full_name)

    #  Generate names of neurons we need to synapse onto PEP Reached (and PEP Reached needs to synapse onto).
    #  Loop through each joint, check if ctr. If so, PEP Reached inhibits leg depression.
    #  If not ctr, synapse that joint's sensory data onto PEP neurons:
    num_not_ctr_joints = 0
    num_ctr_joints = 0
    for jt_key, jt_value in joint_props[leg_name].items():
        if jt_value["is_ctr"]:
            ctr_ext_str = leg_name + "_" + str(jt_key) + "_" + "Ext_RG"
            hyp_ctr_ext = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=-100, max_conductance=400e-3)
            network.add_connection(hyp_ctr_ext, source=pep_reached_full_name, destination=ctr_ext_str)
            num_ctr_joints = num_ctr_joints + 1
        else:
            ext_src_str = leg_name + "_" + str(jt_key) + "_" + "Ext_is_Stance_PEP"
            flx_src_str = leg_name + "_" + str(jt_key) + "_" + "Flx_is_Stance_PEP"

            network.add_connection(hyp_gain_binary, source=ext_src_str, destination=pep_reached_full_name)
            network.add_connection(hyp_gain_binary, source=flx_src_str, destination=pep_reached_full_name)
            num_not_ctr_joints = num_not_ctr_joints + 1

    #  Inhibitory IN from load to AEP Reached
    load_inh_IN = "inh_IN"
    generic_names[load_inh_IN] = network.get_num_neurons()
    load_inh_IN_full_name = add_neuron_input_output(network, fast_neuron, load_inh_IN, leg_name)

    network.add_connection(dep_k_1, source=cs_sens_neur_gp3_full_name, destination=load_inh_IN_full_name)

    #  Inhibitory IN inhibits AEP Reached
    network.add_connection(hyp_gain_binary, source=load_inh_IN_full_name, destination=aep_reached_full_name)

    #  Generate names of neurons we need to synapse onto AEP Reached (and AEP Reached needs to synapse onto).
    #  Loop through each joint, check if ctr. If so, AEP Reached inhibits leg depression.
    #  If not ctr, synapse that joint's sensory data onto AEP neurons:
    for jt_key, jt_value in joint_props[leg_name].items():
        if jt_value["is_ctr"]:
            ctr_ext_str = leg_name + "_" + str(jt_key) + "_" + "Ext_RG"
            #-10, 400
            hyp_ctr_flx = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=-100, max_conductance=400e-3)

            network.add_connection(hyp_ctr_flx, source=pep_reached_full_name, destination=ctr_ext_str)

            ctr_flx_str = leg_name + "_" + str(jt_key) + "_" + "Flx_RG"
            hyp_ctr_ext = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=-100, max_conductance=400e-3)

            network.add_connection(hyp_ctr_ext, source=aep_reached_full_name, destination=ctr_flx_str)
        else:
            ext_src_str = leg_name + "_" + str(jt_key) + "_" + "Ext_is_Stance_AEP"
            flx_src_str = leg_name + "_" + str(jt_key) + "_" + "Flx_is_Stance_AEP"

            if jt_key == "FTi":
                network.add_connection(dep_k_1, source=ext_src_str, destination=aep_reached_full_name)
                network.add_connection(dep_k_1, source=flx_src_str, destination=aep_reached_full_name)

    #  Create a neuron that signals that the leg is "moving in stance": this reinforces stance phase in all joints' RGs.
    #  "Moving in stance" neurons. These reinforce stance phase pattern generator activity.
    moving_in_stance = "moving_in_stance"
    generic_names[moving_in_stance] = network.get_num_neurons()
    moving_in_stance_full_name = add_neuron_input_output(network, slow_neuron_neg_bias, moving_in_stance, leg_name)

    network.add_connection(dep_k_1, source=cs_sens_neur_gp4_full_name, destination=moving_in_stance_full_name)

    k_syn = 1/num_not_ctr_joints
    e_exc = 220
    g_exc = g_syn_from_k(k=k_syn, rev_pot=e_exc, volt_range=r_net)
    dep_vel_to_moving = NonSpikingSynapse(e_lo=0, e_hi=r_net, max_conductance=g_exc, reversal_potential=e_exc)
    for jt_key, jt_value in joint_props[leg_name].items():
        if jt_value["is_ctr"]:
            """
            ctr_flx_str = leg_name + "_" + str(jt_key) + "_" + "Flx_RG"
            #-10, 400
            hyp_ctr_flx = NonSpikingSynapse(e_lo=0, e_hi=r_net, reversal_potential=-100, max_conductance=400e-3)

            network.add_connection(hyp_ctr_flx, source=pep_reached_full_name, destination=ctr_flx_str)
            """
        else:
            #  Connect this joint's velocity feedback to moving_in_stance:
            ext_src_str = leg_name + "_" + str(jt_key) + "_" + "Exting_in_Stance_Dendrite"
            flx_src_str = leg_name + "_" + str(jt_key) + "_" + "Flxing_in_Stance_Dendrite"

            #  Both synapses have the same source, because joint-level logic will suppress the pathway that encodes
            #  swing phase motion.
            network.add_connection(dep_vel_to_moving, source=ext_src_str, destination=moving_in_stance_full_name)
            network.add_connection(dep_vel_to_moving, source=flx_src_str, destination=moving_in_stance_full_name)

            ext_dest_str = leg_name + "_" + str(jt_key) + "_" + "Ext_is_Stance_Load"
            flx_dest_str = leg_name + "_" + str(jt_key) + "_" + "Flx_is_Stance_Load"

            network.add_connection(dep_k_1, source=moving_in_stance_full_name, destination=ext_dest_str)
            network.add_connection(dep_k_1, source=moving_in_stance_full_name, destination=flx_dest_str)

    return generic_names


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

    exting_dend_full_name = joint_name + "_Ext_Vel_to_MN_Dendrite"
    flxing_dend_full_name = joint_name + "_Flx_Vel_to_MN_Dendrite"

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
    exting_dend = "Ext_Vel_to_MN_Dendrite"
    generic_names[exting_dend] = network.get_num_neurons()
    exting_dend_full_name = add_neuron_input_output(network, fast_neuron, exting_dend, joint_name)

    flxing_dend = "Flx_Vel_to_MN_Dendrite"
    generic_names[flxing_dend] = network.get_num_neurons()
    flxing_dend_full_name = add_neuron_input_output(network, fast_neuron, flxing_dend, joint_name)

    network.add_connection(dep_k_1, source=ext_vel_sn_full_name, destination=exting_dend_full_name)
    network.add_connection(dep_k_1, source=flx_vel_sn_full_name, destination=flxing_dend_full_name)

    network.add_connection(dep_k_1, source=exting_dend_full_name, destination=flx_mn_full_name)
    network.add_connection(dep_k_1, source=flxing_dend_full_name, destination=ext_mn_full_name)

    hyp_k_neg_0p4 = NonSpikingSynapse(e_lo=0, e_hi=20, max_conductance=80e-3, reversal_potential=-100)

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

        network.add_connection(hyp_k_neg_0p4, source=flx_stance_load_full_name, destination=ext_rg_full_name)
        network.add_connection(hyp_k_neg_0p4, source=ext_stance_load_full_name, destination=flx_rg_full_name)

        exting = "Exting_in_Stance_Dendrite"
        generic_names[exting] = network.get_num_neurons()
        exting_full_name = add_neuron_input_output(network, fast_neuron, exting, joint_name)

        flxing = "Flxing_in_Stance_Dendrite"
        generic_names[flxing] = network.get_num_neurons()
        flxing_full_name = add_neuron_input_output(network, fast_neuron, flxing, joint_name)

        network.add_connection(dep_k_1, source=ext_vel_sn_full_name, destination=exting_full_name)
        network.add_connection(dep_k_1, source=flx_vel_sn_full_name, destination=flxing_full_name)

        # State-dependent silencing of positive velocity FB to leg level "moving in stance" neuron.
        network.add_connection(hyp_gain, source=ext_is_stance_full_name, destination=flxing_full_name)
        network.add_connection(hyp_gain, source=flx_is_stance_full_name, destination=exting_full_name)

        # Positive feedback to CPG.
        # In reality, this depends on load feedback as well, so is integrated at the leg level.
        #  network.add_connection(dep_nsi_fb, source=exting_full_name, destination=ext_rg_full_name)
        #  network.add_connection(dep_nsi_fb, source=flxing_full_name, destination=flx_rg_full_name)

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
