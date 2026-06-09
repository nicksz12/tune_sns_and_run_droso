import numpy as np
import matplotlib.pyplot as plt
import time

class JointAdapter:
    def __init__(self, name, joint_props, mn_indices, sens_neur_indices, resting_posture_angle, min_angle, max_angle,
                 swing_duration, max_stance_duration, r=20, servo_max_speed=6.0, servo_kp=1.0):
        self.name = name
        self.ctrl_index = joint_props["ctrl_index"]
        self.qpos_index = joint_props["qpos_index"]
        self.is_ctr = joint_props["is_ctr"]
        self.mn_indices = mn_indices
        self.sens_neur_indices = sens_neur_indices
        self.resting_posture_angle = resting_posture_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_servo_speed = servo_max_speed  # rad/s
        self.servo_kp = servo_kp


        self.st_sw_ratio = max_stance_duration / swing_duration
        pos_rom = self.max_angle - self.resting_posture_angle
        neg_rom = self.resting_posture_angle - self.min_angle
        self.max_rom = max([pos_rom, neg_rom])
        speed_fac = 2
        self.max_comm_speed = speed_fac * self.max_rom / swing_duration
        # self.min_speed = max_speed / self.r_net
        self.max_speed = self.max_angle - self.min_angle  # Max required motion speed
        self.min_speed = self.max_comm_speed / 100

        self.comm_angle = None
        self.comm_speed = None
        self.current_angle = None
        self.current_velocity = None

        self.ext_angle_Iapp = None
        self.flx_angle_Iapp = None
        self.ext_velocity_Iapp = None
        self.flx_velocity_Iapp = None

        self.r_net = r

        '''Define neural->mechanical position adapter parameter values'''
        self.ext_comm_angle_C = self.st_sw_ratio*self.max_rom/self.r_net
        self.flx_comm_angle_C = -self.st_sw_ratio*self.max_rom/self.r_net

        self.ext_comm_angle_D = self.resting_posture_angle/2
        self.flx_comm_angle_D = self.resting_posture_angle/2

        self.ext_comm_angle_min_input = 0
        self.flx_comm_angle_min_input = 0
        self.ext_comm_angle_max_input = self.r_net
        self.flx_comm_angle_max_input = self.r_net

        self.ext_comm_angle_output_at_min_input = self.resting_posture_angle/2
        self.flx_comm_angle_output_at_min_input = self.resting_posture_angle/2
        self.ext_comm_angle_output_at_max_input = self.ext_comm_angle_D + self.ext_comm_angle_C * self.r_net
        self.flx_comm_angle_output_at_max_input = self.flx_comm_angle_D + self.flx_comm_angle_C * self.r_net

        '''Define neural->mechanical velocity adapter parameter values'''
        self.ext_comm_speed_C = self.max_comm_speed / self.r_net
        self.flx_comm_speed_C = self.max_comm_speed / self.r_net

        self.ext_comm_speed_D = -self.min_speed
        self.flx_comm_speed_D = -self.min_speed

        self.ext_comm_speed_min_input = self.min_speed / self.max_comm_speed * self.r_net
        self.flx_comm_speed_min_input = self.min_speed / self.max_comm_speed * self.r_net
        self.ext_comm_speed_max_input = 0.75 * self.r_net  # Above this value, we will get top speed, like fast MN.
        self.flx_comm_speed_max_input = 0.75 * self.r_net  # Above this value, we will get top speed, like fast MN.

        self.ext_comm_speed_output_at_min_input = -self.min_speed
        self.flx_comm_speed_output_at_min_input = -self.min_speed
        self.ext_comm_speed_output_at_max_input = np.min([self.max_comm_speed, self.max_servo_speed])
        self.flx_comm_speed_output_at_max_input = np.min([self.max_comm_speed, self.max_servo_speed])

        '''Define mechanical->neural position adapter parameter values'''
        self.ext_sens_pos_C = self.r_net/self.max_rom
        self.ext_sens_pos_D = -self.ext_sens_pos_C*self.resting_posture_angle
        self.flx_sens_pos_C = -self.r_net/self.max_rom
        self.flx_sens_pos_D = self.flx_sens_pos_C*self.resting_posture_angle
        self.ext_sens_pos_output_at_max_angle = self.r_net  # NOT SURE THIS IS RIGHT
        self.flx_sens_pos_output_at_max_angle = 0
        self.ext_sens_pos_output_at_min_angle = 0
        self.flx_sens_pos_output_at_min_angle = self.r_net  # NOT SURE THIS IS RIGHT
        '''
        self.ext_sens_vel_amp = self.r_net
        self.ext_sens_vel_steep = 2.5 * self.r_net  # originally, 25 *
        self.ext_sens_vel_xoff = 25 * self.min_speed  # originally, 2 *

        self.flx_sens_vel_amp = self.r_net
        self.flx_sens_vel_steep = -2.5 * self.r_net
        self.flx_sens_vel_xoff = -25 * self.min_speed
        '''
        self.ext_sens_vel_C = self.r_net / self.max_servo_speed
        self.ext_sens_vel_D = 0
        self.flx_sens_vel_C = -self.r_net / self.max_servo_speed
        self.flx_sens_vel_D = 0
        self.ext_sens_vel_output_at_max_speed = self.r_net
        self.flx_sens_vel_output_at_max_speed = 0
        self.ext_sens_vel_output_at_min_speed = 0
        self.flx_sens_vel_output_at_min_speed = self.r_net


    def verify_motor_maps(self):
        '''
        Method to plot, and thus visually verify, that MN voltages are correctly being converted to commanded positions
        for the joint controllers. Create v_mn to deliberately go outside the bounds of [0, R], in order to ensure
        saturations work properly.
        :return:
        '''
        v_mn = np.linspace(-self.r_net/4, 1.25*self.r_net, 20)
        [v_ext, v_flx] = np.meshgrid(v_mn, v_mn)
        comm_angle = np.zeros_like(v_ext)
        comm_speed = np.zeros_like(v_ext)
        for i in np.arange(np.size(v_ext)):
            self.mn_to_servo(v_mn_ext=v_ext.flat[i], v_mn_flx=v_flx.flat[i])
            comm_angle.flat[i] = self.comm_angle
            comm_speed.flat[i] = self.comm_speed

        # 3D surface plot of commanded angle vs. ext mn and flx mn
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(v_ext, v_flx, comm_angle, label='network output')
        ax.plot_surface(v_ext, v_flx, self.ext_comm_angle_C*v_ext + self.ext_comm_angle_D +
                        self.flx_comm_angle_C*v_flx + self.flx_comm_angle_D, label='mapping w/o saturation')
        plt.legend()
        plt.xlabel('V_ext')
        plt.ylabel('V_flx')
        plt.title(self.name + ' commanded angle')
        plt.draw()

        # Contour plot of commanded angle vs. ext mn and flx mn
        plt.figure()
        cs0 = plt.contour(v_ext, v_flx, comm_angle, levels=np.arange(start=np.min(comm_angle) + 1e-6,
                    stop=np.max(comm_angle) - 1e-6, step=(np.max(comm_angle) + 2e-6 - np.min(comm_angle))/10))
        plt.clabel(cs0, inline=1, fontsize=10)
        cs1 = plt.contour(v_ext, v_flx, comm_angle, levels=[self.min_angle, self.max_angle], colors=['red', 'black'])
        #self.resting_posture_angle - self.max_rom / 2, self.resting_posture_angle + self.max_rom / 2
        plt.clabel(cs1, inline=1, fontsize=10)
        plt.plot(self.r_net/self.st_sw_ratio + np.zeros([3]), np.array([0, self.r_net/self.st_sw_ratio, self.r_net]), 'k*--', label="ext st/sw ratio")
        plt.plot(np.array([0, self.r_net/self.st_sw_ratio, self.r_net]), self.r_net / self.st_sw_ratio + np.zeros([3]), 'r*--', label="flx st/sw ratio")

        plt.legend()
        plt.xlabel('V_ext')
        plt.ylabel('V_flx')
        plt.title(self.name + ' commanded angle')

        # 3D surface plot of commanded speed vs. ext mn and flx mn
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(v_ext, v_flx, comm_speed, label='network output')
        #ax.plot_surface(v_ext, v_flx, self.ext_comm_speed_C * v_ext + self.ext_comm_speed_D +
        #                self.flx_comm_speed_C * v_flx + self.flx_comm_speed_D, label='mapping w/o saturation')
        plt.legend()
        plt.xlabel('V_ext')
        plt.ylabel('V_flx')
        plt.title(self.name + ' commanded speed')

        # Contour plot of commanded speed vs. ext mn and flx mn
        plt.figure()
        cs0 = plt.contour(v_ext, v_flx, comm_speed, levels=np.arange(start=np.min(comm_speed) + 1e-6,
                                                                     stop=np.max(comm_speed) - 1e-6, step=(np.max(
                comm_speed) + 2e-6 - np.min(comm_speed)) / 10))
        plt.clabel(cs0, inline=1, fontsize=10)
        cs1 = plt.contour(v_ext, v_flx, comm_speed, levels=[self.min_speed, self.max_comm_speed], colors=['red', 'black'])
        # self.resting_posture_angle - self.max_rom / 2, self.resting_posture_angle + self.max_rom / 2
        plt.clabel(cs1, inline=1, fontsize=10)
        plt.plot(self.r_net / self.st_sw_ratio + np.zeros([3]),
                 np.array([0, self.r_net / self.st_sw_ratio, self.r_net]), 'k*--', label="ext st/sw ratio")
        plt.plot(np.array([0, self.r_net / self.st_sw_ratio, self.r_net]),
                 self.r_net / self.st_sw_ratio + np.zeros([3]), 'r*--', label="flx st/sw ratio")

        plt.legend()
        plt.xlabel('V_ext')
        plt.ylabel('V_flx')
        plt.title(self.name + ' commanded speed')

        # Generate data to test how sensory neurons encode posture and movement
        num_samps = 301
        servo_ang = np.arange(-1.25 * self.max_rom, 1.25 * self.max_rom, step=2.5 * self.max_rom/num_samps)
        servo_vel = np.arange(-1.25 * self.max_servo_speed, 1.25 * self.max_servo_speed, step=2.5 * self.max_servo_speed / num_samps)
        sens_pos = np.zeros([num_samps, 2], dtype=float)
        sens_vel = np.zeros([num_samps, 2], dtype=float)
        for i in range(len(servo_ang)):
            self.set_joint_state(servo_ang[i], servo_vel[i])
            ext_pos, flx_pos, ext_vel, flx_vel = self.write_state_to_sensory()
            sens_pos[i, :] = [ext_pos, flx_pos]
            sens_vel[i, :] = [ext_vel, flx_vel]

        # Plot position sensitive neuron voltage vs. joint angle
        plt.figure()
        plt.plot(servo_ang, sens_pos[:, 0], label='Ext Pos SN')
        plt.plot(servo_ang, sens_pos[:, 1], label='Flx Pos SN')
        plt.xlabel('Joint Angle')
        plt.ylabel('V_SN')
        plt.legend()

        plt.figure()
        plt.plot(servo_vel, sens_vel[:, 0], '+-', label='Ext Vel SN')
        plt.plot(servo_vel, sens_vel[:, 1], 'o-', label='Flx Vel SN')
        plt.xlabel('Joint Velocity')
        plt.ylabel('V_SN')
        plt.legend()

        ext_c_ratio = self.ext_comm_speed_C/self.ext_sens_vel_C
        flx_c_ratio = -self.flx_comm_speed_C/self.flx_sens_vel_C
        print('for ext, c1/c2 = ' + str(ext_c_ratio))
        print('for flx, c1/c2 = ' + str(flx_c_ratio))
        print('for ext, k_0 = ' + str((1 - ext_c_ratio) * self.max_servo_speed / self.r_net))
        print('for flx, k_0 = ' + str((1 - flx_c_ratio) * self.max_servo_speed / self.r_net))

        plt.show()

        return True

    def mn_to_servo(self, **kwargs):
        '''

        :param kwargs:  v_all: pass all network neuron voltages at this instant, pick out MN indices.
                        v_mn_ext, v_mn_flx: pass 2 scalars for extensor and flexor motor neuron voltage.
        :return: self.comm_angle, self.comm_speed
        '''

        v_mn_ext = None
        v_mn_flx = None
        if len(kwargs) == 1:
            try:
                v_mn_ext = kwargs["v_all"][self.mn_indices[0]]
                v_mn_flx = kwargs["v_all"][self.mn_indices[1]]
            except ValueError:
                print('If passing a single input "v_all", it must have more elements than this joint''s MN indices.')

        elif len(kwargs) == 2:
            try:
                v_mn_ext = kwargs["v_mn_ext"]
                v_mn_flx = kwargs["v_mn_flx"]
            except ValueError:
                print('If passing two inputs "v_mn_ext" and "v_mn_flx", each must be a scalar.')

        else:
            print('uh oh.')
            print(len(kwargs))
            raise 'input to joint_adapter.mn_to_servo() is the wrong shape.'

        comm_angle_ext = self.ext_comm_angle_C * v_mn_ext + self.ext_comm_angle_D
        if v_mn_ext > self.ext_comm_angle_max_input:
            comm_angle_ext = self.ext_comm_angle_output_at_max_input
        if v_mn_ext < self.ext_comm_angle_min_input:
            comm_angle_ext = self.ext_comm_angle_output_at_min_input

        comm_angle_flx = self.flx_comm_angle_C * v_mn_flx + self.flx_comm_angle_D
        if v_mn_flx > self.flx_comm_angle_max_input:
            comm_angle_flx = self.flx_comm_angle_output_at_max_input
        if v_mn_flx < self.flx_comm_angle_min_input:
            comm_angle_flx = self.flx_comm_angle_output_at_min_input

        self.comm_angle = comm_angle_ext + comm_angle_flx

        comm_speed_ext = self.ext_comm_speed_C * v_mn_ext + self.ext_comm_speed_D
        if v_mn_ext > self.ext_comm_speed_max_input:
            comm_speed_ext = self.ext_comm_speed_output_at_max_input
        if v_mn_ext < self.ext_comm_speed_min_input:
            comm_speed_ext = self.ext_comm_speed_output_at_min_input

        comm_speed_flx = self.flx_comm_speed_C * v_mn_flx + self.flx_comm_speed_D
        if v_mn_flx > self.flx_comm_speed_max_input:
            comm_speed_flx = self.flx_comm_speed_output_at_max_input
        if v_mn_flx < self.flx_comm_speed_min_input:
            comm_speed_flx = self.flx_comm_speed_output_at_min_input

        self.comm_speed = comm_speed_ext + comm_speed_flx

        return self.comm_angle, self.comm_speed

    def set_servo_commands(self, mj_struct, input_angle=None, input_speed=None):
        """
        :param mj_struct: MuJoCo data structure for writing control states and reading robot states
        :param input_angle: Commanded servo angle. If not defined, self.comm_angle is used.
        :param input_speed: Commanded servo speed. If not defined, self.comm_speed is used.
        :return: current_angle, current_velocity, comm_speed
        """

        if input_angle is None:
            input_angle = self.comm_angle

        self.current_angle = mj_struct.qpos[self.qpos_index]
        pos_error = input_angle - self.current_angle

        if input_speed is None:
            self.comm_speed = np.clip((self.servo_kp * pos_error), -self.max_comm_speed, self.max_comm_speed)
        else:
            self.comm_speed = np.clip(input_speed, -self.max_comm_speed, self.max_comm_speed)

        self.current_velocity = mj_struct.qvel[self.qpos_index]

        return self.current_angle, self.current_velocity, self.comm_angle, self.comm_speed

    def write_servo_commands(self, mj_struct):

        mj_struct.ctrl[self.ctrl_index] = self.comm_speed

        return True

    def set_joint_state(self, servo_pos=None, servo_vel=None):
        if servo_pos is not None:
            self.current_angle = servo_pos
        if servo_vel is not None:
            self.current_velocity = servo_vel
        return True

    def write_state_to_sensory(self, sns_input=None):
        if self.current_angle > self.resting_posture_angle:
            self.flx_angle_Iapp = 0
            if self.current_angle > self.max_angle:
                self.ext_angle_Iapp = self.ext_sens_pos_output_at_max_angle
            else:
                self.ext_angle_Iapp = self.ext_sens_pos_C * self.current_angle + self.ext_sens_pos_D
        else:
            self.ext_angle_Iapp = 0
            if self.current_angle < self.min_angle:
                self.flx_angle_Iapp = self.flx_sens_pos_output_at_min_angle
            else:
                self.flx_angle_Iapp = self.flx_sens_pos_C * self.current_angle + self.ext_sens_pos_D

        if self.current_velocity > 0:
            self.flx_velocity_Iapp = 0
            if self.current_velocity > self.max_servo_speed:
                self.ext_velocity_Iapp = self.ext_sens_vel_output_at_max_speed
            else:
                self.ext_velocity_Iapp = self.ext_sens_vel_C * self.current_velocity + self.ext_sens_vel_D
        else:
            self.ext_velocity_Iapp = 0
            if self.current_velocity < -self.max_servo_speed:
                self.flx_velocity_Iapp = self.flx_sens_vel_output_at_min_speed
            else:
                self.flx_velocity_Iapp = self.flx_sens_vel_C * self.current_velocity + self.ext_sens_vel_D
        '''
        self.ext_velocity_Iapp = self.ext_sens_vel_amp/(
                    1 + np.exp(self.ext_sens_vel_steep*(self.ext_sens_vel_xoff - self.current_velocity)))
        self.flx_velocity_Iapp = self.flx_sens_vel_amp / (
                    1 + np.exp(self.flx_sens_vel_steep * (self.flx_sens_vel_xoff - self.current_velocity)))
        '''
        if sns_input is not None:
            sns_input[self.sens_neur_indices] = [self.ext_angle_Iapp, self.flx_angle_Iapp, self.ext_velocity_Iapp,
                                                 self.flx_velocity_Iapp]

        return self.ext_angle_Iapp, self.flx_angle_Iapp, self.ext_velocity_Iapp, self.flx_velocity_Iapp

