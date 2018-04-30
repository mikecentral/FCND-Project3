"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = .5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self,
                z_k_p=4,
                z_k_d=2.7,
                xy_k_p=5,
                xy_k_d=3,
                
                k_p_roll=8,
                k_p_pitch=8,
                k_p_yaw=4.5,
                
                k_p_p = 20.0,
                k_p_q = 20.0,
                k_p_r = 5.0,
                
                max_tilt_roll = 1.0,
                max_tilt_pitch = 1.0,
                max_ascent_rate = 5.0,
                max_descent_rate=2.0,
                max_speed=5.0
                ):
        
        
        self.z_k_p = z_k_p
        self.z_k_d = z_k_d
        self.xy_k_p = xy_k_p
        self.xy_k_d = xy_k_d
        self.k_p_roll = k_p_roll
        self.k_p_pitch = k_p_pitch
        self.k_p_yaw = k_p_yaw
        self.k_p_p = k_p_p
        self.k_p_q = k_p_q
        self.k_p_r = k_p_r

        self.max_ascent_rate = max_ascent_rate
        self.max_descent_rate = max_descent_rate
        self.max_tilt_roll = max_tilt_roll
        self.max_tilt_pitch = max_tilt_pitch
        self.max_speed = max_speed


    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]
        
        
        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                
                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]
            
        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)
        
        
        return (position_cmd, velocity_cmd, yaw_cmd)
    
    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command
            
        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """

        speed_cmd = np.linalg.norm(local_velocity_cmd)  # calculate the speed being commanded

        if speed_cmd > self.max_speed:   # if the commanded speed is too high then reduce
            local_velocity_cmd = local_velocity_cmd * self.max_speed/speed_cmd

        pos_err = local_position_cmd - local_position
        vel_err = local_velocity_cmd - local_velocity

        p_term_xy = self.xy_k_p * pos_err
        d_term_xy = self.xy_k_d * vel_err

        accel_command = p_term_xy + d_term_xy + acceleration_ff

        return accel_command


    
    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)
            
        Returns: thrust command for the vehicle (+up)
        """
      
        z_err = altitude_cmd - altitude
        z_err_dot = vertical_velocity_cmd - vertical_velocity

        b_z = np.cos(attitude[0]) * np.cos(attitude[1]) # This is matrix element R33

        p_term = self.z_k_p * z_err
        d_term = self.z_k_d * z_err_dot + vertical_velocity_cmd  # added the second term for ff

        # total_velocity = p_term + vertical_velocity_cmd  # this is the new velocity after the thrust

        # limited_velocity = np.clip(total_velocity, -self.max_descent_rate, self.max_ascent_rate)  # need to limit vertical velocity by ascent/decent rates

        u_1 = p_term + d_term + acceleration_ff  # this is the desired vertical acceleration

        c = u_1 / b_z  # Note that you don't need to factor in gravity since the program sets the ff term to 9.81

        thrust = np.clip(c * DRONE_MASS_KG, 0.0, MAX_THRUST) # Limit thrust to values between 0 and Max Thrust

        return thrust
        
    
    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame
        
        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton
            
        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """

        R = euler2RM(attitude[0], attitude[1], attitude[2])

        R11 = R[0,0]
        R12 = R[0,1]
        R13 = R[0,2]
        R21 = R[1,0]
        R22 = R[1,1]
        R23 = R[1,2]
        R33 = R[2,2]

        # From lesson 14.16 we know that x_dot_dot = c * R13 and y_dot_dot = c * R23 where c is thrust_cmd/mass
        # R13 is -sin(pitch) and R23 is sin(roll)*cos(pitch)

        c = -thrust_cmd/DRONE_MASS_KG

        if thrust_cmd > 0.0:
            # limit the tilt angles using the max_tilt values
            R13_cmd = np.clip(acceleration_cmd[0]/c, -self.max_tilt_roll, self.max_tilt_roll)
            R23_cmd = np.clip(acceleration_cmd[1]/c, -self.max_tilt_pitch, self.max_tilt_pitch)

            b_x_dot = self.k_p_roll  * (R13_cmd - R13)
            b_y_dot = self.k_p_pitch * (R23_cmd - R23)

            p_cmd = (1/R33) * (R21 * b_x_dot - R11 * b_y_dot)

            q_cmd = (1/R33) * (R22 * b_x_dot - R12 * b_y_dot)

        else:  # If thrust is negative or = 0 then set pitch and roll rates to zero

            p_cmd = 0.0
            q_cmd = 0.0

        return np.array([p_cmd, q_cmd])
    
    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2
            
        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        k_p_rate = np.array([self.k_p_p, self.k_p_q, self.k_p_r])
        rate_err = body_rate_cmd - body_rate
        moments_cmd = MOI * np.multiply(k_p_rate, rate_err)

        # Limit the moments to the Max Torque value
        if np.linalg.norm(moments_cmd) > MAX_TORQUE:
            moments_cmd = moments_cmd*MAX_TORQUE/np.linalg.norm(moments_cmd)

        return moments_cmd
    
    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate
        
        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians
        
        Returns: target yawrate in radians/sec
        """

        # Since yaw is decoupled from the other directions, we only need a P controller

        yaw_cmd = np.mod(yaw_cmd, 2.0*np.pi)  # constrain yaw to the range (0,2pi)

        yaw_err = yaw_cmd - yaw

        # We have a choice on which way to rotate the drone to get to the desired yaw angle
        # And should pick the direction (CW or CCW) that requires the smaller rotation

        if yaw_err > np.pi:
            yaw_err = yaw_err - 2.0*np.pi
        elif yaw_err < -np.pi:
            yaw_err = yaw_err + 2.0*np.pi
        
        yaw_rate = self.k_p_yaw * yaw_err

        return yaw_rate
    
