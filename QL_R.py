#!/usr/bin/env python3
# license removed for brevity

import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import cv2
from cv_bridge import CvBridge
import glob
import sys
import rospy
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaCollisionEvent, CarlaEgoVehicleStatus,CarlaStatus, CarlaEgoVehicleInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import random
import time
import subprocess
import os
import signal
import copy
import pickle
import datetime
from itertools import groupby

#TEST NUMBER
TEST = "R_7"

#list of values for each loop iteration
K_vals = [[],[],[],[]] # Kv, Kl, Ks
K_tuple = []
e_vals = [[],[],[]]
x_vals = []
y_vals = []
e_b = np.array([[0.0],[0.0],[0.0]])

# GAINS
K_v = 1# Linear velocitty gain
K_l = 1# Linear gain
K_s = 1# Streering gain
K_i = 0.7
Kv_0 = 1
Kl_0 = 1
Ks_0 = 1
Ki_0 = 0.7
K_max = np.array([0,0,0,0])
K_min = np.array([0,0,0,0])
gain_changer = []

# CONTROL VARIABLES
refpath = []
ref_points = []
REFPATH_FILE = "refpath_round_new.txt"
v = 0
w_s = 0
phi = 0 
x = 0
y = 0
theta = 0
init_pos = [-45.0,0] #initial position
ahead = 25 # Nr. of positions ahead from closes position in refpath
err2target = float('inf')
vel_vals = []

#Control Loop time
start_time = 0
LOOP_TIME = 30
hop = 0.05
frame_start = float('inf')
frame = 0
delta = 0
ind = 0

# Simulation flags
collided = False
collision = None
control_end = False
moving = False

#Aux variables
ros_proc = None
f = open("test_{}_output.txt".format(TEST),'w')

#CARL-ROS VARIABLES
init_data = AckermannDrive()
init_data.speed = 0.5
odometry = Odometry()
odometry.header.seq = 100
velocity = CarlaEgoVehicleStatus()
velocity.header.seq = 100
pub = None

# Q-Learning Variables
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPISODES = 20
STEPS = 2
aggr_ep_rewards = {'ep': [], 'sum': [], 'last': []}
aggr_gain_time = {'gain': [], 't': [], 'avg_vel': [], 'avg_ey': [], 'avg_et': []}
state_list = []
WIN = 0.001

# Initial Q values (reward range)
LOW_Q_VALUE = -0.01
HIGH_Q_VALUE = 0.01

# Observation space range => [Eb_y, Eb_teta] value range
OBS_HIGH = np.array([1, 0.1])
OBS_LOW =  np.array([0 , 0])

#q_table_name = "qtable-test-24_2-39.pickle"
q_table_name = None
# Action space size : all possible combination of increase, decrease and mantain Kv, Kl, Ks and ki
action_space_size = 81

# Observation space size => 40 * [Eb_y, Eb_teta]
DISCRETE_OS_SIZE = [40]*2
discrete_os_unit_size = (OBS_HIGH - OBS_LOW)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 0
if EPISODES == 1:
    END_EPSILON_DECAYING = 1
else:
    END_EPSILON_DECAYING =EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

if q_table_name is None:
#RANDOMIZED Q-TABLE
    q_table = np.random.uniform(low=LOW_Q_VALUE, high=HIGH_Q_VALUE, size=(DISCRETE_OS_SIZE + [action_space_size]))
else:
    print("Using Q-table: ", q_table_name, file=f)
    with open(q_table_name, "rb") as f:
        q_table = pickle.load(f)

# Utility function
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# Discreticize state
def get_discrete_state(state):
    
    obs = copy.copy(state)

    for i in range(0, len(obs)):
        if obs[i] < OBS_LOW[i]:
            obs[i] = OBS_LOW[i]
        elif obs[i] > OBS_HIGH[i]:
            obs[i] = OBS_HIGH[i]

    discrete_state = (obs - OBS_LOW)/discrete_os_unit_size

    for i in range(0, len(discrete_state)):
        if discrete_state[i] >= DISCRETE_OS_SIZE[i]:
            discrete_state[i] = DISCRETE_OS_SIZE[i]-1
  
    return tuple(discrete_state.astype(int)) 

#Initiate some variables
def initiate():
    global pub
    get_refpath()
    rospy.init_node('ackermann', anonymous=True)
    pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size=1)
    gain_values()

# Find closest point from position pos
def closest_point(pos, refs):
    refs = np.asarray(refs)
    deltas = refs - pos
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

# Get reference path from file
def get_refpath():

    global refpath, ref_points

    file1 = open(REFPATH_FILE,"r+") 
    
    ref_x = [float(s) for s in file1.readline().split(', ')]
    ref_y = [float(s) for s in file1.readline().split(', ')]
    ref_theta = [float(s) for s in file1.readline().split(', ')]
    file1.close()
    refpath=np.array([ref_x,ref_y,ref_theta])
    for i in range(0,len(refpath[1])):
        ref_points.append([refpath[0][i], refpath[1][i], refpath[2][i]])

#Register values for plots
def update_gain_lists(clear=False):
    global K_vals, K_tuple
    if clear:
        K_vals.clear()
        K_vals = [[],[],[],[]]
    else:

        if K_v == 1.0:
            K_tuple.append(tuple([round(K_v),round(K_l),round(K_s), K_i]))
            K_vals[0].append(round(K_v))
            K_vals[1].append(round(K_l))
            K_vals[2].append(round(K_s))
            K_vals[3].append(K_i)

        else:
            K_tuple.append(tuple([K_v,round(K_l),round(K_s), K_i]))
            K_vals[0].append(K_v)
            K_vals[1].append(round(K_l))
            K_vals[2].append(round(K_s))
            K_vals[3].append(K_i)

# Define new gain limits to explore
def define_limits():
    
    global Kv_0, Kl_0, Ks_0, Ki_0, K_max, K_min

    if len(K_vals[0])>=8:

        if all_equal(K_vals[0][-5:]):
            K_max[0] = K_vals[0][-1]
            K_min[0] = K_vals[0][-1]
            Kv_0 = K_vals[0][-1]
            print("Changed limit for Kv to ", K_max[0], file=f)
        
        if all_equal(K_vals[1][-5:]):
            K_max[1] = K_vals[1][-1]
            K_min[1] = K_vals[1][-1]
            Kl_0 = K_vals[1][-1]
            print("Changed limit for Kl to ", K_max[1], file=f)
        
        if all_equal(K_vals[2][-5:]):
            K_max[2] = K_vals[2][-1]
            K_min[2] = K_vals[2][-1]
            Ks_0 = K_vals[2][-1]
            print("Changed limit for Ks to ", K_max[2], file=f)
        '''
        if all_equal(K_vals[3][-5:]):
            K_max[3] = K_vals[3][-1]
            K_min[3] = K_vals[3][-1]
            Ki_0 = K_vals[3][-1]
            print("Changed limit for Ki to ", K_max[3], file=f)
            '''
    

# ROS Subscriber for vehicle velocity
def check_velocity(data):
    global velocity
    velocity = data
 
#Calculate current error - error to closest refpath position
def get_current_error():

    ref_x = refpath[0,ind]
    ref_y = refpath[1,ind]
    ref_theta =refpath[2,ind] #rad
    
    #print("Ref: ", ref_x, ref_y, ref_theta, " pose: ", x,y,theta)
    # ---- ERRORS ----
    e_world = np.array([[ref_x-x], [ref_y-y], [adjust(ref_theta-theta)]])
    matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
    e_car = np.matmul(matrix,e_world)
    return [abs(e_car[1][0]),abs(e_car[2][0])] 
        
#Plots
def plotter():
    
    plt.cla()
    plt.scatter(np.array(range(0,EPISODES)),[str(i) for i in K_tuple], linestyle='None', marker = 'x')
    
    plt.title("K VALUES")
    #plt.canvas.start_event_loop(sys.float_info.min) #workaround for Exception in Tkinter callback
    plt.savefig('gains.png', bbox_inches = 'tight', pad_inches = 0.7)

    plt.cla()
    plt.title("Last rewards of each episode")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['last'])
    plt.show()  

    plt.cla()
    plt.title("Sum of rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['sum'])
    plt.show()

#Get error average
def get_avg_error():
    if len(e_vals[0]) != 0 and len(e_vals[1]) != 0 :
        return [np.mean(e_vals[0]), np.mean(e_vals[1])]
    else:
        return [0,0]

#Define gain changer matrix - all posible combinations of -1, 0 and 1
def gain_values():
    changers = np.array([-1, 0, 1]) # Either add, maintain or subtract constant to gain
    global gain_changer
    
    for i in range(0,3): 
        v1 = changers[i]
        for j in range(0, 3):
            v2 = changers[j]
            for a in range(0, 3):
                v3 = changers[a]
                for b in range(0, 3):
                    v4 = changers[b]
                    gain_changer.append(np.array([v1, v2, v3, v4]))

#Capping function: if any value of array a is greater than its h_limit or less than l_limit, set it to be equal
def set_on_limit(array, l_limit=[], h_limit=[]): 
    i =0
    if len(h_limit) != 0:
        for g in np.greater(array,h_limit):
            if g:
                array[i] = h_limit[i]
            i+=1
    
    i =0
    if len(l_limit) != 0:
        for s in np.less(array,l_limit):
            if s:
                array[i] = l_limit[i]
            i +=1
    return array

#ROS Subscriber for odometry
def odometry_listener(data):
    
    global odometry, x, y, theta, ind, e, moving, x_vals, y_vals, err2target, done, aggr_gain_time, e_vals, vel_vals
    odometry = data
    
    if control_end or collided or done or err2target <= 5:
        return

    x = odometry.pose.pose.position.x
    y = odometry.pose.pose.position.y
    _, _, theta = quaternion_to_euler(odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w) #rad

    # ---- FIND CLOSEST TRAJECTORY POSITION ----
    
    ind = closest_point([x,y,theta], ref_points)

    # UPDATE ERROR VALUE LIST (only after car starts moving)
    if abs(init_pos[0] - x)>0.1 or abs(init_pos[1]- y)>0.1:
        moving = True
    
    # ---- NEW VEHICLE POSE ----
    x_vals.append(x)
    y_vals.append(y)

    if moving and not control_end and not collided:

        e = get_current_error()

        e_vals[0].append(round(e[0],2))
        e_vals[1].append(round(e[1],2))
        vel_vals.append(velocity.velocity)

#ROS Subscriber - check current frame
def check_status(data):
    global control_end, frame_start, frame, delta, hop
    frame = data.frame
    delta = data.fixed_delta_seconds
    hop = delta
    if moving and frame_start == float('inf'):
        frame_start = frame 
        print("STARTED MOVING: ", frame_start, file=f)
        
    if frame_start != float('inf'):
        if (frame - frame_start) >= int((1/data.fixed_delta_seconds)*LOOP_TIME):
            control_end = True

#Collision subscriber
def collision_sensor(data):
    global collision  
    collision = data

#Adjust angle to be inside range
def adjust(angle):

    final_angle = angle

    while 1:
        if final_angle < -math.pi:
            final_angle = final_angle + 2*math.pi
        elif final_angle > math.pi:
            final_angle = final_angle - 2*math.pi
        else:
            break

    return final_angle

#Convert quaternions to euler angles
def quaternion_to_euler(x, y, z, w):

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z

#ROS Publisher Ackermann Controller - converts velocity & steering values into control actions
def ackermann_controller():

    #Control action
    data = AckermannDrive()
    data.steering_angle = phi
    data.speed = v
    data.steering_angle_velocity = w_s
    #time.sleep(0.05)
    if not moving:
        pub.publish(init_data)
    else:
        pub.publish(data) #Publish control actions
    

#Vehicle follow reference path, register trajectory errors
def control_loop():
    
    global e_b, refpath, v, w_s, phi, ind, e, moving, err2target, done, aggr_gain_time, e_vals, vel_vals

    while ind+ahead<refpath.shape[1] and err2target > 5.0 and not control_end and not collided and not done:

    # ---- DEFINE NEW REF POSITION ----
        if ind+ahead >= refpath.shape[1]:
            idx = refpath.shape[1]-1
        else:
            idx = ind + ahead

        x_ref = refpath[0,idx]
        y_ref = refpath[1,idx]
        theta_ref =refpath[2,idx] #rad
       
        # ---- ERRORS ----
        e_w = np.array([[x_ref-x], [y_ref-y], [adjust(theta_ref-theta)]])
        matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
        e_b = np.matmul(matrix,e_w)
       
        # ---- VELOCITIES ----
        v = K_v*e_b[0][0]
        w_s = K_s*e_b[2][0] + K_l*e_b[1][0] #rad/s
        phi = K_i*(phi + w_s*hop) #steering angle (rad)
    
        if phi>math.radians(30): #Steering is limited to 20 degrees
            phi=math.radians(30)
        elif phi < math.radians(-30):
            phi =math.radians(-30)

        ackermann_controller()

        # ---- CALCULATE ERROR TO TARGET/DESTINATION ----
        err2target = np.linalg.norm(np.subtract([refpath[0][refpath.shape[1]-1],refpath[1][refpath.shape[1]-1]], [x,y]))
    
    # Register values about gains
    t = (frame - frame_start)*delta
    if K_v == 1.0:
        aggr_gain_time['gain'].append(tuple([round(K_v),round(K_l),round(K_s), K_i])) 
    else:
        aggr_gain_time['gain'].append(tuple([K_v,round(K_l),round(K_s), K_i]))

    aggr_gain_time['t'].append(t)
    aggr_gain_time['avg_vel'].append(np.mean(vel_vals))
    aggr_gain_time['avg_ey'].append(np.mean(e_vals[0]))
    aggr_gain_time['avg_et'].append(np.mean(e_vals[1]))
    
    

if __name__ == '__main__':

    try:

        initiate()
        # GAIN LIMITS
        K_min = np.array([1, 1, 1, 0.7])
        K_max = np.array([5.8, 21, 21, 0.98])

        dist_list=[]
        ep_reward_list = []
        
        for episode in range (0, EPISODES):
            
            if f.closed:
                f = open("test_{}_output.txt".format(TEST),'a')

            ep_reward_list.clear()
            ep_reward_list = []
            print("New Episode: ", episode, file=f)
            done = False
            launched = False
            odometry = Odometry()
            odometry.header.seq = 100

            LEARNING_RATE = (episode+1)**(-0.9)
            print("Learning rate: ", LEARNING_RATE, file=f)
            
            # RESET GAINS
            K_v = Kv_0# Linear velocitty gain
            K_l = Kl_0# Linear gain
            K_s = Ks_0# Streering gain
            K_i = 0.7

            #STARTING ROS-BRIDGE - initialize environment
            try:
                ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch', shell=True, preexec_fn=os.setsid)
                time.sleep(4)
                print("Taking a break")
            except:
                print("Trying ros again")
                os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
                time.sleep(4)
                ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch', shell=True, preexec_fn=os.setsid)
                time.sleep(5)
            

            launched = True
            step = 0
            ind = 0
            
            odometry_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, odometry_listener, queue_size=1)

            while abs(init_pos[0] - odometry.pose.pose.position.x)>1 or abs(init_pos[1] - odometry.pose.pose.position.y)>1:
                #print("Waiting for Odometry: ", odometry.header.seq)
                time.sleep(0.1)

            print("ODOMETRY: ", odometry.pose.pose.position.x, " ", odometry.pose.pose.position.y, file=f)

            time.sleep(5)

            #Initialize state S
            state = get_current_error()
            print("Init episode state: ", state, file=f)
            discrete_state = get_discrete_state(state)
            dist_s_d = np.linalg.norm(np.subtract([discrete_state[0], discrete_state[1]], [0,0]))
            dist_s = np.linalg.norm(np.subtract([state[0], 10*state[1]], [0,0]))
            #dist_list.append(dist_s)
            dist_list.append(3.4)

            while not done:
                
                if f.closed:
                    f = open("test_{}_output.txt".format(TEST),'a')         
                
                print("Episode:", episode, file=f)
                print("Step:", step, "Frame: ", frame, file=f)

                print("STATE: ", state, file=f)
                print("DISCRETE STATE: ", discrete_state, file=f)

                E_s = copy.copy(state) # error in state S
                dist_s = np.linalg.norm(np.subtract([state[0], 10*state[1]], [0,0]))
                dist_s_d = np.linalg.norm(np.subtract([discrete_state[0], discrete_state[1]], [0,0]))

                # Reset variables
                vel_vals.clear()
                vel_vals = []
                e_vals.clear()
                e_vals = [[],[],[]]
                x_vals.clear()
                y_vals.clear()
                x_vals = []
                y_vals = []
                collided = False
                collision = None
                moving = False
                frame_start = float('inf')
                err2target = float('inf')
                control_end = False
                odometry = Odometry()
                odometry.header.seq = 100
                velocity = CarlaEgoVehicleStatus()
                velocity.header.seq = 100
                ind = 0

                
                #STARTING ROS-BRIDGE - initialize environment
                if not launched:
                    try:
                        ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch', shell=True, preexec_fn=os.setsid) 
                        print("Taking a break")
                        time.sleep(5)
                    except:
                        print("Trying ros again")
                        os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
                        time.sleep(2)
                        ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch ', shell=True, preexec_fn=os.setsid)
                        time.sleep(5)

                    odometry_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, odometry_listener, queue_size = 1)

                
                # Initializing Subscribers
                data_collision = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent, collision_sensor, queue_size = 1)
                vel_sensor = rospy.Subscriber("/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus, check_velocity, queue_size = 1)
                carla_status = rospy.Subscriber("/carla/status", CarlaStatus, check_status, queue_size = 1)
                
                while abs(init_pos[0] - odometry.pose.pose.position.x)>1 or abs(init_pos[1] - odometry.pose.pose.position.y)>1 or velocity.velocity > 0.2:
                    time.sleep(0.1)

                print("ODOMETRY: ", odometry.pose.pose.position.x, " ", odometry.pose.pose.position.y, file=f)
                print("VELOCITY; ", velocity.velocity, "SEC: ", velocity.header.seq, file=f)

                # Register initial error
                if len(e_vals[0]) == 0 and len(e_vals[1]) == 0:
                    e = get_current_error()
                    e_vals[0].append(e[0])
                    e_vals[1].append(e[1])

                #Take an action (change gain values)
                if np.random.random() > epsilon:
                # Get action from Q table
                    action = np.argmax(q_table[discrete_state])
                    print("Q-table action: ", gain_changer[action], file=f)
                else:
                # Get random action
                    action = np.random.randint(0, action_space_size)
                    print("Random action", file=f)

                K_v = K_v + 1.2*gain_changer[action][0]
                K_l = K_l + 5*gain_changer[action][1]
                K_s = K_s + 5*gain_changer[action][2]
                K_i = K_i + 0.07*gain_changer[action][3]

                aux_k = [K_v, K_l, K_s, K_i]
                aux_k = set_on_limit(aux_k, K_min, K_max)

                K_v = round(aux_k[0],1)
                K_l = round(aux_k[1])
                K_s = round(aux_k[2])
                K_i = round(aux_k[3],2)

                #Register values for plots
                print("GAINS:", K_v, K_l, K_s, K_i, file=f)
                
                # Starting control loop
                control_loop()
                
                print("Step ended. Control_end: ", control_end, " Collided: ", collided, " Done: ", done, file=f)
                print("Time elapsed: ", (frame - frame_start)*delta, file=f)
                print("Control loop finished", file=f)

                # -- UPDATE STATE S'
                new_state = get_avg_error()

                print("NEW STATE: ", new_state, file=f)
                E_new_s = copy.copy(new_state) # error of state S'

                # Discretize S'
                new_discrete_state = get_discrete_state(new_state)

                print("NEW DISCRETE STATE: ", new_discrete_state, file=f)

                state_list.append(new_discrete_state)
                dist_new_s = np.linalg.norm(np.subtract([new_state[0], 10*new_state[1]], [0,0]))
                dist_new_s_d = np.linalg.norm(np.subtract([new_discrete_state[0], new_discrete_state[1]], [0,0]))

                print("Dist: ", dist_new_s, " Min: ", min(dist_list), " WIN: ", WIN, file=f)
                print("Discrete Dist: ", dist_new_s_d, file=f)

                if dist_new_s <= min(dist_list)+WIN:
                    done = True
                    dist_list.append(dist_new_s)
                
                if step >= 80:
                    done = True
                    WIN = WIN + 0.0005


                if done:
                    
                    if step < 80:
                        WIN = 0.001

                    update_gain_lists()
                    define_limits()
                    plt.cla()
                    #axes = plt.gca()
                    #axes.set_xlim([x_vals[len(x_vals)//2]-5,x_vals[len(x_vals)//2]+5])
                    plt.plot(x_vals,y_vals,linestyle='None', marker ="x")
                    plt.plot(refpath[0][:ind], refpath[1][:ind], linestyle='None', marker ="x")
                    plt.title("TRAJECTORY")
                    plt.savefig('/opt/CARLA_0.9.9.4/PythonAPI/examples/plots/test_episode_{}_{}_traj.png'.format(episode,step), bbox_inches='tight')
                    plt.cla()                        
                    plt.title("Errors in y")
                    plt.plot(range(0,len(e_vals[0])), e_vals[0], linestyle='None', marker ="x")
                    plt.savefig('/opt/CARLA_0.9.9.4/PythonAPI/examples/errors/test_sim_episode_{}_error.png'.format(episode), bbox_inches='tight')

                # -- UPDATE REWARD R
                reward_d = (1/(1+(dist_new_s)))-(1/(1+(dist_s))) # component of error in theta
                print("Reward dist component: ", reward_d, file=f)
                reward = reward_d 

                print("Avg velocity: ", np.mean(vel_vals), file=f)

                if collided:
                    reward -= 100

                # -- UPDATE PLOT VARIABLES
                ep_reward_list.append(reward)
                print("REWARD:", reward, file=f)

                #If terminal state reached
                if done:
                    print("Terminal state reached", file=f)

                # -- UPDATE Q TABLE        
                
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # Equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

                if done:
                    print("Q table value for ", discrete_state + (action,) , " is: ", new_q, file=f)

                #S <- S' 
                state = copy.copy(new_state)
                discrete_state = get_discrete_state(state)

                #Reset environment
                odometry_sub.unregister()
                data_collision.unregister()
                vel_sensor.unregister()
                carla_status.unregister()
                os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
                print("Killing roslaunch.Please wait.", file=f)
                time.sleep(2)
                launched = False
                collided = False
                collision = None
                moving = False
                frame_start = float('inf')
                err2target = float('inf')
                control_end = False

                with open(f"qtable-test-{TEST}.pickle", "wb") as a:
                    pickle.dump(q_table, a)

                print("Last Q-Table: episode ", episode, "step ", step, file=f)
                
                f.close()
                step += 1

            # Decaying is being done every episode if episode number is within decaying range
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
                if f.closed:
                    f = open("test_{}_output.txt".format(TEST),'a')
                print("EPSILON: ", epsilon, file=f)
                f.close()
            
            # REGISTER VALUES OF REWARD
            sum_reward = sum(ep_reward_list)
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['sum'].append(sum_reward)
            aggr_ep_rewards['last'].append(ep_reward_list[-1])
            
            # SAVE LISTS IN MEMORY
            with open(f"aggr_ep_rewards_test_{TEST}.pickle", "wb") as b:
                pickle.dump(aggr_ep_rewards, b)

            with open(f"k_tuple-test_{TEST}.pickle", "wb") as c:
                pickle.dump(K_tuple, c)
            
            with open(f"state_list-test_{TEST}.pickle", "wb") as d:
                pickle.dump(state_list, d)
            
            with open(f"aggr_gain_time-test_{TEST}.pickle", "wb") as g:
                pickle.dump(aggr_gain_time, g)

            if f.closed:
                f = open("test_{}_output.txt".format(TEST),'a') 
            print("Last dump: episode ", episode, file=f)

            f.close()

        if f.closed:
            f = open("test_{}_output.txt".format(TEST),'a')      
        
        # RECORD GAIN VALUES
        gain_list = []
        for g in aggr_gain_time['gain']:
            time_list=[]
            vel_list=[]
            ey_list=[]
            et_list=[]
            n = 0
            if g not in gain_list:
                gain_list.append(g)
                for a in range(0,len(aggr_gain_time['gain'])):
                    if aggr_gain_time['gain'][a] == g:
                        n += 1
                        time_list.append(aggr_gain_time['t'][a])
                        vel_list.append(aggr_gain_time['avg_vel'][a])
                        ey_list.append(aggr_gain_time['avg_ey'][a])
                        et_list.append(aggr_gain_time['avg_et'][a])
                avg_time = np.mean(time_list)
                avg_v = np.mean(vel_list)
                avg_ey = np.mean(ey_list)
                avg_et = np.mean(et_list)
                print("Stats of gain ", g, " are avg time: ", avg_time, " avg vel: ", avg_v, " avg ey: ", avg_ey, " avg etheta: ", avg_et, "tested: ", n, file=f)
        f.close()
        plotter()

        # SAVE Q-TABLE
        with open(f"qtable-test-{TEST}-{EPISODES}.pickle", "wb") as a:
            pickle.dump(q_table, a)
        
        f.close()

    except rospy.ROSInterruptException:

        f.close()

        print("Killing ros-bridge")
        os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM) 

        pass