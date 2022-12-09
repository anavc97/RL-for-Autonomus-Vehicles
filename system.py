#!/usr/bin/env python3
# license removed for brevity

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib as mpl
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
import os
import subprocess
import os
import signal
import copy
import pickle
import datetime
from std_msgs.msg import String

# TEST NUMBER
TEST = "test"

#List of values for each loop iteration
K_vals = [[],[],[]] # Kv, Kl, Ks
K_tuple = []
e_vals = [[],[],[]]
x_vals = []
y_vals = []
error = [[], [], []]
vel_vals = []
v_vals = []
theta_vals = []
xref_vals = []
yref_vals = []
tref_vals = []
phi_vals = []
ws_vals = []
#LOOP_TIME = 8
#LOOP_BREAK = 8
LOOP_TIME = 30
LOOP_BREAK = 30

# Map flags
change = False # if True, we are in a lane changing zone
roundabout = False # if True, we are in a roundabout zone

# GAINS
K_v = 0.1# Linear velocitty gain
K_l = 1# Linear gain
K_s = 16# Streering gain
K_i = 0.7
K_d = 0
k1 = 0
k2 = 0
k3 = 0
k4 = 0
k5 = 0
k6 = 0

# Control variables
v = 0
w_s = 0
phi = 0 
x = 0
y = 0
theta = 0
PHI_MAX = 30
pub = None
ahead = 25 # Nr. of positions ahead from closes position in refpath
err2target = float('inf')
refpath = []
ref_points = []
#REFPATH_FILE = "refpath_full.txt"
#REFPATH_FILE = "refpath_half_full_new.txt"
REFPATH_FILE = "refpath_round_new.txt"
#REFPATH_FILE = "refpath_3.txt"

#Initial positions
#init_pos = [-84.9,127.6056]
init_pos = [-45.0,0]
#init_pos = [-84.9,124.6056]

#Control Loop time
start_time = 0
hop = 0.05
end_frame = 0
start_frame = 0
last = 0
frame_start = float('inf')
frame = 0
delta = 0
ind = 0

# Simulation flags
collision = None
moving = False
control_end = False
collided = False

#Aux variables
end = False # flag to stop simulation
f = open("{}_system_output.txt".format(TEST),'w')

# CARLA-ROS variables
odometry = Odometry()
odometry.header.seq = 100
velocity = CarlaEgoVehicleStatus()
velocity.header.seq = 100
ros_proc = None
obs_detect = None
init_data = AckermannDrive()
init_data.speed = 0.5

# Initiate some variables
def initiate():
    global pub
    get_refpath()
    rospy.init_node('ackermann', anonymous=True)
    pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size=1)

# Find closest point from position pos
def closest_point(pos, refs):
    refs = np.asarray(refs)
    deltas = refs - pos
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

#Calculate current error - error to closest refpath position
def get_current_error():

    global error

    # Current reference
    ref_x = refpath[0,ind]
    ref_y = refpath[1,ind]
    ref_theta =refpath[2,ind] #rad
    
    # ---- ERRORS ----
    e_world = np.array([[ref_x-x], [ref_y-y], [adjust(ref_theta-theta)]]) # error in world frame
    matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
    e_car = np.matmul(matrix,e_world) # error in vehicle frame
    error[0].append(e_car[0][0])
    error[1].append(e_car[1][0])
    error[2].append(e_car[2][0])

    return [abs(e_car[1][0]),abs(e_car[2][0])] 

# Find center of mass of points
def centroid(points):
    
    x_coords = []        
    y_coords = []
    z_coords = []
    for p in points:
        x_coords.append(p[0])
        y_coords.append(p[1])
        z_coords.append(p[2])
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    centroid_z = sum(z_coords)/_len
    return [centroid_x, centroid_y, centroid_z]

# Get reference path from file
def get_refpath():

    global refpath, ref_points

    file1 = open(REFPATH_FILE,"r+") 
    
    ref_x = [float(s) for s in file1.readline().split(', ')]
    ref_y = [float(s) for s in file1.readline().split(', ')]
    ref_theta = [float(s) for s in file1.readline().split(', ')]
    file1.close()
    refpath=np.array([ref_x,ref_y,ref_theta])
    #refpath=np.array([ref_x[:500],ref_y[:500],ref_theta[:500]])
    for i in range(0,len(refpath[1])):
        ref_points.append([refpath[0][i], refpath[1][i], refpath[2][i]])

# ROS Subscriber for vehicle velocity
def check_velocity(data):
    global velocity
    velocity = data

# ROS Subscriber for vehicle velocity
def check_obstacle(data):
    global obs_detect
    obs_detect = data
        
#Get error average
def get_avg_error():
    if len(e_vals[0]) != 0 and len(e_vals[1]) != 0 :
        return [np.mean(e_vals[0]), np.mean(e_vals[1])]
    else:
        return [0,0]

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

# HIGH LEVEL CONTROLLER - controls map flags and max steering angle range
def high_level_controller():

    global roundabout, change, PHI_MAX

    if -58 < x < 33 and -39 < y < 50:
        #print("Roundabout frame: ", frame, file=f)
        roundabout = True
        change = False
    elif -90.5 < x < -73 and 8.5 < y < 127.4:
        #print("Lane frame: ", frame, file=f)
        change = True
        roundabout = False
    else:
        roundabout = False
        change = False
        PHI_MAX = 30
    '''
    if 8.5<y<73.0 and -90.5 < x < -73:
        PHI_MAX = 10
    elif -7< x <10 and 50< y <127.4:
        PHI_MAX = 10
    else:    
        PHI_MAX = 30
    '''
    print("Flags - roundabout: ", roundabout, " lane change: ", change, "GAINS: ", K_v, K_l, K_s, K_i, "PHI: ", PHI_MAX)
    #print("Phi Max: ", PHI_MAX)
    #print("Velocity: ", velocity.velocity)
    #print("GAINS:", K_v, round(K_l,2), round(K_s,2))

#ROS Subscriber for odometry
def odometry_listener(data):
    
    global odometry, x, y, theta, v_vals, ind, e, moving, x_vals, y_vals, err2target, done, aggr_gain_time, e_vals, vel_vals, end_frame, start_frame, xref_vals, yref_vals, tref_vals, phi_vals, ws_vals
    odometry = data

    theta_noise = np.random.triangular(-0.088, 0, 0.088)

    #print(theta_noise)
    x = odometry.pose.pose.position.x# + np.random.normal(loc=0.0, scale=0.1, size=None)
    y = odometry.pose.pose.position.y# + np.random.normal(loc=0.0, scale=0.1, size=None)
    _, _, theta = quaternion_to_euler(odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w) #rad
    #theta += theta_noise

    # ---- FIND CLOSEST TRAJECTORY POSITION ----
    
    ind = closest_point([x,y,theta], ref_points)

    # Checking stopping conditions
    if err2target <= 5 or collided:
        return

    #Checking movement of the vehicle
    if abs(init_pos[0] - x)>0.1 or abs(init_pos[1]- y)>0.1:
        moving = True

    if moving and not control_end:
        
        if start_frame == 0:
            start_frame = frame
            print("Start frame: ", start_frame, file=f)
            end_frame = frame
            
        # Register current control values
        for i in range(0,(frame-end_frame)):

            e = get_current_error()
            e_vals[0].append(e[0])
            e_vals[1].append(e[1])

            x_vals.append(x)
            y_vals.append(y)
            theta_vals.append(theta)
            phi_vals.append(phi)
            ws_vals.append(w_s)

            xref_vals.append(refpath[0,ind])
            yref_vals.append(refpath[1,ind])
            tref_vals.append(refpath[2,ind])
            
            # Current velocity
            vel_vals.append(copy.copy(velocity.velocity))
            '''
            # Imposed velocity
            if v >= 5.56:# 14.4kmh
                v_vals.append(5.56)
            else:
                v_vals.append(v)
            '''
            v_vals.append(v)
            
        end_frame = frame

    # Update map flags
    high_level_controller()

#ROS Subscriber - check current frame
def check_status(data):
    global control_end, frame_start, frame, delta, hop
    frame = data.frame
    delta = data.fixed_delta_seconds
    hop = delta
    if moving and frame_start == float('inf'):
        frame_start = frame
    if frame_start != float('inf'):
        if (frame - frame_start) >= int((1/data.fixed_delta_seconds)*LOOP_TIME):
            control_end = True

#Collision subscriber - only callbacked if collision is registered
def collision_sensor(data):
    global collision, collided  
    collision = data
    collided = True

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
    pub.publish(data) #Publish control actions

# Updates gains etc. according to map flags
def agent():

    global K_v, K_l, K_s, ahead, change, roundabout
    
    # If inside blue zone 
    if change and not roundabout:

        K_v = k1
        K_l = k2
        K_s = k3

        K_v = 3
        K_l = 21
        K_s = 21
        K_i = 0.7
        ahead = 20
        

    #If inside red zone
    elif roundabout and not change: 

        K_v = k4
        K_l = k5
        K_s = k6

        K_v = 3.4
        K_l = 21
        K_s = 1
        K_i = 0.84
        ahead = 25

    else:
        ahead = 40

#Vehicle follow reference path, register trajectory errors
def control_loop():
    
    global e_b, refpath, v, w_s, phi, ind, e, moving, err2target, done, aggr_gain_time, e_vals, vel_vals, end, PHI_MAX, last

    e_b = None
    vi = 0

    while not control_end and err2target > 5.0 and not collided and not end:

    # ---- DEFINE NEW REF POSITION ----
        if ind+ahead >= refpath.shape[1]:
            idx = refpath.shape[1]-1
        else:
            idx = ind + ahead
        
        x_ref = refpath[0,idx]
        y_ref = refpath[1,idx]
        theta_ref =refpath[2,idx] #rad

        if e_b is None:
            e_b_prev = None
        else:
            e_b_prev = copy.deepcopy(e_b)

        # ---- ERRORS ----
        e_w = np.array([[x_ref-x], [y_ref-y], [adjust(theta_ref-theta)]])
        matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
        e_b = np.matmul(matrix,e_w)

        agent()

        # Calculate derivative component for e_y
        if e_b_prev is None:
            d_comp = 0
        else:
            d_comp = (e_b[0][0] - e_b_prev[0][0])/hop

        # ---- VELOCITIES ----
        v = K_v*e_b[0][0]
        w_s = K_s*e_b[2][0] + K_l*e_b[1][0] #rad/s
        phi = K_i*(phi + w_s*hop) #steering angle (rad)

        '''
        # ---- CHECK FOR OBSTACLES/ MANAGE VELOCITY ----
        if obs_detect.data != 100:
            if vi == 0:
                vi = v
            if float(obs_detect.data) <= float(20.0):
                print("Obj frame: ", frame, file=f)
                print("LEN: ", len(error[0]), file=f)
                if last == 0:
                    last = len(error[0])
                v = vi*(obs_detect.data/20)
                w_s = 0
                phi = 0
        else:
            v = K_v*e_b[0][0]+ K_d*d_comp
        
        v = K_v*e_b[0][0]+ K_d*d_comp
        
        print("Velocity imposed: ", v)
        #if d_comp != 0.0:
        #    print("d_comp: ", K_d*d_comp)
        
        # Define imposed velocity limit    
        if v >= 5.56:# 14.4kmh
            v = 5.56
        '''
        if velocity.velocity >= 4:
            print("Velocity exceeded")
        
        #Orange zones
        if phi>math.radians(PHI_MAX): #Steering is limited to 20 degrees
            phi=math.radians(PHI_MAX)
        elif phi < math.radians(-PHI_MAX):
            phi =math.radians(-PHI_MAX)

        #Green Zone
        if -90.5 < x < -73 and y > 127.4:
            phi = 0 
            w_s = 0

        ackermann_controller()
        
        '''
        # Check if vehicle has stopped
        if len(vel_vals) > 0:
            if np.mean(vel_vals[-200:]) < 0.1 and obs_detect.data != 100:
                end = True
        '''
        # ---- CALCULATE ERROR TO TARGET/DESTINATION ----
        err2target = np.linalg.norm(np.subtract([refpath[0][refpath.shape[1]-1],refpath[1][refpath.shape[1]-1]], [x,y]))
      

if __name__ == '__main__':

    try:

        initiate()
        
        # GAIN LIMITS
        K_min = np.array([1, 1, 1])
        K_max = np.array([8.2, 21, 21])
            
        if f.closed:
            f = open("test_{}_system_output.txt",'a')
        
        odometry = Odometry()
        odometry.header.seq = 100
        '''
        # RESET GAINS FOR LANE CHANGE
        with open("k_tuple-test_24_5-29.pickle", "rb") as a:
            K_list = pickle.load(a)
        
        #popping values from episodes 21, 22, 23 and 27
        K_list.pop(21)
        K_list.pop(21)
        K_list.pop(21)
        K_list.pop(24)
        
        # Defining gains for lane change
        [k1, k2, k3] = centroid(K_list[-15:])

        print('The center of mass is ', [k1,k2,k3], file=f)
        print('The center of mass is ', [k1,k2,k3])
        
        # RESET GAINS FOR ROUNDABOUT NAVIGATION
        with open("k_tuple-test_6-19.pickle", "rb") as a:
            K_list2 = pickle.load(a)

        #popping values from episode 4
        K_list2.pop(4)
        
        # Defining gains for roundabout nav
        [k4, k5, k6] = centroid(K_list2[-10:])

        print('The center of mass is ', [k4,k5,k6], file=f)
        print('The center of mass is ', [k4,k5,k6])
        '''
        #STARTING ROS-BRIDGE - initialize environment
        try:
            ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch', shell=True, preexec_fn=os.setsid)
            time.sleep(2)
            print("Taking a break")
        except:
            print("Trying ros again")
            os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
            time.sleep(2)
            ros_proc = subprocess.Popen('roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch', shell=True, preexec_fn=os.setsid)
            time.sleep(5)
        
        # Initializing Subscribers
        odometry_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, odometry_listener, queue_size=1)
        data_collision = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent, collision_sensor, queue_size = 1)
        vel_sensor = rospy.Subscriber("/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus, check_velocity, queue_size = 1)
        carla_status = rospy.Subscriber("/carla/status", CarlaStatus, check_status, queue_size = 1)
        obstacle_detector = rospy.Subscriber("/carla/obstacle_detector", Float32, check_obstacle, queue_size = 1)

        while abs(init_pos[0] - odometry.pose.pose.position.x)>1 or abs(init_pos[1] - odometry.pose.pose.position.y)>1:
            time.sleep(0.1)

        #print("ODOMETRY: ", odometry.pose.pose.position.x, " ", odometry.pose.pose.position.y, file=f)
            
        if f.closed:
            f = open("test_{}_system_output.txt",'a')            
        
        # GIVE TIME TO MANUALLY INITIALIZE OBJECT DETECTOR
        print("Waiting for object detector...")
        time.sleep(10)

        # Reset variables
        vel_vals.clear()
        vel_vals = []
        e_vals.clear()
        e_vals = [[],[],[]]
        collided = False
        collision = None
        err2target = float('inf')
        odometry = Odometry()
        odometry.header.seq = 100
        velocity = CarlaEgoVehicleStatus()
        velocity.header.seq = 100
        ind = 0
        

        #Register values for plots
        print("GAINS:", K_v, K_l, K_s, K_i, file=f)
        
        # Starting control loop
        control_loop()
        
        print("Simulation ended. Collided: ", collided, file=f)
        print("Control loop finished", file=f)

        #Reset environment
        odometry_sub.unregister()
        data_collision.unregister()
        vel_sensor.unregister()
        carla_status.unregister()
        obstacle_detector.unregister()
        os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
        print("Killing roslaunch.Please wait.", file=f)
        time.sleep(2)
        launched = False
        collided = False
        collision = None
        err2target = float('inf')

        t = [i * delta for i in list(range(0,(end_frame-start_frame)))] # ARRAY OF TIME-STEPS

        # CALCULATE MSE
        mse = []
        for i in range(0,len(error[0])):
            mse.append((error[1][i]**2 + error[0][i]**2)/2)
        
        mse2 = mse[:last] # COSTUM MSE VALUES
        print("LEN MSE: ", len(mse), file=f)
        print("LEN MSE2: ", len(mse2), file=f)

        print("The average MSE is ", np.average(mse), file=f)
        print("The average MSE2 is ", np.average(mse2), file=f)
        
        print("The average MSE is ", np.average(mse))
        print("The average MSE2 is ", np.average(mse2))

        # SAVE LIST VALUES IN MEMORY
        with open("test_{}-x_vals.pickle".format(TEST), "wb") as b:
            pickle.dump(x_vals, b)
            
        with open("test_{}-y_vals.pickle".format(TEST), "wb") as b:
            pickle.dump(y_vals, b)
        
        with open("test_{}_mse.pickle".format(TEST), "wb") as b:
            pickle.dump(mse, b)
            
        with open("test_{}_v_vals.pickle".format(TEST), "wb") as b:
            pickle.dump(v_vals, b)

        with open("test_{}_vel_vals.pickle".format(TEST), "wb") as b:
            pickle.dump(vel_vals, b)

        with open("test_{}-t.pickle".format(TEST), "wb") as b:
            pickle.dump(t, b)

        with open("test_{}_error.pickle".format(TEST), "wb") as b:
            pickle.dump(error, b)

        with open("test_{}_phi.pickle".format(TEST), "wb") as b:
            pickle.dump(phi_vals, b)
        
        with open("test_{}_ws.pickle".format(TEST), "wb") as b:
            pickle.dump(ws_vals, b)

        # PLOTS
        plt.plot(refpath[0][:ind], refpath[1][:ind], marker ="x", color="orange", markersize=7)
        plt.plot(x_vals,y_vals,linestyle='None', marker ="x", color="blue")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("TRAJECTORY")
        plt.show()
        plt.plot(t[:len(v_vals)],v_vals,linestyle='None', marker ="x")
        plt.plot(t[:len(vel_vals)],vel_vals,linestyle='None', marker ="x")
        plt.title("VELOCITY")
        plt.xlabel("Time (sim. secs.)")
        plt.ylabel("Velocity (m/s)")
        plt.show()
        '''
        plt.cla()
        plt.title("Mean Square Error")
        plt.plot(t, mse, linestyle='None', marker ="x")
        plt.xlabel("Time (sim. secs)")
        plt.ylabel("MSE")
        plt.show()
        
        plt.cla()                        
        plt.title("Errors in " + r"$\theta$")
        plt.plot(t, [abs(i) for i in error[2]], linestyle='None', marker ="x")
        plt.xlabel("Time (sim. secs)")
        plt.ylabel(r"$e_{\theta}$")
        plt.show()
        '''
        plt.cla()                        
        plt.title("Steering angle velocity")
        plt.plot(t[:len(ws_vals)], ws_vals, linestyle='None', marker ="x")
        plt.xlabel("Time (sim. secs)")
        plt.ylabel(r"$\omega_s}$ (rad/s)")
        plt.show()
        
        f.close()

    except rospy.ROSInterruptException:

        f.close()


        print("Killing ros-bridge")
        os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM) 

        pass