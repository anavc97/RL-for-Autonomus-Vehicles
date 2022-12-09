#!/usr/bin/env python3
# license removed for brevity

from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt
from pygame import K_7
plt.rcParams.update({'font.size': 22})
import rospy
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaCollisionEvent, CarlaEgoVehicleStatus,CarlaStatus, CarlaEgoVehicleInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import time
import os
import subprocess
import statistics
import os
import signal
import pickle
from mpl_toolkits.mplot3d import Axes3D

TEST = "test"

#MSE & K LISTS - list of values for each loop iteration
K_vals = [[],[],[]] # Kv, Kl, Ks
K_tuple = []
e_vals = [[],[],[]]
x_vals = []
y_vals = []
theta_vals = []
xref_vals = []
yref_vals = []
tref_vals = []
error = [[],[],[]]
#LOOP_TIME = 30
#LOOP_BREAK = 30
LOOP_TIME = 5
LOOP_BREAK = 5
#LOOP_TIME = 5000
#LOOP_BREAK = 4000
NR_GAINS = 1

# Map flags
change = False # if True, we are in a lane changing zone
roundabout = False # if True, we are in a roundabout zone

# GAINS
K_v = 0.1# Linear velocitty gain
K_l = 1# Linear gain
K_s = 16# Streering gain
K_i = 0.7
k1 = 0
k2 = 0
k3 = 0
k4 = 0

k5 = 0
k6 = 0
k7 = 0
k8 = 0

v = 0
w_s = 0
phi = 0 
x = 0
y = 0
theta = 0
pub = None
#init_pos = [-45.0,0]
init_pos = [-84.9,124.6056]
PHI_MAX = 30

#Control Loop time
start_time = 0
hop = 0.05
last = 0

#Aux variables
ros_proc = None
collided = False
collision = None
moving = False
start = 0
end= 0
frame_start = float('inf')
control_end = False
frame = 0
delta = 0
velocity = 0
vel_vals = []
ws_vals = []

msg_number = 0
ahead = 20 # Nr. of positions ahead from closes position in refpath
err2target = float('inf')

# State variables
refpath = []
ref_points = []
#REFPATH_FILE = "refpath_round_new.txt"
REFPATH_FILE = "refpath_3.txt"
#REFPATH_FILE = "refpath_full.txt"
ind = 0
odometry = Odometry()
odometry.header.seq = 100
velocity = CarlaEgoVehicleStatus()
velocity.header.seq = 100

#Control action
init_data = AckermannDrive()
init_data.speed = 0.5

#Initiate some variables
def initiate():
    global pub
    get_refpath()
    rospy.init_node('ackermann', anonymous=True)
    pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size=1)


def closest_point(pos, refs):
    refs = np.asarray(refs)
    deltas = refs - pos
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

#Calculate current error - error to closest refpath position
def get_current_error():

    global error

    ref_x = refpath[0,ind]
    ref_y = refpath[1,ind]
    ref_theta =refpath[2,ind] #rad
    
    print("NOISE: ", x-odometry.pose.pose.position.x,  y-odometry.pose.pose.position.y)

    #print("Ref: ", ref_x, ref_y, ref_theta, " pose: ", x,y,theta)
    # ---- ERRORS ----
    e_world = np.array([[ref_x-x], [ref_y-y], [adjust(ref_theta-theta)]])
    matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
    e_car = np.matmul(matrix,e_world)
    error[0].append(e_car[0][0])
    error[1].append(e_car[1][0])
    error[2].append(e_car[2][0])
    print("ERROR: ", error[0][-1], error[1][-1], error[2][-1])
    return [abs(e_car[1][0]),abs(e_car[2][0])] 

# Get reference path from file
def get_refpath():

    global refpath, ref_points

    file1 = open(REFPATH_FILE,"r+") 
    
    ref_x = [float(s) for s in file1.readline().split(', ')]
    ref_y = [float(s) for s in file1.readline().split(', ')]
    ref_theta = [float(s) for s in file1.readline().split(', ')]
    file1.close()
    #refpath=np.array([ref_x,ref_y,ref_theta])
    refpath=np.array([ref_x[:600],ref_y[:600],ref_theta[:600]])
    #refpath=np.array([[-88.9]*len(ref_x[:ahead*10]),[100]*len(ref_y[:ahead*10]),[-1.57]*len(ref_theta[:ahead*10])])
    for i in range(0,len(refpath[1])):
        ref_points.append([refpath[0][i], refpath[1][i], refpath[2][i]])

# ROS Subscriber for vehicle velocity
def check_velocity(data):
    global velocity
    velocity = data
        
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

def high_level_controller():

    global roundabout, change, PHI_MAX

    if -58 < x < 33 and -39 < y < 50:
        roundabout = True
        change = False
    elif -90.5 < x < -73 and 8.5 < y < 127.4:
        change = True
        roundabout = False
    else:
        roundabout = False
        change = False
        PHI_MAX = 30

    #if 8.5<y<100.0 and -90.5 < x < -73:
    #    PHI_MAX = 10

    print("Flags - roundabout: ", roundabout, " lane change: ", change, "GAINS: ", K_v, K_l, K_s, "PHI: ", PHI_MAX)

#ROS Subscriber for odometry
def odometry_listener(data):
    
    global odometry, x, y, theta, msg_number, ind, e, moving, x_vals, y_vals, theta_vals, err2target, e_vals, vel_vals, start, end, xref_vals, yref_vals, tref_vals, ws_vals
    msg_number += 1
    odometry = data
    if control_end or collided:
        return

    theta_noise = np.random.triangular(-0.088, 0, 0.088)

    #print(theta_noise)
    x = odometry.pose.pose.position.x + np.random.normal(loc=0.0, scale=0.1, size=None)
    y = odometry.pose.pose.position.y + np.random.normal(loc=0.0, scale=0.1, size=None)
    _, _, theta = quaternion_to_euler(odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w) #rad
    theta += theta_noise

    #print("NOISE: ", x-odometry.pose.pose.position.x,  y-odometry.pose.pose.position.y, theta_noise)
    # ---- FIND CLOSEST TRAJECTORY POSITION ----
    
    ind = closest_point([x,y,theta], ref_points)

    if abs(init_pos[0] - x)>0.1 or abs(init_pos[1]- y)>0.1:
        moving = True

    if moving and not control_end:

        if start == 0:
            start = frame
            end = frame
            

        for i in range(0,(frame-end)):
            
            e = get_current_error()
            e_vals[0].append(e[0])
            e_vals[1].append(e[1])

            x_vals.append(x)
            y_vals.append(y)
            theta_vals.append(theta)

            xref_vals.append(refpath[0,ind])
            yref_vals.append(refpath[1,ind])
            tref_vals.append(refpath[2,ind])

            # Current velocities
            vel_vals.append(v)
            ws_vals.append(w_s)
            
        end = frame

    high_level_controller()

#ROS Subscriber - check current frame
def check_status(data):
    global control_end, frame_start, frame, delta, hop, last
    frame = data.frame
    delta = data.fixed_delta_seconds
    hop = delta
    if moving and frame_start == float('inf'):
        frame_start = frame
    if frame_start != float('inf'):
        if (frame - frame_start) == int((1/data.fixed_delta_seconds)*(LOOP_TIME-LOOP_BREAK)):
            print("SECOND CHECKPOINT")
            last = len(error[0])
        if (frame - frame_start) >= int((1/data.fixed_delta_seconds)*LOOP_TIME):
            control_end = True

#Collision subscriber
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

def agent():

    global K_v, K_l, K_s, K_i, ahead

    if change and not roundabout:

        K_v = k1
        K_l = k2
        K_s = k3
        K_i = k4

        #K_v = 0.1
        #K_l = 1
        #K_s = 1
        #ahead = 40
        #Register values for plots
        #print("GAINS:", K_v, K_l, K_s)

    elif roundabout and not change: 

        K_v = k5
        K_l = k6
        K_s = k7
        K_i= k8
        
        #K_v = 3.4
        #K_l = 16
        #K_s = 1
        #ahead = 25
        #Register values for plots
        #print("GAINS:", K_v, K_l, K_s)
    
    else:
        ahead = 40
    

#Vehicle follow reference path, register trajectory errors
def control_loop():
    
    global e_b, refpath, v, w_s, phi, ind, e, moving, err2target, e_vals, vel_vals, end, PHI_MAX

    while not control_end and not collided and err2target > 5.0:
    # ---- DEFINE NEW REF POSITION ----
        if ind+ahead >= refpath.shape[1]:
            idx = refpath.shape[1]-1
        else:
            idx = ind + ahead
        
        x_ref = refpath[0,idx]
        y_ref = refpath[1,idx]
        theta_ref =refpath[2,idx] #rad
    
        # ---- ERRORS -----
        e_w = np.array([[x_ref-x], [y_ref-y], [adjust(theta_ref-theta)]])
        matrix = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],[0, 0, 1]])
        e_b = np.matmul(matrix,e_w)

        agent()

        # ---- VELOCITIES ----
        v = K_v*e_b[0][0]
        w_s = K_s*e_b[2][0] + K_l*e_b[1][0] #rad/s
        phi = K_i*(phi + w_s*hop) #steering angle (rad)

        if phi>math.radians(PHI_MAX): #Steering is limited to 30 degrees
            phi=math.radians(PHI_MAX)
        elif phi < math.radians(-PHI_MAX):
            phi =math.radians(-PHI_MAX)

        if -90.5 < x < -73 and y > 127.4:
            phi = 0 
            w_s = 0

        ackermann_controller()

        # ---- CALCULATE ERROR TO TARGET/DESTINATION ----
        err2target = np.linalg.norm(np.subtract([refpath[0][refpath.shape[1]-1],refpath[1][refpath.shape[1]-1]], [x,y]))
    
    
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
    

if __name__ == '__main__':

    try:
        laps = 0
        N_LAPS = 10
        f = open('test_{}_output.txt'.format(TEST),'w')
        K_list = []
        #K_list.append([[1.84,1,1,0.91], [2.2,6,11,0.7]])
        #K_list.append([[3,21,21,0.98], [5.8,16,11,0.84]])
        #K_list.append([3.4,11,21,0.84])
        #K_list.append([3.4,21,11,0.77])
        #K_list.append([4.6,6,1,0.84])
        K_list.append([[3,21,21,0.7], [3.4,21,1,0.84]])#CHOSEN ONE

        for nr in range(0, NR_GAINS):
            
            [k1,k2,k3,k4] = K_list[nr][0]
            [k5,k6,k7,k8] = K_list[nr][1]
            
            if f.closed:
                f = open("test_{}_output.txt".format(TEST),'a')
                 
            print("##################################### GAINS:", K_list[nr], file=f)
            highest = 0
            mse_vals = []
            mse2_vals = []
            e_list = []

            for laps in range(0,N_LAPS):

                initiate()
                control_end = False
                moving = False
                frame_start = float('inf')
                x_vals = []
                y_vals = []
                
                K_min = np.array([1, 1, 1])
                K_max = np.array([8.2, 21, 21])
                #K_min = np.array([0.1, 1, 1, 0.7])
                #K_max = np.array([3, 21, 21, 0.98])
                    
                if f.closed:
                    f = open("test_{}_output.txt".format(TEST),'a')
                
                
                odometry = Odometry()
                odometry.header.seq = 100

                '''
                with open("k_tuple-test_24_5-29.pickle", "rb") as a:
                    K_list = pickle.load(a)
                
                K_list.pop(21)
                K_list.pop(21)
                K_list.pop(21)
                K_list.pop(24)
                

                [k1, k2, k3] = centroid(K_list[-15:])

                if laps == 0:
                    print('The center of mass is ', [k1,k2,k3], file=f)
                
                print('The center of mass is ', [k1,k2,k3])
                
                with open("k_tuple-test_6-19.pickle", "rb") as a:
                    K_list2 = pickle.load(a)
                
                K_list2.pop(4)
                
                [k4, k5, k6] = centroid(K_list2[-10:])
                if laps == 0:
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

                while abs(init_pos[0] - odometry.pose.pose.position.x)>1 or abs(init_pos[1] - odometry.pose.pose.position.y)>1:
                    #print("Waiting for Odometry: ", odometry.header.seq)
                    time.sleep(0.1)

                #print("ODOMETRY: ", odometry.pose.pose.position.x, " ", odometry.pose.pose.position.y, file=f)
                    
                if f.closed:
                    f = open("test_{}_output.txt".format(TEST),'a')              
                
                # Reset variables
                vel_vals.clear()
                vel_vals = []
                e_vals.clear()
                e_vals = [[],[],[]]
                collided = False
                collision = None
                start = 0
                end = 0
                err2target = float('inf')
                odometry = Odometry()
                odometry.header.seq = 100
                velocity = CarlaEgoVehicleStatus()
                velocity.header.seq = 100
                ind = 0
                msg_number = 0
                
                # Starting control loop
                control_loop()

                #if laps == N_LAPS-1:
                #print("Simulation ended. Collided: ", collided, file=f)
                #    print("Control loop finished", file=f)
                #print("Integral gain: ", 1-0.025*laps, file=f)
                #Reset environment
                odometry_sub.unregister()
                data_collision.unregister()
                vel_sensor.unregister()
                carla_status.unregister()
                os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM)
                #if laps == N_LAPS-1:
                    #print("Killing roslaunch.Please wait.", file=f)
                time.sleep(2)
                launched = False
                collided = False
                collision = None
                err2target = float('inf')
                #odometry = None
                    
                mse = []
                print("LEN ERROR: ", len(error[0]))
                
                for i in range(0,len(error[0])):
                    mse.append((error[1][i]**2 + error[0][i]**2)/2)
                
                print("LEN MSE: ", len(mse))
                t = [i * delta for i in list(range(0,(end-start)))]

                mse2 = mse[last:]
                
                print("The average MSE is ", np.average(mse), file=f)
                print("The average MSE2 is ", np.average(mse2), file=f)

                print("LEN MSE2: ", len(mse2), " last: ", last)
                print("The average MSE is ", np.average(mse))
                print("The average MSE2 is ", np.average(mse2))
                mse_vals.append(np.average(mse))
                mse2_vals.append(np.average(mse2))

                e_list.append(np.mean(e_vals[1][last:]))
                #print("The average theta error is ", np.mean(e_vals[1][last:]), file=f)

                if np.average(mse2) > highest:
                    highest = np.average(mse2)
                    print("Saved with mse2 = ", np.average(mse2), file=f)

                    with open(f"test-{TEST}-refpath.pickle", "wb") as a:
                        pickle.dump(refpath, a)

                    with open(f"test-{TEST}-x_vals.pickle", "wb") as a:
                        pickle.dump(x_vals, a)  

                    with open(f"test-{TEST}-y_vals.pickle", "wb") as a:
                        pickle.dump(y_vals, a)    

                    with open(f"test-{TEST}-t.pickle", "wb") as a:
                        pickle.dump(t, a)

                    with open(f"test-{TEST}-error.pickle", "wb") as a:
                        pickle.dump(error, a)  

                    with open(f"test-{TEST}-mse.pickle", "wb") as a:
                        pickle.dump(mse, a)  
                    
                    with open("test_{}_ws.pickle".format(TEST), "wb") as b:
                        pickle.dump(ws_vals, b)

                    with open("test_{}_vel_vals.pickle".format(TEST), "wb") as b:
                        pickle.dump(vel_vals, b)
                
                if len(mse_vals) >= 2 and laps == N_LAPS-1:
                    print("MSE STDEV: ", statistics.stdev(mse_vals), np.std(mse_vals))
                    print("MSE2 STDEV: ", statistics.stdev(mse2_vals), np.std(mse2_vals))
                    print("E STDEV: ", statistics.stdev(e_list), np.std(e_list))
                    print("MSE STDEV: ", statistics.stdev(mse_vals), np.std(mse_vals), file=f)
                    print("MSE2 STDEV: ", statistics.stdev(mse2_vals), np.std(mse2_vals), file=f)
                    print("E STDEV: ", statistics.stdev(e_list), np.std(e_list), file=f)
                
                print("-----------------------------------------", file=f)
                
                f.close()

                # Reset variables
                vel_vals.clear()
                vel_vals = []
                e_vals.clear()
                mse.clear()
                mse2.clear()
                error.clear()
                x_vals.clear()
                y_vals.clear()
                error = [[],[],[]]
                e_vals = [[],[],[]]
                collided = False
                collision = None
                start = 0
                end = 0
                err2target = float('inf')
                odometry = Odometry()
                odometry.header.seq = 100
                velocity = CarlaEgoVehicleStatus()
                velocity.header.seq = 100
                ind = 0
                msg_number = 0
                control_end = False
                moving = False
                frame_start = float('inf')
        
    except rospy.ROSInterruptException:

        f.close()

        print("Killing ros-bridge")
        os.killpg(os.getpgid(ros_proc.pid), signal.SIGTERM) 

        pass