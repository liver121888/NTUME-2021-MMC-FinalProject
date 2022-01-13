import argparse
import threading
import time
import pickle as pk
from dynamixel_sdk.protocol2_packet_handler import ERRNUM_DATA_LENGTH
import serial

from Ax12 import Ax12

def go_to(tar_pos,speed):
    num = len(tar_pos)
    cur_pos = []

    #goto end point with specific speed
    for i in range(num):
        id=i+1
        Ax12(id).set_cw_compliance_slope(128)
        Ax12(id).set_ccw_compliance_slope(128)
        Ax12(id).set_moving_speed(speed)

    for i in range(num):
        id=i+1
        Ax12(id).set_goal_position(tar_pos[i])

    # time.sleep(1)
    # Ax12(num).set_goal_position(tar_pos[num-1])

    #get current position 
    for i in range(num):
        id=i+1
        cur_pos.append(Ax12(id).get_present_position())

    print(cur_pos)
        

#request current position
num_motors = 5
baud = 1000000
rest = 0
left = 2
fist = 1

# motor_rest = [512,700,375,350,256]
# motor_left1 = [780,800,490,345,150]
# motor_left2 = [780,800,490,345,510]
#motor_fist1 = [320,650,350,600,510]
#motor_fist2 = [320,650,350,600,150]
motor_rest = [512,700,375,350,256]
motor_left1 = [780,733,154,644,150]
motor_left2 = [780,733,154,644,510]
motor_fist0 = [320,853,354,544,500]
motor_fist1 = [320,733,154,644,500]
motor_fist2 = [320,733,154,644,150]

mspeed = [50,180,150]

status = rest
motor_pos = motor_rest
current_pos = []
Ax12.BAUDRATE = baud
Ax12.DEVICENAME = 'COM5'
Ax12.connect()
go_to(motor_pos,30)
try:
    while(True):
    #input position mode
        input_number=int(input("input:(0~2)\t"))
        if (input_number<=2) & (input_number>=0) & (input_number!=status):
            status = input_number
            if input_number==left:
                motor_pos = motor_left1
                go_to(motor_pos,mspeed[status])
                time.sleep(0.5)
                motor_pos = motor_left2
                
            elif input_number==fist:
                motor_pos = motor_fist0
                go_to(motor_pos,mspeed[status])
                time.sleep(1)
                motor_pos = motor_fist1
                go_to(motor_pos,mspeed[status])
                time.sleep(0.5)
                motor_pos = motor_fist2
                
            else:
                motor_pos = motor_rest
            go_to(motor_pos,mspeed[status])
            
            

except:
    print('Press Ctrl-C to terminate while statement')

Ax12.disconnect()
