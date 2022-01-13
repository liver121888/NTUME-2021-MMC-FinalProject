import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit
import csv
import numpy as np

# Set up the serial line
ser = serial.Serial('COM6', 115200) # Arduino COM and Baudrate
time.sleep(1)

# Start counting time
# timer_start = timeit.default_timer()
while(True):
    input_number=input("input:\t")
    name_file = f"Left_{input_number}.csv"
    # Waiting for starting
    time.sleep(1)
    # Start 
    print("Starting...")
    # Prepare
    time.sleep(0.5)
    # Read and record the data
    # name="TurnLeft_27.csv"

    # Empty list to store the data
    SaveData = []
    empty = [None]*10000
    
    
    for i in range(10000): 
        EMGval = ser.readline()                  # Read a byte string
        string_n = EMGval.decode("UTF-8")        # Decode byte string to Unicode
        string_EMGval = string_n.rstrip()          # Remove \n and \r
        Float_EMG1 = float(string_EMGval)
        SaveData.append(Float_EMG1)
        print(Float_EMG1)

    


    with open(name_file, "w+",newline ="") as csvfile:     # Open csv file, If not create csv file
        w = csv.writer(csvfile)
        w.writerow('L')
        for i in range(10000):
            w.writerow([SaveData[i], empty[i]])




    print("Done {}".format(name_file))

    time.sleep(0.001)                  # wait  (sleep) 0.1 seconds

    # Stop counting time
    timer_stop = timeit.default_timer()   


    # multi ch
    # with open(name_file, "a+", newline="") as csvfile:    
    #     w = csv.writer(csvfile)
    #     for i in range(10000):         
    #         EMGval = ser.readline()                  # Read a byte string
    #         string_n = EMGval.decode("UTF-8")        # Decode byte string to Unicode
    #         string_EMGval = string_n.rstrip()          # Remove \n and \r
    #         print(EMGval)
    #         EMG1 = string_EMGval.split(" ")
    #         Float_EMG1 = float(EMG1)     # Convert string to float
    #         print(Float_EMG1)
    #         SaveData = [Float_EMG1,Float_EMG2,Float_EMG3,Float_EMG4]                    # Add to the end of data list
    #         with open(name_file, "a+",newline ="") as csvfile:
    #             w = csv.writer(csvfile)
    #             w.writerow(SaveData)
    #         w.writerow(SaveData)
    # print("Done {}".format(name_file))
    #     #time.sleep(0.001)                  # wait  (sleep) 0.1 seconds

    # # Stop counting time
    # timer_stop = timeit.default_timer()


    # EMGval = ser.readline()                  # Read a byte string
    # string_n = EMGval.decode("UTF-8")        # Decode byte string to Unicode
    # string_EMGval = string_n.rstrip()          # Remove \n and \r
    # Float_EMG1 = float(string_EMGval)
    # print(Float_EMG1)





