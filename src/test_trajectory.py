
import os
import sys
import optparse
import subprocess
import random

import traci
import xlsxwriter
import time 
#generator(1)

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

## Write data to Excel 

begin_step = 10
WRITE_EXCEL = True


def run():
    STEP = 0
    steplength = 1
    EXCEL = True
    row_count = 0
    if EXCEL == True:
        workbook = xlsxwriter.Workbook('../results/excel/trajectory1_day4.xlsx') 
        worksheet1 = workbook.add_worksheet('tra')
        worksheet1.write(row_count,0,"TimeStep")
        worksheet1.write(row_count,1,"VehID")
        worksheet1.write(row_count,2,"Distance")
        worksheet1.write(row_count,3,"VehType")
        worksheet1.write(row_count,4,"Speed")
        worksheet1.write(row_count,5,"LightState")
        worksheet1.write(row_count,6,"LaneState")         
        row_count +=1

    while STEP < 29400:
        traci.simulationStep()
        currentIDList = traci.vehicle.getIDList()
        STEP += steplength
        if STEP >= 300 and WRITE_EXCEL: 
            for VehicleID in currentIDList:
                travel_distance = traci.vehicle.getDistance(VehicleID)
                travel_speed = traci.vehicle.getSpeed(VehicleID)
                vehicle_type = traci.vehicle.getTypeID(VehicleID) ## write values into worksheet 
                traffic_light = traci.trafficlight.getRedYellowGreenState("0")
                worksheet1.write(row_count,0,STEP)
                worksheet1.write(row_count,1,VehicleID)
                worksheet1.write(row_count,2,travel_distance)                
                worksheet1.write(row_count,3,vehicle_type)
                worksheet1.write(row_count,4,travel_speed)
                worksheet1.write(row_count,5,traffic_light)
                worksheet1.write(row_count,6,traffic_light[-5]) 
                row_count +=1
            print('STEP',STEP)
    workbook.close()
    traci.close()
    sys.stdout.flush()  


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options
    # this is the main entry point of this script


if __name__ == "__main__":
    options = get_options()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "../cfg/cross.sumocfg",
                             "--tripinfo-output", "../results/tripinfo.xml",
                             "--step-length","1",
                             "--seed","3",
                             "--log","logtext.txt",])

    run()