"""
生成随机车
"""
import random
random.seed(0)  # make tests reproducible
N = 20000  # number of time steps
thgh = 1. / 12
left = 1. / 20
right = 1./ 30
thgh=thgh*1
left=left*1
right=right*1
vehNr = {
    'n2': 1, 'e2': 1, 's2': 1, 'w2': 1, 'n1': 1, 'e1': 1, 's1': 1, 'w1': 1, 'w3': 1,'n3': 1, 'e3': 1, 's3': 1
}
with open("../cfg/cross_s.rou.xml", "w") as routes:
    print("""<routes>
    <vType id="HV" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="20" guiShape="passenger"/>
    <vType id="CV" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="20" guiShape="passenger" color="1,0,0"/>
    <route id="n3" edges="4i 1o" />
    <route id="n2" edges="4i 3o" />
    <route id="n1" edges="4i 2o" />
    <route id="e3" edges="2i 4o" />        
    <route id="e2" edges="2i 1o" />
    <route id="e1" edges="2i 3o" />
    <route id="s3" edges="3i 2o" />        
    <route id="s2" edges="3i 4o" />
    <route id="s1" edges="3i 1o" />
    <route id="w3" edges="1i 3o" />        
    <route id="w2" edges="1i 2o" />
    <route id="w1" edges="1i 4o" />""", file=routes)
    for i in range(N):
        direction = "w2"
        if random.random() < thgh:
            print('    <vehicle id="%s_%i" type="%s" route="%s" depart="%i" departspeed="%i"/>' %
                (direction, vehNr[direction],
                'CV' if random.random() <= 0.1 else 'HV',
                direction, i, 15), file=routes)
            vehNr[direction] += 1
            
    print("</routes>", file=routes)            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    