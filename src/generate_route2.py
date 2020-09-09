"""
生成随机车
"""
import random
random.seed(2)  # make tests reproducible
arrival_rate = 300 

rate = arrival_rate / 3600 ## Total arrival rate for all vehicles(CVs and HVs)
penetration = 0.1 ## penetration rate for CV 


probability_1 = rate *(1-penetration) ## prbability for generating HVs
probability_2 = rate * penetration ## probability to generating CVs
with open("../cfg/cross_s.rou.xml", "w") as routes:
    print("""<routes>
    <vType id="HV" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="20" guiShape="passenger" />
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

    print('    <flow id="W_S_HV"  begin="0" end= "7200" route ="w2" probability="%f" type="HV">'%(probability_1),file=routes)
    print('</flow>',file=routes)
    print('    <flow id="W_S_CV" begin="0" end= "7200" route ="w2" probability="%f" type="CV">'%(probability_2),file=routes)
    print('</flow>',file=routes)
    print("</routes>", file=routes)     
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    