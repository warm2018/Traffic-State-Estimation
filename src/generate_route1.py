"""
生成随机车
"""
import random
random.seed(2)  # make tests reproducible

approach_rate_am = 500 
approach_rate_pm = 800

left_am = 0.5
straight_am = 0.3

left_pm = 0.5
straight_pm = 0.4



prob_l_am =  approach_rate_am * left_am / 3600
prob_s_am = approach_rate_am * straight_am / 3600


prob_l_pm =  approach_rate_pm * left_pm / 3600
prob_s_pm =  approach_rate_pm * straight_pm / 3600


with open("../cfg/cross_s.rou.xml", "w") as routes:
    print("""<routes>
    <vType id="mv" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="20" guiShape="passenger" color="1,0,0"/>
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

    print('    <flow id="s2_am"  begin="0" end= "14700" route ="s2" probability="%f" type="mv"/>'%(prob_s_am),file=routes)
    print('    <flow id="s1_am"  begin="0" end= "14700" route ="s1" probability="%f" type="mv"/>'%(prob_l_am),file=routes)
    print('    <flow id="s2_pm"  begin="14700" end= "29400" route ="s2" probability="%f" type="mv"/>'%(prob_s_pm),file=routes)
    print('    <flow id="s1_pm"  begin="14700" end= "29400" route ="s1" probability="%f" type="mv"/>'%(prob_l_pm),file=routes)
    print("</routes>", file=routes)                 

            
            
            
            
            
            
            
            
            
            
            
            
            
    