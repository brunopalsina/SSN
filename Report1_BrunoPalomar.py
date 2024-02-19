import numpy as n
import matplotlib.pyplot as plt
import random

airdrag = 0.3
puck_weight = 0.170
ice_puck_friction = 0.06
global_figure_num = 1

def random_color():
    
    return (random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255)

class Puck():

    def __init__(self, v0, m = puck_weight, f = ice_puck_friction, d = airdrag): #mass, friction, drag, initial velocity

        self.v0 = v0
        self.m = m
        self.f = f
        self.d = d
        self.velocity()

    def velocity(self):
        
        print('\nAbout to shoot a virtual ice-hockey puck... Bam!\n')
        self.t = -(self.m/self.d)*n.log(self.m*self.f/(self.d*self.v0 + self.f*self.m))
        #print('The puck stops moving ' + str(round(t, 3)) + ' seconds after being shot.\n\nYou can simulate the shot with mass, drag, and friction parametres of your choice!')
        self.time = [0]
        self.vel = [self.v0]
        self.pos = [0]
        self.accel = []
        i = 1
        ticks = 20

        while i/ticks <= 1:
            self.time.append(i*self.t/ticks)
            self.vel.append(((self.d*self.v0 + self.f*self.m)/self.d)*n.exp(-self.d*(i*self.t/ticks)/self.m) - self.m*self.f/self.d)
            self.pos.append(self.pos[i-1] + (self.vel[i-1]+self.vel[i])*(self.time[i]-self.time[i-1]))
            self.accel.append((self.vel[i]-self.vel[i-1])*(self.time[i]-self.time[i-1]))
            i = i + 1

    def plot_shot(self):
        
        global global_figure_num
        self.plotnum = global_figure_num
        fig = plt.figure(self.plotnum)
        ax = fig.subplots()
        ax.plot(self.time, self.vel, color = random_color(), label = 'Velocity(t)')
        ax.plot(self.time, self.pos, color = random_color(), label = 'Position(t)')
        ax.plot(self.time[1::], self.accel, color = random_color(), label = 'Acceleration(t)')
        ax.set_title('Shot simulation of a puck with Vinitial = ' + str(self.v0) + ' m/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('f(t)')
        ax.minorticks_on()
        ax.tick_params("both", direction="in",which="both", right=True, top=True)
        ax.grid(True)
        ax.legend()
        global_figure_num = self.plotnum + 1
        #plt.show()

def compare_shots(puck_list):

    v0 = []
    stop_t = []
    distance = []

    for puck in puck_list:

        v0.append(puck.v0)
        stop_t.append(puck.t)
        distance.append(puck.pos[-1])

    fig = plt.figure(global_figure_num)
    ax = fig.subplots()
    ax.plot(v0, stop_t, '--o',color = random_color(), label = 'Sliding time (Vinitial)')
    ax.plot(v0, distance, '--o', color = random_color(), label = 'Traveled distance (Vinitial)')
    ax.set_title('Puck travelled distance and time as a function of initial velocity')
    ax.set_xlabel('Initial velocity (m/s)')
    ax.set_ylabel('f(Vinitial)')
    ax.minorticks_on()
    ax.tick_params("both", direction="in",which="both", right=True, top=True)
    ax.grid(True)
    ax.legend()

v_list = [1, 10, 20, 40, 60, 80]
puck1, puck10, puck20, puck40, puck60, puck80 = [Puck(i) for i in v_list]
puck_list = [puck1, puck10, puck20, puck40, puck60, puck80]

puck60.plot_shot()
puck20.plot_shot()
compare_shots(puck_list)

#Use plt.show after all plots have been generated to show them at once:
plt.show()