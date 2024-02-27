# -------------------------------------------------------
# NUMERICAL SOLUTION NEWTON EQUATIONS OF MOTION
# EULER METHOD
# April 2020 Python3 version
# 
# Very simple example with Harmonic oscillator
# Units (nanoSI): nm, ns, ng, ...
# By Jordi Faraudo 2018
# -------------------------------------------------------

# Here we import the mathematical library and the plots library
import numpy as np
import matplotlib.pyplot as plt

# Particle submitted to harmonic force
# mass
m = 1.0
# Period in ns
T = 1.0
#frequency
w = 2.0*np.pi/T
#Force constant
k=m*w*w

#Initial condition (position and velocity)
x0=0.1
v0=0.0
# initial Energy
E0=(m/2.0)*v0*v0+(k/2)*x0*x0

#Show data of the program

print('\n--------------------------------------------------------')
print('SIMPLE MD SIMULATION OF A SINGLE PARTICLE IN HARMONIC TRAP')
print('----------------------------------------------------------')
print('Force constant:',k,' N/m')
print('Particle of mass:',m,' ng')
print('Period according to analytical solution of harmonic oscillator:',T,' ns')

### I want to see the effect of having variable time steps as a function of the speed and acceleration at a given time ###
### It might reduce the energetic symmetry of the system because i believe it is produced by higher speeds that lead to an erratic trajectory ###

### After trying it, it does lower that erratic effect, but it does not nullify it thus it is no solution. ###
### I realise that this is actually just a way of having dynamic resolution in our calculation (having a higher number of steps only where it matters). ###

# input time step
dt = 0.01
# Final time
tt = 10
# step distance threshold for variable dt calculation
dx = 0.02

# create empty array starting at zero with time, position, velocity
x = [0]
v = [0]
t = [0]

#Initial conditions
x[0] = x0
v[0] = v0
last_t = 0
i = 0

# Time evolution
print('\n Calculating time evolution...')

#Since we do not know how many iterations the calculation will take, use while loop.
while last_t < tt:
    print(i)
    deltat = dt
    #Calculate Force over the particle
    f = -k*x[i]
    #Calculate acceleration from 2nd Law
    a = f/m 
    # New velocity after time dt
    v.append(v[i]+a*deltat)
    #Average velocity from t to t+dt
    v_av= (v[i]+v[i+1])/2.0
    # New position
    x.append(x[i]+v_av*deltat)

    # Check if the step was too long. It it was, perform quadratic equation calculation of delta time and re-do the step
    if (x[i+1] - x[i]) > dx:
        dt1 = (-v[i] - np.sqrt(v[i]**2 - 4*0.5*a*(-v[i]/abs(v[i]))*dx))/a
        dt2 = (-v[i] + np.sqrt(v[i]**2 - 4*0.5*a*(-v[i]/abs(v[i]))*dx))/a
        if dt1 >= 0 and dt2 >= 0:
            deltat = np.asarray([dt1, dt2]).min()
        elif dt1 > 0 and dt2 <= 0:
            deltat = dt1
        elif dt2 > 0 and dt1 <= 0:
            deltat = dt2
        else:
            deltat = dt
            print('There has been an error somewhere; negative dt found')
        v[-1] = (v[i]+a*deltat)
        v_av = (v[i]+v[i+1])/2.0
        x[-1] = (x[i]+v_av*deltat)
    print(deltat)
    
    #Update time
    t.append(t[i]+deltat)
    last_t = t[-1]
    i += 1
    
# plot output
print('Calculation finished. Showing plot with results')

#
# Create a plot with x(t) and v(t)
# 
#plt.plot(t,x, 'ro', t, v, 'bv')
plt.figure(1)

plt.subplot(211)
plt.plot(t,x)
plt.ylabel('x (nm)')

plt.subplot(212)
plt.plot(t,v)
plt.ylabel(' v (nm/ns)')
plt.xlabel('time (ns)')

#create axis
#plt.axhline(0, color='black')
#plt.axvline(0, color='black')
#Show plot in screen
plt.show()
#Show plot of phase space
plt.plot(x,v,'k')
plt.xlabel('x (nm)')
plt.ylabel('v (nm/ns)')
#Show plot in screen
plt.show()
#Also plot energy
#energy at all steps
E=(m/2.0)*np.asarray(v)*np.asarray(v)+(k/2)*np.asarray(x)*np.asarray(x)
#Relative value (E/E0)
RE=E/E0
plt.plot(t,RE,'k')
plt.xlabel('time (ns)')
plt.ylabel('E/E0')
#Show plot in screen
plt.show()