# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:50:10 2022

@author: Oxos
"""

import matplotlib.cm
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import numpy as np


st.title('Heating and Cooling of the Oceanic Lithosphere')
st.markdown(r'''In the lecture, we briefly talked about Lord Kelvin's approach to assess the age of the Earth by assuming the solution of a cooling semi-infinite half-space. Basically, he assumed that the specific heat flow at the surface is the result of a cooling process of the Earth. In the beginning, he assumed, Earth had the same temperature still present at its core.  
As we saw, his approach using the diffusion equation is flawed because he did not consider / did not know about concepts like radiogenic heat generation in the mantle or thermal convection (solid-state) in the mantle and he assumed heat transport by diffusion only:   

$$ \frac{\partial T}{\partial t} = \kappa \frac{\partial^2 T}{\partial x^2} $$

Nonetheless, this equation can also be used for determining the thermal structure of oceanig lithosphere. At a MOR (Mid Ocean Ridge), new hot crust is exposed to cold sea water. With time and seafloor spreading, rocks near the interface between seafloor and water cool down and lose their heat to the water. ''')
st.markdown('''In a semi-infinite half-space defined with y > 0, we can obtain the solution to the scenario above. At t=0, the lithosphere (half-space) has its original temperature $T_0$. Now, at the interface to the water (the *surface*), the temperature changes to a lower temperature $T_1$ at times $t > 0$. That causes a cooling from above, i.e. heat flowing upwards towards the surface. 

One can change the diffusion equation above for including the different temperatures by introducing a dimensionless time $\Theta$ using the concept of *similarity*. We first introduce a dimensionless time variable:

$$\Theta = \frac{T - T_0}{T_1 - T_0} $$  

And then re-write the diffusion equation with this variable:


$$ \frac{\partial \Theta}{\partial t} = \kappa \frac{\partial^2 \Theta}{\partial x^2} $$  

This step makes the boundary condistions for solving the equation significantly simpler. We now consider: 

- $\Theta(x,0) = 0$, 
- $\Theta(0,t) = 1$, 
- $\Theta(\infty,t) = 0$.''')

st.markdown(r'''(150*(10**6))As stated above, the half-space solution can be used to model the cooling of oceanic lithosphere. The bottom of said lithosphere, which is moving horizontally with a velocity $v$ above the mantle, can be seen as an isotherm. So, the lithosphere is a package, moving relative to the mantle, and bounded by the surface (seafloor), and an isotherm (around 1600 K). The lithosphere thickens over time, so the isothermal boundary at its bottom will be deeper the older the lithosphere is. Due to the seafloor spreading at a MOR, age is also a function of velocity. With a constant spreading-velocity, the lithosphere at a distance $x$ to the MOR can be considered Y years old.  

The cooling / thickening of the lithosphere can be described as an equation similar to the one in Kelvin's exercise:  
$$ t = x v^{-1} $$
$$\Theta = erf\bigg(\frac{y}{2\sqrt{\kappa x v^{-1}}}\bigg)$$  

<div class="alert alert-info"> Task: 
Re-arrange the equation above to come up with a formulation of `y` (the depth, thickness of the oceanic lithosphere).  
Use the obtained equation to calculate, what additional information do you need to look up? Plot the age dependent thickness of oceanic lithosphere (so `y`) for the following parameters (i.e. plot isotherms): 
    
T_0 = 277.15 K  

T_1 = 1600 K  

T = 600 K  

$\kappa$ = 1.2 mm² s$^{-1}$  

t = 0 Myr to 150 Myr in steps of 50 Myr  

</div>

The definition of the error function is

$$ erf(\eta) = \frac{2}{\sqrt{\pi}} \int^{\eta}_{0}e^{-\eta^{'2}} d \eta'$$

The approximation of the evaluation of the error function is given by (Taylor series):
$$ erf(\eta) \approx \frac{2}{\sqrt{\pi}} (\eta - \frac{\eta^3}{3}) $$

The error function can be used as a function in `scipy.special`.''')

data_bh='https://github.com/AlexanderCadmus/streamlit_test/borehole.txt'
#data_bh='C:/Users/Oxost/Documents/Streamlit_HiWi/geothermics_notebooks/streamlit_versions/borehole.txt'
data = pd.read_csv(data_bh, delim_whitespace=True)

dept=data['DEPT'].head(2000)
dt = data['DT'].head(2000)
edtc = data['GR_EDTC'].head(2000)
gtem = data['GTEM'].head(2000)
rwa = data['RWA'].head(2000)
print(rwa)

fig1=plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(8,16), sharey=True)
plt.gca().invert_yaxis()
axs[0,0].plot(dt, dept);
axs[0,0].set_title('dt')

axs[1,0].plot(edtc, dept);
axs[1,0].set_title('edtc')

axs[1,1].plot(gtem, dept);
axs[1,1].set_title('gtem')

axs[0,1].plot(rwa, dept);
axs[0,1].set_title('rwa')

corr_DT_GREDTC_guess = 0.2
corr_DT_RWA_guess = -1
corr_RWA_GREDTC = -0.2

col1, col2 = st.columns(2)
selection = col1.radio("Plot solution", ('Lithology', 'Probability', 'Entropy'))
col2.image(fig1)