### v2 save has sidebar


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:58:38 2022

@author: Oxos
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
###Introduction###
st.title('MCMC from scratch')
st.markdown(r'''A simple example of Markov Chain Monte Carlo (MCMC) for sampling from the posterior distribution - 

adapted from: https://towardsdatascience.com/introduction-to-mcmc-1c8e3ea88cc9''')
with st.sidebar:
    st.header('''Part 2 Options: 1-D Example''')
    st.markdown('''Display number of samples picked from population''')
    n_plot = st.select_slider('Sampled Poplation Data', options=[0, 1, 2, 3, 5, 8, 15, 25, 40, 60, 100, 150, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000])
    st.markdown('''Factor by which proposal values are modified''')
    var = st.slider('Var', min_value=1, max_value=10, value=1, step=1, format=None, key=3, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    st.markdown('''Sequential Proposals in Metropolis Sampling''')
    #n_plot_samples = st.slider('n_plot_samples', min_value=1, max_value=100, value=1, step=1, format=None, key=4, help=None, on_change=None, args=None, kwargs=None, disabled=False)        
    n_plot_samples=st.number_input('Proposal Step',min_value=1, max_value=100,step=1)
    st.header('''Part 3 Options: 2-D Example''')
    st.markdown('''Bimodal Distribution Characteristics''')
    st.markdown('''First Mode Settings''')
    mfx=st.slider('First mode x position',value=2.0,min_value=-2.0, max_value=3.5,step=0.1)
    mfy=st.slider('First mode y position',value=2.0,min_value=-2.0, max_value=3.5,step=0.1)
    st.markdown('''Second Mode Settings''')
    msx=st.slider('Second mode x position',value=-1.0,min_value=-2.0, max_value=3.5,step=0.1)
    msy=st.slider('Second mode y position',value=-1.0,min_value=-2.0, max_value=3.5,step=0.1)
    #s_plot = st.slider('Number of Samples', min_value=1, max_value=5000, value=10, step=100, format=None, key=1, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    st.markdown('''Samples to be picked and plotted''')
    s_plot=st.number_input('Number of Samples Plotted',min_value=51, max_value=10000,step=1)
    #s_plot = st.slider('Number of Samples', min_value=1, max_value=5000, value=10, step=100, format=None, key=1, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    
###########################################################################
###################-----Part 1-----########################################
###########################################################################
###########################################################################
###################-----Define Sampling Procedure-----##################### 
###########################################################################

st.markdown('''___________________________________________''')
st.header('''Part 1a: Monte Carlo Simulations''')
st.markdown('''___________________________________________''')
st.markdown('''Often complex systems with many degrees of freedom or high uncertainty are difficult 
to directly model. It can be far more efficient to simply sample a population to derive its characteristics 
and then predict its behavior. This is called a **Monte Carlo simulation**. In other words, the use of 
**ordinary Monte Carlo** simulations (OMC or simply MC) numerically solves otherwise analytical problems. 
How this is applied is often up to the author whose methods of population sampling may vary based on 
particular interest or concerns.''')

st.markdown('''___________________________________________''')
st.header('''Part 1b: Markov Chains''')
st.markdown('''___________________________________________''')
st.markdown('''A popular method of sampling is generally referred to as a **Markov chain**. This procedure is similar 
to a random walk in that each step is determined by the previous outcome 
and an addition of a random factor.  This, in turn, can be linked to moving around a board game 
where each new position is a product of the previous position and a new roll of the dice. When 
combined with a OMC simulation it is capable of solving some potential sampling issues in higher 
dimensional spaces observed with less sophisticated techniques.''')

st.markdown('''___________________________________________''')
st.header('''Part 1c: Defining the Metropolis Procedure''')
st.markdown('''___________________________________________''')
st.markdown('''The Metropolis sampling procedure (formally the Metropolis–Hastings algorithm) is a 
designed to randomly draw samples from a population of known density.  This is accomplished by using 
a function, to generate sampled points, that is proportional to the density of the population of interest. 
After initialization of the system, sampled points are determined by previously sampled data points with the 
addition of a random value.  In this exercise, a proposal of a new sampled point is determined by the summation 
of the last sampled point and second value determined by a gaussian distribution. Each 
new sample point called a proposal and undergoes an acceptance process defined by the author. 
If rejected, a new value is computed. If accepted, the process continues.''')

with st.expander(r'''See Code: Metropolis Sampling'''):
    metroplis_code='''def metropolis(pi, dims, n_samples, burn_in=0.1, var=1):
        # start with random initial position.  Here a gaussian distribution is used.
        theta_ = np.random.randn(dims)*var
        samples = np.empty((n_samples, dims))
        # sampling loop
        for i in range(n_samples):
        # while len(samples) < n_samples:
            # proposal step
            theta = theta_ + np.random.randn(dims)*var
            
            # ratio of probabilities between proposed and current step
            ratio = pi(theta)/pi(theta_)

            # check acceptance - note: if ratio > 1, by def. larger than rv and accepted:
            if np.random.rand(1) < ratio:
                sample = theta
                theta_ = theta
                samples[i,:] = sample
                # samples.append(sample)

            # reject: remain at original state and add to trace:
            else:
                sample = theta_
                samples[i,:] = sample

        # remove burn-in phase (to do)
        return samples[int(n_samples*burn_in):,:]'''
    st.code(metroplis_code, language='python')

@st.cache
def metropolis(pi, dims, n_samples, burn_in=0.1, var=1):
    # start with random initial position.  Here a gaussian distribution is used.
    theta_ = np.random.randn(dims)*var
    samples = np.empty((n_samples, dims))
    # sampling loop
    for i in range(n_samples):
    # while len(samples) < n_samples:
        # proposal step
        theta = theta_ + np.random.randn(dims)*var
        
        # ratio of probabilities between proposed and current step
        ratio = pi(theta)/pi(theta_)

        # check acceptance - note: if ratio > 1, by def. larger than rv and accepted:
        if np.random.rand(1) < ratio:
            sample = theta
            theta_ = theta
            samples[i,:] = sample
            # samples.append(sample)

        # reject: remain at original state and add to trace:
        else:
            sample = theta_
            samples[i,:] = sample

    # remove burn-in phase (to do)
    return samples[int(n_samples*burn_in):,:]

###########################################################################
###################-----Part 2-----########################################
###########################################################################
###########################################################################
###############-----Simple 1-D Distribution-----########################### 
###########################################################################

st.markdown('''___________________________________________''')
st.header('''Part 2a: 1-D Example''')
st.markdown('''___________________________________________''')
st.markdown('''Generate a normal 1-D population with a mean=2 and standard deviation=2''') 

with st.expander(r'''See Code'''):
    oned_code='''from scipy.stats import norm
    pdf_1D = norm(2, 2);
    theta = np.arange(-7,11,0.1);
    fig_oneD_data, axs = plt.subplots(figsize=(20,8))
    fig_oneD_data.suptitle('1-D Population Distribution', fontsize=22)
    axs.plot(theta, pdf_1D.pdf(theta))
    axs.set_xlabel('X Value',fontsize=14)
    axs.set_ylabel('Frequency',fontsize=14)
    st.pyplot(fig_oneD_data)'''
    st.code(oned_code, language='python')

from scipy.stats import norm
pdf_1D = norm(2, 2);
theta = np.arange(-7,11,0.1);
fig_oneD_data, axs = plt.subplots(figsize=(20,8))
fig_oneD_data.suptitle('1-D Population Distribution', fontsize=22)
axs.plot(theta, pdf_1D.pdf(theta),c='orange')
axs.set_xlabel('X Value',fontsize=14)
axs.set_ylabel('Frequency',fontsize=14)
st.pyplot(fig_oneD_data)

###########################################################################
############-----1-D Sampling Procedure with Plotted Data-----############# 
###########################################################################  
st.markdown('''___________________________________________''')
st.header('''Part 2b: 1-D Metropolis Sample''')
st.markdown('''___________________________________________''')

st.markdown('''Here the group of 10,000 samples were generated using the Metropolis 
sampling technique developed above. Use the slider to plot the number of samples desired. 
Note how the distribution more resembles the population with increased sampling.''')

with st.expander(r'''See Code'''):
    oned_code='''samples = metropolis(pdf_1D.pdf, 1, 10_000, 0., 1)

    #1D sample data display
    fig_all_data, axs = plt.subplots(2,1, figsize=(20,12),gridspec_kw={'height_ratios': [1, 4]})
    fig_all_data.suptitle('1-D Sample Data', fontsize=24)
    axs[0].plot(samples[:n_plot,0])
    axs[0].set_title('Sample Traces',fontsize=24)
    axs[0].set_xlabel('Sample Number',fontsize=20)
    axs[0].set_ylabel('X Value',fontsize=20)
    axs[1].hist(samples[:n_plot,0],100)
    axs[1].set_title('Sample Position Frequency',fontsize=24)
    axs[1].set_ylabel('Frequency',fontsize=20)
    axs[1].set_xlabel('X Value',fontsize=20)
    axs[1].set_xlim([-7, 10])
    fig_all_data.tight_layout()'''
    st.code(oned_code, language='python')
    
samples = metropolis(pdf_1D.pdf, 1, 10_000, 0., 1)
samples_var_two = metropolis(pdf_1D.pdf, 1, 10_000, 0., 2)
samples_var_three = metropolis(pdf_1D.pdf, 1, 10_000, 0., 4)
samples_var_four = metropolis(pdf_1D.pdf, 1, 10_000, 0., 8)
#1D sample data display
fig_all_data, axs = plt.subplots(2,1, figsize=(20,12),gridspec_kw={'height_ratios': [1, 4]})
fig_all_data.suptitle('1-D Sample Data', fontsize=24)
axs[0].plot(samples[:n_plot,0])
axs[0].set_title('Sample Traces',fontsize=24)
axs[0].set_xlabel('Sample Number',fontsize=20)
axs[0].set_ylabel('X Value',fontsize=20)
axs[1].hist(samples[:n_plot,0],100)
axs[1].set_title('Sample Position Frequency',fontsize=24)
axs[1].set_ylabel('Frequency',fontsize=20)
axs[1].set_xlabel('X Value',fontsize=20)
axs[1].set_xlim([-7, 10])
fig_all_data.tight_layout()
st.pyplot(fig_all_data)

with st.expander(r'''See Code: Density Plot'''):
        oned_code='''fig_oneD_data_sampled, axs = plt.subplots(figsize=(20,8))
        fig_oneD_data_sampled.suptitle('1-D Sample Data with Density of Sampled Data', fontsize=22)
        axs.hist(samples[:n_plot,0],100,density=True,label="Density of Sampled Data")
        axs.plot(theta, pdf_1D.pdf(theta),linewidth=3, label="Density of Population Data")
        axs.set_title('Data Distribution',fontsize=24)
        axs.set_xlabel('X Value',fontsize=20)
        axs.set_ylabel('Denisty',fontsize=20)
        axs.set_xlim([-7, 10])
        axs.legend(loc='upper right', fontsize=20)'''
        st.code(oned_code, language='python')

fig_oneD_data_sampled, axs = plt.subplots(figsize=(20,8))
fig_oneD_data_sampled.suptitle('1-D Population Density \n with Sampled Data Density', fontsize=22)
axs.hist(samples[:n_plot,0],100,density=True,label="Density of Sampled Data")
axs.plot(theta, pdf_1D.pdf(theta),linewidth=3, label="Density of Population Data")
#axs.set_title('Data Distribution',fontsize=24)
axs.set_xlabel('X Value',fontsize=20)
axs.set_ylabel('Denisty',fontsize=20)
axs.set_xlim([-7, 10])
axs.legend(loc='upper right', fontsize=20)
st.pyplot(fig_oneD_data_sampled)

#for 2D sample coordinates
# fig, axs = plt.subplots(2,2, figsize=(20,12))
# fig.suptitle('All Sample Data', fontsize=22)
# axs[0,0].plot(samples[:,0])
# axs[0,0].set_title('Sample x Traces',fontsize=18)
# axs[0,0].set_xlabel('Sample Number',fontsize=14)
# axs[0,0].set_ylabel('x Position',fontsize=14)
# axs[1,0].hist(samples[:,0],100)
# axs[1,0].set_title('Sample x Coordinate Frequency',fontsize=18)
# axs[1,0].set_ylabel('Frequency',fontsize=14)
# axs[1,0].set_xlabel('x Position',fontsize=14)
# axs[0,1].plot(samples[:,1])
# axs[0,1].set_title('Sample y Traces',fontsize=18)
# axs[0,1].set_xlabel('Sample Number',fontsize=14)
# axs[0,1].set_ylabel('y Position',fontsize=14)
# axs[1,1].hist(samples[:,1],100)
# axs[1,1].set_title('Sample y Coordinate Frequency',fontsize=18)
# axs[1,1].set_xlabel('y Position',fontsize=14)
# fig.tight_layout()

st.markdown('''Plot 1-D data in simple curve showing sampled regions highlighted by read ticks.''')
with st.expander(r'''See Code'''):
    oned_code='''def plot_1D_with_samples(n_plot_samples):
        fig, axs = plt.subplots(figsize=(12,8));
        axs.set_title('Population Distribution')
        axs.set_xlabel('Position')
        axs.set_ylabel('Frequency')
        theta = np.arange(-7,11,0.1)
        axs.plot(theta, pdf_1D.pdf(theta))
        axs.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.2)
        return fig'''
    st.code(oned_code, language='python')

def plot_1D_with_samples(n_plot_samples):
    fig, axs = plt.subplots(figsize=(12,6));
    axs.set_title('Population Distribution \n with sampled points using different variances')
    axs.set_xlabel('X Value')
    axs.set_ylabel('Density')
    theta = np.arange(-7,11,0.1)
    axs.plot(theta, pdf_1D.pdf(theta),c='orange',label='Population Distribution')
    axs.vlines(samples_var_four[:n_plot_samples],0,0.05,'b', alpha=0.4,label='Variance of 8')
    axs.vlines(samples_var_three[:n_plot_samples],0,0.04,'g', alpha=0.4,label='Variance of 4')
    axs.vlines(samples_var_two[:n_plot_samples],0,0.03,'y', alpha=0.4,label='Variance of 2')
    axs.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.4,label='Variance of 1')
    axs.legend()
    return fig

st.pyplot(plot_1D_with_samples(n_plot))
###########################################################################
###################-----Stepwise Sampling Procedure-----################### 
###########################################################################  

st.markdown('''Here we can define a stepwise Metropolis sampling procedure using proposals for sequentially 
generated sample points. Each proposal may be rejected or accepted if it satisfies the desired criterion. 
Here that is that ratio of probabilities between proposed and current step is larger than a 
randomly generated value between [0,1].''')

with st.expander('See Code: Metropolis Sampling with Proposals'):
    sws_code='''def metropolis_with_proposals(pi, dims, n_samples, burn_in=0.1, var=1):
    # start with random initial position
    theta_ = np.random.randn(dims)*var
    samples = np.empty((n_samples, dims))
    proposals = np.empty((n_samples, dims))
    accepted = np.empty(n_samples)
    # sampling loop
    for i in range(n_samples):
    # while len(samples) < n_samples:
        # proposal step
        proposal = np.random.randn(dims)*var
        theta = theta_ + proposal

        # store proposals for later vis
        proposals[i,:] = proposal
        
        # ratio of probabilities between proposed and current step
        ratio = pi(theta)/pi(theta_)

        # check acceptance - note: if ratio > 1, by def. larger than rv and accepted:
        if np.random.rand(1) < ratio:
            sample = theta
            theta_ = theta
            samples[i,:] = sample
            accepted[i] = True
            # samples.append(sample)

        # reject: remain at original state and add to trace:
        else:
            sample = theta_
            samples[i,:] = sample
            accepted[i] = False

    # remove burn-in phase (to do)
    return samples[int(n_samples*burn_in):,:], proposals, accepted'''
    st.code(sws_code, language='python')

@st.cache
def metropolis_with_proposals(pi, dims, n_samples, burn_in=0.1, var=1):
    # start with random initial position
    theta_ = np.random.randn(dims)*var
    samples = np.empty((n_samples, dims))
    proposals = np.empty((n_samples, dims))
    accepted = np.empty(n_samples)
    # sampling loop
    for i in range(n_samples):
    # while len(samples) < n_samples:
        # proposal step
        proposal = np.random.randn(dims)*var
        theta = theta_ + proposal

        # store proposals for later vis
        proposals[i,:] = proposal
        
        # ratio of probabilities between proposed and current step
        ratio = pi(theta)/pi(theta_)

        # check acceptance - note: if ratio > 1, by def. larger than rv and accepted:
        if np.random.rand(1) < ratio:
            sample = theta
            theta_ = theta
            samples[i,:] = sample
            accepted[i] = True
            # samples.append(sample)

        # reject: remain at original state and add to trace:
        else:
            sample = theta_
            samples[i,:] = sample
            accepted[i] = False

    # remove burn-in phase (to do)
    return samples[int(n_samples*burn_in):,:], proposals, accepted

samples, proposals, accepted = metropolis_with_proposals(pdf_1D.pdf, 1, 10_000, 0., 1)

###########################################################################
###############-----Plotting of Stepwise Sampled Data-----################# 
###########################################################################

st.markdown('''Here, proposals are plotted against the population. For each step the current value 
is displayed as a black line.  A normal distribution is displayed in yellow around the current 
step showing the probability distribution of the next step or as we say the proposal. 
If the proposal is accepted it will be displayed as a green line. 
If it is rejected it will be displayed as a red line.''')

with st.expander('See Code'):
    sws_code='''def plot_1D_with_samples_and_proposal(n_plot_samples):
        # create plot with sampled locations and proposal step for current iteration
        fig, axs = plt.subplots(figsize=(12,8))
        theta = np.arange(-7,11,0.1)
    axs.set_title('Population Distribution with Proposal Steps')
        axs.set_xlabel('Position')
        axs.set_ylabel('Frequency')
        axs.plot(theta, pdf_1D.pdf(theta))
        axs.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.2)
        # proposal pdf
        proposal_pdf = norm(samples[n_plot_samples], var)
        axs.plot(theta, 0.3*proposal_pdf.pdf(theta))
        prop_pos = samples[n_plot_samples] + proposals[n_plot_samples]

        axs.plot(samples[n_plot_samples], pdf_1D.pdf(samples[n_plot_samples]), 'ko')
        axs.vlines(samples[n_plot_samples], 0, pdf_1D.pdf(samples[n_plot_samples]), 'k')
        
        if accepted[n_plot_samples]:
            axs.plot(prop_pos, pdf_1D.pdf(prop_pos), 'go')
            axs.vlines(prop_pos, 0, pdf_1D.pdf(prop_pos), 'g')
        else:
            axs.plot(prop_pos, pdf_1D.pdf(prop_pos), 'ro')
            axs.vlines(prop_pos, 0, pdf_1D.pdf(prop_pos), 'r')
            # plt.plot(prop_pos, 0.3*proposal_pdf.pdf(prop_pos), 'ro')
            # plt.vlines(prop_pos, 0, 0.3*proposal_pdf.pdf(prop_pos), 'r')
        return fig'''
    st.code(sws_code, language='python')

def plot_1D_with_samples_and_proposal(n_plot_samples):
    # create plot with sampled locations and proposal step for current iteration
    fig, axs = plt.subplots(figsize=(12,8))
    theta = np.arange(-7,11,0.1)
    axs.set_title('Population Distribution \n with Proposal Steps')
    axs.set_xlabel('Position')
    axs.set_ylabel('Frequency')
    axs.plot(theta, pdf_1D.pdf(theta), c='orange',label='population Distribution')
    axs.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.2,label='Sampled Locations')
    # proposal pdf
    proposal_pdf = norm(samples[n_plot_samples], var)
    axs.plot(theta, 0.3*proposal_pdf.pdf(theta))
    prop_pos = samples[n_plot_samples] + proposals[n_plot_samples]

    axs.plot(samples[n_plot_samples], pdf_1D.pdf(samples[n_plot_samples]), 'ko')
    axs.vlines(samples[n_plot_samples], 0, pdf_1D.pdf(samples[n_plot_samples]), 'k')
    
    if accepted[n_plot_samples]:
        axs.plot(prop_pos, pdf_1D.pdf(prop_pos), 'go')
        axs.vlines(prop_pos, 0, pdf_1D.pdf(prop_pos), 'g')
    else:
        axs.plot(prop_pos, pdf_1D.pdf(prop_pos), 'ro')
        axs.vlines(prop_pos, 0, pdf_1D.pdf(prop_pos), 'r')
        # plt.plot(prop_pos, 0.3*proposal_pdf.pdf(prop_pos), 'ro')
        # plt.vlines(prop_pos, 0, 0.3*proposal_pdf.pdf(prop_pos), 'r')
    return fig
        

st.pyplot(plot_1D_with_samples_and_proposal(n_plot_samples))

###########################################################################
###############-----Part 3-----############################################
###########################################################################
###########################################################################
############-----Generate & Plot Bimodal Gaussian Model-----############### 
###########################################################################

st.markdown('''___________________________________________''')
st.header('''Part 3: Multimodal Example''')
st.markdown('''___________________________________________''')    
st.markdown('''Here a Bimodal Gaussian distribution is generated.''')
with st.expander(r'''See Code'''):
    mgm_code='''from scipy.stats import multivariate_normal
    def make_pdf(mean1, mean2, cov1, cov2):
        pdf1 = multivariate_normal(mean1, cov1)
        pdf2 = multivariate_normal(mean2, cov2)
        def pdf(x):
            return pdf1.pdf(x) + pdf2.pdf(x)
        return pdf

    mean1 = [3, 3]
    mean2 = [-1, -1]
    cov1 = np.array([[1,0.5],[0.5,1]], dtype=float)
    cov2 = np.array([[1,-0.3],[-0.3,1]], dtype=float)
    pdf1 = multivariate_normal(mean1, cov1)
    pdf2 = multivariate_normal(mean2, cov2)

    mgm_plt=plt.figure(figsize=(10,10))
    x, y = np.mgrid[-4:6:.01, -4:6:.01]
    pos = np.dstack((x, y))
    plt.title('Population Density Map',fontsize=18)
    plt.xlabel('X Position',fontsize=10)
    plt.ylabel('Y Position',fontsize=10)
    plt.contour(x, y, pdf1.pdf(pos) + pdf2.pdf(pos))
    plt.colorbar()
    plt.savefig("multigauss.png")

    pdf = make_pdf(mean1, mean2, cov1, cov2)
    
    samples = metropolis(pdf, 2, 10_000, 0., 1)'''
    st.code(mgm_code, language='python')

from scipy.stats import multivariate_normal

def make_pdf(mean1, mean2, cov1, cov2):
    pdf1 = multivariate_normal(mean1, cov1)
    pdf2 = multivariate_normal(mean2, cov2)
    def pdf(x):
        return pdf1.pdf(x) + pdf2.pdf(x)
    return pdf

mean1 = [mfx, mfy] #first mean x (mfx) and first mean y (mfy) defined in sidebar
mean2 = [msx, msy] #second mean x (msx) and first mean y (msy) defined in sidebar
cov1 = np.array([[1,0.5],[0.5,1]], dtype=float)
cov2 = np.array([[1,-0.3],[-0.3,1]], dtype=float)
pdf1 = multivariate_normal(mean1, cov1)
pdf2 = multivariate_normal(mean2, cov2)

mgm_plt=plt.figure(figsize=(10,10))
x, y = np.mgrid[-4:6:.01, -4:6:.01]
pos = np.dstack((x, y))
plt.title('Population Density Map',fontsize=18)
plt.xlabel('X Position',fontsize=10)
plt.ylabel('Y Position',fontsize=10)
plt.contour(x, y, pdf1.pdf(pos) + pdf2.pdf(pos))
plt.colorbar()

pdf = make_pdf(mean1, mean2, cov1, cov2)

samples = metropolis(pdf, 2, 10_000, 0., 1)

st.pyplot(mgm_plt)

###########################################################################
###########-----Implement Sampling Procedure & Plot Data-----############## 
###########################################################################

st.markdown('''Following the same Metropolis sampling procedure as before we can trace the generation of each 
sample point. Use the Part 3 options to slowly increase the number of samples. Sequential points are connected 
by a blue line.''')
with st.expander(r'''See Code'''):
    trace_code='''    
        def plot_samples(s_plot=1000):
        fig, axs=plt.subplots(figsize=(10,10));
        axs.set_title('Metropolis Sampling',fontsize=18)
        axs.set_ylabel('y Position',fontsize=10)
        axs.set_xlabel('x Position',fontsize=10)
        Met_Pop=axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.6) #, linewidth=0);
        axs.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k');
        axs.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5);
        axs.set_xlim([-4,6]);
        axs.set_ylim([-4,6]);
        cbar = plt.colorbar(Met_Pop)
        cbar.set_label('Population Density', rotation=270, fontsize=10,labelpad=15)
        return fig
    
    s_plot_max = 1000
    fig = plt.figure(figsize=(14,6));
    fig.suptitle('Sample Coordinate Traces', fontsize=16)
    ax1 = fig.add_subplot(211);
    ax2 = fig.add_subplot(212);
    ax1.plot(samples[:s_plot_max,0]);
    ax1.set_ylabel('x Position');
    ax2.plot(samples[:s_plot_max,1]);
    ax2.set_xlabel('Sample Number');
    ax2.set_ylabel('y Position');'''
    st.code(trace_code, language='python')

def plot_samples(s_plot=1000):
    fig, axs=plt.subplots(figsize=(10,10));
    axs.set_title('Metropolis Sampling',fontsize=18)
    axs.set_ylabel('Y Position',fontsize=10)
    axs.set_xlabel('X Position',fontsize=10)
    Met_Pop=axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.6) #, linewidth=0);
    axs.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k');
    axs.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5);
    axs.set_xlim([-4,6]);
    axs.set_ylim([-4,6]);
    cbar = plt.colorbar(Met_Pop)
    cbar.set_label('Population Density', rotation=270, fontsize=10,labelpad=15)
    return fig
    # plt.savefig("multigauss_sampling.png")

st.pyplot(plot_samples(s_plot))   

###########################################################################
###############-----Examine Trace Data-----################################ 
###########################################################################  
st.markdown('''As the sample amount increases, we can investigate traces to gain insight to how the sampling 
progresses as well as how the sample distribution more and more resembles that of the population.''')
with st.expander(r'''See Code: plots'''):
    trace_code='''#for 2D sample coordinates
    fig, axs = plt.subplots(2,2, figsize=(20,12))
    fig.suptitle('All Sample Data', fontsize=22)
    axs[0,0].plot(samples[:s_plot,0])
    axs[0,0].set_title('Sample x Traces',fontsize=18)
    axs[0,0].set_xlabel('Sample Number',fontsize=14)
    axs[0,0].set_ylabel('x Position',fontsize=14)
    axs[1,0].hist(samples[:s_plot,0],100)
    axs[1,0].set_title('Sample x Coordinate Frequency',fontsize=18)
    axs[1,0].set_ylabel('Frequency',fontsize=14)
    axs[1,0].set_xlabel('x Position',fontsize=14)
    axs[0,1].plot(samples[:s_plot,1])
    axs[0,1].set_title('Sample y Traces',fontsize=18)
    axs[0,1].set_xlabel('Sample Number',fontsize=14)
    axs[0,1].set_ylabel('y Position',fontsize=14)
    axs[1,1].hist(samples[:s_plot,1],100)
    axs[1,1].set_title('Sample y Coordinate Frequency',fontsize=18)
    axs[1,1].set_xlabel('y Position',fontsize=14)
    fig.tight_layout()'''
    st.code(trace_code, language='python')
    
# number of samples to plot from beginning of trace     

s_plot_max = 1000
#for 2D sample coordinates
fig, axs = plt.subplots(2,2, figsize=(20,12))
fig.suptitle('All Sample Data', fontsize=22)
axs[0,0].plot(samples[:s_plot,0])
axs[0,0].set_title('Sample x Traces',fontsize=18)
axs[0,0].set_xlabel('Sample Number',fontsize=14)
axs[0,0].set_ylabel('x Position',fontsize=14)
axs[1,0].hist(samples[:s_plot,0],100)
axs[1,0].set_title('Sample x Coordinate Frequency',fontsize=18)
axs[1,0].set_ylabel('Frequency',fontsize=14)
axs[1,0].set_xlabel('x Position',fontsize=14)
axs[0,1].plot(samples[:s_plot,1])
axs[0,1].set_title('Sample y Traces',fontsize=18)
axs[0,1].set_xlabel('Sample Number',fontsize=14)
axs[0,1].set_ylabel('y Position',fontsize=14)
axs[1,1].hist(samples[:s_plot,1],100)
axs[1,1].set_title('Sample y Coordinate Frequency',fontsize=18)
axs[1,1].set_xlabel('y Position',fontsize=14)
fig.tight_layout()
st.pyplot(fig)   

###########################################################################
###############-----Re-Plot Data with Contour Lines-----################### 
###########################################################################  
st.markdown('''Estimate probability density from samples''')
with st.expander(r'''See Code'''):
    prob_code='''
from scipy.stats import gaussian_kde
X,Y = np.mgrid[-4:6:.1, -4:6:.1]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = gaussian_kde(samples[:2000,:].T)
Z = np.reshape(kernel(positions).T, X.shape)

def plot_samples_and_density(s_plot=1000):
    fig, axs = plt.subplots(figsize=(10,10));
    kernel = gaussian_kde(samples[:s_plot,:].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    Met_Pop_I=axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4) #, linewidth=0)
    Met_Pop_RS=axs.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)
    axs.set_title('Metropolis Sampling',fontsize=18)
    axs.set_ylabel('y Position',fontsize=10)
    axs.set_xlabel('x Position',fontsize=10)
    axs.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k')
    axs.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5)
    axs.set_xlim([-4,6]);
    axs.set_ylim([-4,6]);
    cbar1 = plt.colorbar(Met_Pop_I)
    cbar1.set_label('Ideal Population Density', rotation=270, fontsize=10,labelpad=15)
    cbar2 = plt.colorbar(Met_Pop_RS)
    cbar2.set_label('Sampled Population Density', rotation=270, fontsize=10,labelpad=15)
    return fig'''
    st.code(prob_code, language='python')   

from scipy.stats import gaussian_kde
X,Y = np.mgrid[-4:6:.1, -4:6:.1]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = gaussian_kde(samples[:2000,:].T)
Z = np.reshape(kernel(positions).T, X.shape)

def plot_samples_and_density(s_plot=1000):
    fig, axs = plt.subplots(figsize=(10,10));
    kernel = gaussian_kde(samples[:s_plot,:].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    Met_Pop_I=axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4) #, linewidth=0)
    Met_Pop_RS=axs.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)
    axs.set_title('Metropolis Sampling',fontsize=18)
    axs.set_ylabel('y Position',fontsize=10)
    axs.set_xlabel('x Position',fontsize=10)
    axs.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k')
    axs.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5)
    axs.set_xlim([-4,6]);
    axs.set_ylim([-4,6]);
    cbar1 = plt.colorbar(Met_Pop_I)
    cbar1.set_label('Ideal Population Density', rotation=270, fontsize=10,labelpad=15)
    cbar2 = plt.colorbar(Met_Pop_RS)
    cbar2.set_label('Sampled Population Density', rotation=270, fontsize=10,labelpad=15)
    return fig

psd_plot=plot_samples_and_density(s_plot)
st.pyplot(psd_plot)

st.markdown('''Here is a short video showing the progression of the sampleing procedure.  Note how the sample path meanders due to the sample generation being dependent on the previous samples.  In mathematics and statistics, this behavior is refered to as a *random walk*.''')
#Gif showing the "random walk" of the Markov Chain
video_file = open('Metropolis_2D_sampling.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)  
