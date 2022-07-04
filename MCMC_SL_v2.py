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
st.markdown('''___________________________________________''')
st.markdown('''Part 1: Multimodal Example''')
st.markdown('''___________________________________________''')
with st.sidebar:
    st.markdown('''Part 1 Options:''')
    st.markdown('''Samples to be picked and plotted''')
    s_plot = st.slider('Number of Samples', min_value=0, max_value=5000, value=10, step=100, format=None, key=1, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    st.markdown('''Part 2 Options:''')
    st.markdown('''Display number of samples picked from population''')
    n_plot = st.slider('Sampled Poplation Data', min_value=1, max_value=1000, value=5, step=10, format=None, key=2, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    st.markdown('''Variance of population''')
    var = st.slider('Var', min_value=0, max_value=10, value=1, step=1, format=None, key=3, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    st.markdown('''Size of sample population''')
    n_plot_samples = st.slider('n_plot_samples', min_value=1, max_value=100, value=1, step=1, format=None, key=4, help=None, on_change=None, args=None, kwargs=None, disabled=False)        
###########################################################################
###################-----Define Sampling Procedure-----##################### 
###########################################################################
###Metropolis Function###
st.markdown('''Define Metropolis sampling procedure''')
with st.expander(r'''See Code'''):
    metroplis_code='''def metropolis(pi, dims, n_samples, burn_in=0.1, var=1):
        # start with random initial position
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

def metropolis(pi, dims, n_samples, burn_in=0.1, var=1):
    # start with random initial position
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
############-----Generate & Plot Bimodal Gaussian Model-----############### 
###########################################################################    
st.markdown('''Example: multimodal Gaussian model''')
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
    plt.title('Metropolis Population Density',fontsize=18)
    plt.xlabel('x Position',fontsize=10)
    plt.ylabel('y Position',fontsize=10)
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

mean1 = [3, 3]
mean2 = [-1, -1]
cov1 = np.array([[1,0.5],[0.5,1]], dtype=float)
cov2 = np.array([[1,-0.3],[-0.3,1]], dtype=float)
pdf1 = multivariate_normal(mean1, cov1)
pdf2 = multivariate_normal(mean2, cov2)

mgm_plt=plt.figure(figsize=(10,10))
x, y = np.mgrid[-4:6:.01, -4:6:.01]
pos = np.dstack((x, y))
plt.title('Metropolis Population Density',fontsize=18)
plt.xlabel('x Position',fontsize=10)
plt.ylabel('y Position',fontsize=10)
plt.contour(x, y, pdf1.pdf(pos) + pdf2.pdf(pos))
plt.colorbar()
plt.savefig("multigauss.png")

pdf = make_pdf(mean1, mean2, cov1, cov2)

samples = metropolis(pdf, 2, 10_000, 0., 1)

st.pyplot(mgm_plt)

###########################################################################
###########-----Implement Sampling Procedure & Plot Data-----############## 
###########################################################################

st.markdown('''Generate and plot samples''')
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
    # plt.savefig("multigauss_sampling.png")

st.pyplot(plot_samples(s_plot))   

###########################################################################
###############-----Examine Trace Data-----################################ 
###########################################################################  
st.markdown('''Investigate traces to gain insight''')
with st.expander(r'''See Code'''):
    trace_code='''s_plot_max = 1000
    fig_data = plt.figure(figsize=(14,6));
    fig_data.suptitle('Sample Coordinate Traces', fontsize=16)
    ax1 = fig_data.add_subplot(211);
    ax2 = fig_data.add_subplot(212);
    ax1.plot(samples[:s_plot_max,0]);
    ax1.set_ylabel('x Position');
    ax2.plot(samples[:s_plot_max,1]);
    ax2.set_xlabel('Sample Number');
    ax2.set_ylabel('y Position');
    st.pyplot(fig_data)'''
    st.code(trace_code, language='python')
    
# number of samples to plot from beginning of trace     

s_plot_max = 1000
fig_data = plt.figure(figsize=(14,6));
fig_data.suptitle('Sample Coordinate Traces', fontsize=16)
ax1 = fig_data.add_subplot(211);
ax2 = fig_data.add_subplot(212);
ax1.plot(samples[:s_plot_max,0]);
ax1.set_ylabel('x Position');
ax2.plot(samples[:s_plot_max,1]);
ax2.set_xlabel('Sample Number');
ax2.set_ylabel('y Position');
st.pyplot(fig_data)
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

###########################################################################
###############-----Simple 1-D Distribution-----########################### 
###########################################################################
st.markdown('''___________________________________________''')
st.markdown('''Part 2: 1-D Example''')
st.markdown('''___________________________________________''')
  
st.markdown('''Simple 1-D example''')
with st.expander(r'''See Code'''):
    oned_code='''from scipy.stats import norm
    pdf_1D = norm(2, 2);
    oned_plot=plt.figure(figsize=(12,8));
    theta = np.arange(-7,11,0.1);
    plt.plot(theta, pdf_1D.pdf(theta));
    plt.title('1-D Population Distribution');
    plt.xlabel('Position');
    plt.ylabel('Frequency');'''
    st.code(oned_code, language='python')

from scipy.stats import norm
pdf_1D = norm(2, 2);
oned_plot=plt.figure(figsize=(12,8));
theta = np.arange(-7,11,0.1);
plt.plot(theta, pdf_1D.pdf(theta));
plt.title('1-D Population Distribution');
plt.xlabel('Position');
plt.ylabel('Frequency');
###########################################################################
############-----1-D Sampling Procedure with Plotted Data-----############# 
###########################################################################  
st.markdown('''Gernerate a new metropolis sample''')
with st.expander(r'''See Code'''):
    oned_code='''samples = metropolis(pdf_1D.pdf, 1, 10_000, 0., 1)
    
    fig_all_data, axs = plt.subplots(2,1, figsize=(20,12))
    fig_all_data.suptitle('1-D Sample Data', fontsize=22)
    axs[0].plot(samples[:,0])
    axs[0].set_title('Sample Traces',fontsize=18)
    axs[0].set_xlabel('Sample Number',fontsize=14)
    axs[0].set_ylabel('Position',fontsize=14)
    axs[1].hist(samples[:,0],100)
    axs[1].set_title('Sample Position Frequency',fontsize=18)
    axs[1].set_ylabel('Frequency',fontsize=14)
    axs[1].set_xlabel('Position',fontsize=14)
    fig_all_data.tight_layout()'''
    st.code(oned_code, language='python')
    
samples = metropolis(pdf_1D.pdf, 1, 10_000, 0., 1)
#sp_plot=plt.plot(samples)
#st.pyplot(plt.plot(samples))
#sh_plot=plt.hist(samples, 100)
#st.pyplot(plt.hist(samples, 100))

#1D sample data display
fig_all_data, axs = plt.subplots(2,1, figsize=(20,12))
fig_all_data.suptitle('1-D Sample Data', fontsize=22)
axs[0].plot(samples[:,0])
axs[0].set_title('Sample Traces',fontsize=18)
axs[0].set_xlabel('Sample Number',fontsize=14)
axs[0].set_ylabel('Position',fontsize=14)
axs[1].hist(samples[:,0],100)
axs[1].set_title('Sample Position Frequency',fontsize=18)
axs[1].set_ylabel('Frequency',fontsize=14)
axs[1].set_xlabel('Position',fontsize=14)
fig_all_data.tight_layout()
st.pyplot(fig_all_data)

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
st.markdown('''Plot 1-D data in simple curve''')
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
    fig, axs = plt.subplots(figsize=(12,8));
    axs.set_title('Population Distribution')
    axs.set_xlabel('Position')
    axs.set_ylabel('Frequency')
    theta = np.arange(-7,11,0.1)
    axs.plot(theta, pdf_1D.pdf(theta))
    axs.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.2)
    return fig

st.pyplot(plot_1D_with_samples(n_plot))
###########################################################################
###################-----Stepwise Sampling Procedure-----################### 
###########################################################################  
st.markdown('''Stepwise sampling vis''')
with st.expander('See Code'):
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
st.markdown('''Plotting Sampled data''')
with st.expander('See Code'):
    sws_code='''def plot_1D_with_samples_and_proposal(n_plot_samples):
        # create plot with sampled locations and proposal step for current iteration
        fig, axs = plt.subplots(figsize=(12,8))
        theta = np.arange(-7,11,0.1)
        axs.set_title('Population Distribution')
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
    axs.set_title('Population Distribution')
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
    return fig
        

st.pyplot(plot_1D_with_samples_and_proposal(n_plot_samples))