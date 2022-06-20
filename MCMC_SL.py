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

###Metropolis Function###
st.markdown('''Define Metropolis sampling procedure''')
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

    plt.figure(figsize=(10,10))
    x, y = np.mgrid[-4:6:.01, -4:6:.01]
    pos = np.dstack((x, y))
    plt.contour(x, y, pdf1.pdf(pos) + pdf2.pdf(pos))
    plt.savefig("multigauss.png")

    pdf = make_pdf(mean1, mean2, cov1, cov2)'''
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
plt.contour(x, y, pdf1.pdf(pos) + pdf2.pdf(pos))
plt.savefig("multigauss.png")

pdf = make_pdf(mean1, mean2, cov1, cov2)

st.pyplot(mgm_plt)

###Perform sampling###

samples = metropolis(pdf, 2, 10_000, 0., 1)

def plot_samples(s_plot=1000):
    plt.figure(figsize=(10,10))
    plt.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4) #, linewidth=0)
    plt.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k')
    plt.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5)
    plt.xlim([-4,6])
    plt.ylim([-4,6])
    #plt.savefig("multigauss_sampling.png")

# number of samples to plot from beginning of trace     
s_plot = st.slider('Number of Samples', min_value=0, max_value=5000, value=10, step=100, format=None, key=1, help=None, on_change=None, args=None, kwargs=None, disabled=False)
st.pyplot(plot_samples(s_plot))

st.markdown('''Investigate traces to gain insight''')
with st.expander(r'''See Code'''):
    trace_code='''s_plot_max = 1000
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(samples[:s_plot_max,0])
    ax2.plot(samples[:s_plot_max,1])'''
    st.code(trace_code, language='python')

s_plot_max = 1000
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(samples[:s_plot_max,0])
ax2.plot(samples[:s_plot_max,1])
st.pyplot(fig)

st.markdown('''Estimate probability density from samples''')
with st.expander(r'''See Code'''):
    prob_code='''
    from scipy.stats import gaussian_kde
    X, Y = np.mgrid[-4:6:.1, -4:6:.1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(samples[:2000,:].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    def plot_samples_and_density(s_plot=1000):
        plt.figure(figsize=(10,10))
        kernel = gaussian_kde(samples[:s_plot,:].T)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4) #, linewidth=0)
        plt.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)
        plt.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k')
        plt.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5)
        plt.xlim([-4,6])
        plt.ylim([-4,6])'''
    st.code(prob_code, language='python')
    
from scipy.stats import gaussian_kde
X,Y = np.mgrid[-4:6:.1, -4:6:.1]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = gaussian_kde(samples[:2000,:].T)
Z = np.reshape(kernel(positions).T, X.shape)

def plot_samples_and_density(s_plot=1000):
    plt.figure(figsize=(10,10))
    kernel = gaussian_kde(samples[:s_plot,:].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    plt.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4) #, linewidth=0)
    plt.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)
    plt.scatter(samples[:s_plot,0], samples[:s_plot,1], s=1.5, c='k')
    plt.plot(samples[:s_plot,0], samples[:s_plot,1], lw=1, alpha=0.5)
    plt.xlim([-4,6])
    plt.ylim([-4,6])
    # plt.savefig("multigauss_sampling.png")

#plt.figure(figsize=(12,6))
#plt.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4, linewidth=0)
#plt.scatter(samples[:2000,0], samples[:2000,1], s=10.5)
psd_plot=plot_samples_and_density(s_plot)
st.pyplot(psd_plot)

st.markdown('''Simple 1-D example''')
with st.expander(r'''See Code'''):
    oned_code='''from scipy.stats import norm
    pdf_1D = norm(2, 2)
    plt.figure(figsize=(12,8))
    theta = np.arange(-7,11,0.1)
    plt.plot(theta, pdf_1D.pdf(theta))'''
    st.code(oned_code, language='python')

from scipy.stats import norm
pdf_1D = norm(2, 2)
oned_plot=plt.figure(figsize=(12,8))
theta = np.arange(-7,11,0.1)
plt.plot(theta, pdf_1D.pdf(theta))
st.pyplot(oned_plot)

st.markdown('''Gernerate a new metropolis sample''')
samples = metropolis(pdf_1D.pdf, 1, 10_000, 0., 1)
#sp_plot=plt.plot(samples)
st.pyplot(plt.plot(samples))
#sh_plot=plt.hist(samples, 100)
st.pyplot(plt.hist(samples, 100))

def plot_1D_with_samples(n_plot_samples):
    plt.figure(figsize=(12,8))
    theta = np.arange(-7,11,0.1)
    plt.plot(theta, pdf_1D.pdf(theta))
    plt.vlines(samples[:n_plot_samples],0,0.02,'r', alpha=0.2)
n_plot = st.slider('Number of Samples', min_value=1, max_value=1000, value=5, step=10, format=None, key=2, help=None, on_change=None, args=None, kwargs=None, disabled=False)
st.pyplot(plot_1D_with_samples(n_plot))

