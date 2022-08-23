from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors
import os


def metropolis(pi, dims, n_samples, burn_in=0.1, var=1):
    # start with random initial position.  Here a gaussian distribution is used.
    theta_ = np.random.randn(dims) * var
    samples = np.empty((n_samples, dims))
    # sampling loop
    for i in range(n_samples):
        # while len(samples) < n_samples:
        # proposal step
        theta = theta_ + np.random.randn(dims) * var

        # ratio of probabilities between proposed and current step
        ratio = pi(theta) / pi(theta_)

        # check acceptance - note: if ratio > 1, by def. larger than rv and accepted:
        if np.random.rand(1) < ratio:
            sample = theta
            theta_ = theta
            samples[i, :] = sample
            # samples.append(sample)

        # reject: remain at original state and add to trace:
        else:
            sample = theta_
            samples[i, :] = sample

    # remove burn-in phase (to do)
    return samples[int(n_samples * burn_in):, :]


def make_pdf(mean1, mean2, cov1, cov2):
    pdf1 = multivariate_normal(mean1, cov1)
    pdf2 = multivariate_normal(mean2, cov2)

    def pdf(x):
        return pdf1.pdf(x) + pdf2.pdf(x)

    return pdf


def plot_samples_and_density(s_plot=1000):
    fig, axs = plt.subplots(figsize=(10, 10))
    kernel = gaussian_kde(samples[:s_plot, :].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    Met_Pop_I = axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.4)  # , linewidth=0)
    Met_Pop_RS = axs.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)
    axs.set_title('Metropolis Sampling', fontsize=18)
    axs.set_ylabel('y Position', fontsize=10)
    axs.set_xlabel('x Position', fontsize=10)
    axs.scatter(samples[:s_plot, 0], samples[:s_plot, 1], s=1.5, c='k')
    axs.plot(samples[:s_plot, 0], samples[:s_plot, 1], lw=1, alpha=0.5)
    axs.set_xlim([-4, 6]);
    axs.set_ylim([-4, 6]);
    cbar1 = plt.colorbar(Met_Pop_I)
    cbar1.set_label('Ideal Population Density', rotation=270, fontsize=10, labelpad=15)
    cbar2 = plt.colorbar(Met_Pop_RS)
    cbar2.set_label('Sampled Population Density', rotation=270, fontsize=10, labelpad=15)
    return fig


def plot_samples(s_plot=1000):
    fig, axs = plt.subplots(figsize=(10, 10));
    axs.set_title('Metropolis Sampling', fontsize=18)
    axs.set_ylabel('Y Position', fontsize=10)
    axs.set_xlabel('X Position', fontsize=10)
    Met_Pop = axs.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', alpha=0.6)  # , linewidth=0);
    axs.scatter(samples[:s_plot, 0], samples[:s_plot, 1], s=1.5, c='k')
    axs.plot(samples[:s_plot, 0], samples[:s_plot, 1], lw=1, alpha=0.5)
    axs.set_xlim([-4, 6]);
    axs.set_ylim([-4, 6]);
    cbar = plt.colorbar(Met_Pop)
    cbar.set_label('Population Density', rotation=270, fontsize=10, labelpad=15)
    return fig


# %%
mean1 = [2, 2]
mean2 = [-1, -1]
cov1 = np.array([[1, 0.5], [0.5, 1]], dtype=float)
cov2 = np.array([[1, -0.3], [-0.3, 1]], dtype=float)
pdf1 = multivariate_normal(mean1, cov1)
pdf2 = multivariate_normal(mean2, cov2)

mgm_plt = plt.figure(figsize=(10, 10))
x, y = np.mgrid[-4:6:.01, -4:6:.01]
pos = np.dstack((x, y))

pdf = make_pdf(mean1, mean2, cov1, cov2)
samples = metropolis(pdf, 2, 10_000, 0., 1)

X, Y = np.mgrid[-4:6:.1, -4:6:.1]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = gaussian_kde(samples[:2000, :].T)
Z = np.reshape(kernel(positions).T, X.shape)

# %%
try:
    os.mkdir(r'..\frames')
except:
    print('frames folder exists!')

# %%
count = 1

for s_plot in range(5, 501):

    x_pad = 0.08
    y_pad = 0.08
    low_row_dy = 0.15
    legend_height = 0.3

    fig1 = plt.figure(figsize=(8, 8))
    left_trace = fig1.add_axes([x_pad, y_pad, 0.5 - (1.5 * x_pad), low_row_dy])
    right_trace = fig1.add_axes([0.5 + x_pad, y_pad, 0.5 - (1.5 * x_pad), low_row_dy])

    top_row_y = y_pad + low_row_dy + (2 * y_pad)
    main_frame = fig1.add_axes([x_pad, top_row_y, 0.6 - x_pad, 1 - (top_row_y + 1.5 * y_pad)])

    legend_1 = fig1.add_axes([0.6 + 1 * x_pad, top_row_y, 0.05, legend_height])
    legend_2 = fig1.add_axes([0.6 + 2.3 * x_pad + 0.05, top_row_y, 0.05, legend_height])
    text_box = fig1.add_axes([0.6 + 1.5 * x_pad, 0.75, 1 - (0.6 + 1.5 * x_pad + x_pad), 0.10])
    fig1.suptitle('Metropolis sampling 2D\nStepwise sampling and trace plots', fontsize='20')

    main_frame.set_xlabel('X position')
    main_frame.set_ylabel('Y position')

    text_box.text(0.95, 0.5, str(s_plot), ha='right', va='center', fontsize=25)
    text_box.set_title('Sample Nr.')
    plt.axis('off')

    # contour plot
    kernel = gaussian_kde(samples[:s_plot, :].T)
    Z = np.reshape(kernel(positions).T, X.shape)
    main_frame.scatter(samples[:s_plot, 0], samples[:s_plot, 1], s=5, c='k', zorder=20)
    main_frame.scatter(samples[s_plot - 1, 0], samples[s_plot - 1, 1], s=25, facecolors='none', edgecolors='r',
                       marker='o', zorder=20)
    Met_Pop_I = main_frame.contourf(x, y, pdf1.pdf(pos) + pdf2.pdf(pos), 10, cmap='gray_r', zorder=0,
                                    alpha=0.4)  # , linewidth=0)
    Met_Pop_RS = main_frame.contour(X, Y, Z, 20, cmap='viridis_r', alpha=0.5, zorder=10)
    main_frame.plot(samples[:s_plot, 0], samples[:s_plot, 1], lw=1, alpha=0.5)
    main_frame.set_xlim([-4, 5]);
    main_frame.set_ylim([-4, 5]);

    # trace plots
    left_trace.plot(samples[:s_plot, 0])
    left_trace.set_ylabel('X Position')
    left_trace.set_xlabel('Sample Number')
    left_trace.set_title('Sample X trace')

    right_trace.plot(samples[:s_plot, 1])
    right_trace.set_ylabel('Y Position')
    right_trace.set_xlabel('Sample Number')
    right_trace.set_title('Sample Y trace')

    # colorbars
    cbar2 = plt.colorbar(Met_Pop_I, cax=legend_2, orientation='vertical')
    cbar2.set_label('Ideal Population Density', rotation=270, fontsize=10, labelpad=15)
    norm = matplotlib.colors.Normalize(vmin=Met_Pop_RS.cvalues.min(), vmax=Met_Pop_RS.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=Met_Pop_RS.cmap)
    sm.set_array([])

    cbar1 = plt.colorbar(sm, cax=legend_1, orientation='vertical', ticks=Met_Pop_RS.levels[::4])
    cbar1.set_label('Sampled Population Density', rotation=270, fontsize=10, labelpad=15)

    if s_plot < 15:
        for i in range(6):
            fig1.savefig(r'..\frames\frame_%04d.png' % count)
            count += 1
    if (s_plot >= 10) and (s_plot < 50):
        for i in range(3):
            fig1.savefig(r'..\frames\frame_%04d.png' % count)
            count += 1
    if s_plot >= 50:
        fig1.savefig(r'..\frames\frame_%04d.png' % count)
        count += 1

