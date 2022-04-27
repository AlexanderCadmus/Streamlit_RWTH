import matplotlib.cm
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import numpy as np

# set data source

lith_URL = 'https://raw.githubusercontent.com/NilsChudalla/ET_Sardinia_WS/master/lith_block.csv'
prob_URL = 'https://raw.githubusercontent.com/NilsChudalla/ET_Sardinia_WS/master/prob_block.csv'
ent_URL = 'https://raw.githubusercontent.com/NilsChudalla/ET_Sardinia_WS/master/ent_block.csv'
leg_URL = 'https://raw.githubusercontent.com/NilsChudalla/ET_Sardinia_WS/master/legend.png'
logo_URL = 'https://raw.githubusercontent.com/NilsChudalla/ET_Sardinia_WS/master/ET_logo.png'

resolution = np.array([80, 120, 60])

cdict = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.01, 0.01),
                   (1.0,  0.01, 0.01)),

         'green': ((0.0, 1.0, 1.0),
                   (0.5, 0.36, 0.36),
                   (1.0, 0.99, 0.99)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.5,  0.98, 0.98),
                   (1.0,  0.96, 0.96)),

         'alpha':  ((0.0,  0.0, 0.0),
                   (0.5,  0.5, 0.5),
                   (1.0,  1.0, 1.0))}

cmap_name = 'entropy_layers'
cm = LinearSegmentedColormap(cmap_name, cdict)
plt.register_cmap(cmap=cm)

head1, head2, head3= st.columns(3)
head1.image(logo_URL)

st.title('Uncertainty in site investigation')
st.markdown(r'''This is a simple visualization of uncertainties in the model presented by Nils Chudalla 
(RWTH Aachen University, CGRE). The model focuses on the Cretaceous sediments in the EMR region. Data sources are the
SCAN 2D seismic campaign and openly accessible borehole data. Uncertainties are computed over 300 iterations; we assume a 
standard deviation of +/- 40 m for the seismic data and +/- 10 m for the borehole data.''')
st.markdown('''It contains options to view either the implicit solution, probability or information entropy. 
These were calculated using the open source modeling library **Gempy**. ''')

# Todo: 2 columns, one for radio, one for legend
col1, col2 = st.columns(2)
selection = col1.radio("Plot solution", ('Lithology', 'Probability', 'Entropy'))
col2.image(leg_URL)




#https://github.com/NilsChudalla/ET_Sardinia_WS/blob/master/ent_block.npy?raw=true

# load data into cache
@st.cache
def load_data():
    lith_block = pd.read_csv(lith_URL, usecols=['vals']).values.reshape(resolution)
    prob_block = pd.read_csv(prob_URL, usecols=['vals']).values.reshape(resolution)
    ent_block = pd.read_csv(ent_URL, usecols=['vals']).values.reshape(resolution)
    #block = np.load(DATA_URL, allow_pickle=True).reshape(resolution)
    return lith_block, prob_block, ent_block

# info text for loading data

lith_block, prob_block, ent_block = load_data()

if selection == 'Lithology':
    block = (lith_block * 2).astype(int)
    title = 'Implicit solution'
    colormap = 'gray'
    vmin = 1.5
    vmax = 3.5
    fill_c = '#000000'

elif selection == 'Probability':
    block = prob_block
    title = 'Probability visualization'
    colormap = 'Purples'
    vmin = 0.0
    vmax = 1.0
    fill_c = '#000000'

elif selection == 'Entropy':
    block = ent_block
    title = 'Entropy visualization'
    colormap = 'entropy_layers'
    vmin = 0.0
    vmax = 1.0
    fill_c = '#000000'

curr_cmap = matplotlib.cm.get_cmap(colormap)
curr_cmap.set_bad(color=fill_c)



# Setting plotting parameters
xmin, xmax, ymin, ymax, zmin, zmax = [692695, 716177, 5613791, 5641768, -400, 400]

d_grid = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
cell_width = d_grid / resolution
grid_min = np.array([xmin, ymin, zmin]) + cell_width/0.5
grid_max = np.array([xmax, ymax, zmax]) - cell_width/0.5

x_array = np.linspace(grid_min[0], grid_max[0], resolution[0].astype(int))
y_array = np.linspace(grid_min[1], grid_max[1], resolution[1].astype(int))
z_array = np.linspace(grid_min[2], grid_max[2], resolution[2].astype(int))

XX_xy, YY_xy = np.meshgrid(x_array, y_array)
XX_xz, ZZ_xz = np.meshgrid(x_array, z_array)
YY_yz, ZZ_yz = np.meshgrid(y_array, z_array)

# Add slider

st.subheader(title)



# Create subsection

depth = float(np.mean(z_array))
WE_profile = float(np.mean(y_array))
NS_profile = float(np.mean(x_array))


depth = st.slider(label='Depth slider', min_value=float(np.min(z_array)), max_value=float(np.max(z_array)),
                      value=float(np.mean(z_array)))
depth_index = np.argmin(np.abs(z_array - depth))
WE_profile = st.slider(label='Profile slider (N-S)', min_value=float(np.min(y_array)), max_value=float(np.max(y_array)),
                       value=float(np.mean(y_array)))
WE_index = np.argmin(np.abs(y_array - WE_profile))
NS_profile = st.slider(label='Profile slider (W-E)', min_value=float(np.min(x_array)), max_value=float(np.max(x_array)),
                       value=float(np.mean(x_array)))
NS_index = np.argmin(np.abs(x_array - NS_profile))
cross_sections = st.checkbox('Toggle relative profile positions')



with st.expander('Depthmap'):

    hslice_coords = block[:, :, depth_index]

    fig1 = plt.figure()
    AX = gridspec.GridSpec(1,1)
    AX.update(wspace = 0.1, hspace = 0.5)
    ax1 = plt.subplot(AX[0,0])

    ax1.set_title('Depthmap %s' %selection)

    if selection == 'Entropy':
        hslice_background = (lith_block[:, :, depth_index] * 2).astype(int)
        ax1.imshow(hslice_background.T, extent=[692695, 716177, 5613791, 5641768], origin='lower', cmap='gray', vmin=1.5, vmax=3.5)

    ax1.imshow(hslice_coords.T, extent=[692695, 716177, 5613791, 5641768], origin='lower', cmap=curr_cmap, vmin=vmin, vmax=vmax)
    #ax1.axis('equal')

    if cross_sections:
        ax1.vlines(NS_profile, ymin=grid_min[1], ymax=grid_max[1], color='#a32632', lw=1)
        ax1.hlines(WE_profile, xmin=grid_min[0], xmax=grid_max[0], color='#a32632', lw=1)
    st.pyplot(fig1)

with st.expander('W-E profile'):

    WE_slice = block[:, WE_index, :]

    fig2 = plt.figure()
    AX = gridspec.GridSpec(1,1)
    AX.update(wspace = 0.1, hspace = 0.5)

    ax2 = plt.subplot(AX[0,0])
    ax2.set_title('%s - W-E' %selection)
    #ax2.contourf(XX_xz, ZZ_xz, WE_slice.T, cmap='magma', vmin=vmin, vmax=vmax)
    if selection == 'Entropy':
        WE_background = (lith_block[:, WE_index, :] * 2.0).astype(int)
        ax2.imshow(WE_background.T, extent=[692695, 716177, -1200, 1200], origin='lower', cmap='gray',
                   vmin=1.5, vmax=3.5)

    ax2.imshow(WE_slice.T, extent=[692695, 716177, -1200, 1200], vmin=vmin, vmax=vmax, cmap=curr_cmap, origin='lower')
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[1] = '-285'
    labels[-2] = '285'
    ax2.yaxis.set_ticks(np.linspace(-1200, 1200, 7))
    ax2.yaxis.set_ticklabels(labels)

#ax2.axis('equal')

#ax2.set_ylim(np.min(z_array), np.max(z_array))

    if cross_sections:
        ax2.vlines(NS_profile, ymin=-1200, ymax=1200, color='#a32632', lw=1)
        ax2.hlines(depth*3, xmin=grid_min[0], xmax=grid_max[0], color='#a32632', lw=1)
    st.pyplot(fig2)

with st.expander('N-S profile'):

    NS_slice = block[NS_index, :, :]
    fig3 = plt.figure()
    AX = gridspec.GridSpec(1, 1)
    AX.update(wspace=0.1, hspace=0.5)

    ax3 = plt.subplot(AX[0,0])
    ax3.set_title('%s - N-S' %selection)
    if selection == 'Entropy':
        NS_background = (lith_block[NS_index, :, :] * 2).astype(int)
        ax3.imshow(NS_background.T, extent=[5613791, 5641768, -1200, 1200], origin='lower', cmap='gray',
                   vmin=1.5, vmax=3.5)

    ax3.imshow(NS_slice.T, cmap=curr_cmap, extent=[5613791, 5641768, -1200, 1200], vmin=vmin, vmax=vmax, origin='lower')
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[1] = '-285'
    labels[-2] = '285'
    ax3.yaxis.set_ticks(np.linspace(-1200, 1200, 7))
    ax3.yaxis.set_ticklabels(labels)
    ax3.invert_xaxis()

    if cross_sections:
        ax3.vlines(WE_profile, ymin=-1200, ymax=1200, color='#a32632', lw=1)
        ax3.hlines(depth*3, xmin=grid_min[1], xmax=grid_max[1], color='#a32632', lw=1)
    st.pyplot(fig3)

# ax4 = fig.add_axes([0.65, 0.05, 0.05, 0.37])
# x1 = np.array([0.0,1.0])
# XX1, ZZ1 = np.meshgrid(x1, z_array)
# vals = np.vstack((block[NS_index, WE_index,:], block[NS_index, WE_index,:]))
# ax4.set_title('"Borehole view"')
# ax4.contourf(XX1, ZZ1, vals.T, cmap='magma', vmin=vmin, vmax=vmax)
# ax4.set_xlim(0.1, 0.9)
# ax4.axes.get_xaxis().set_ticks([])
#
# if cross_sections:
#     ax4.hlines(depth, xmin=-1, xmax=2, color='white', lw=1)
#
# ax5 = fig.add_axes([0.8, 0.05, 0.05, 0.36])
# fig.colorbar(im, cax=ax5, orientation='vertical', ticks=np.linspace(vmin, vmax, 4), boundaries=[vmin, vmax], label='entropy')
#
#
# st.pyplot(fig)


