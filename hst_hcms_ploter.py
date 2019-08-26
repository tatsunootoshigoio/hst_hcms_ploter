#-----------------------------------------------------#
# vsm measurement plotter v0.6			  ----#
# former name: vsm_plt_v05_id_Hc-i.py 	          ----#
# author: tatsunootoshigo, 7475un00705hi90@gmail.com  #
#-----------------------------------------------------#

# Imports
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy import stats
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
# script version
version = '0.6'
version_name = 'hst_hcms_ploter_' + version + '.py'

# some useful constants
cm = 1e-2
nm = 1e-9
mu0 = 4*np.pi*1e-7
mega = 1e+6
tesla = 1e+4 # Gauss to tesla

# =================== SAMPLE DATA STARTS HERE =================

sample_mdate = '311118'
sample_id = 'E2978'
sample_no = 0
sample_bg = 'sub'

# measurment configuration
hdir = 'inpl'
label_hdir = r'$\parallel$'
#label_hdir = r'$\perp$'

slayer_step = 0.5
#slayer5 = 	r'$1W$'
#slayer4 = 	r'$Mg$'
slayer3 = 	r'$AlO_x$'
slayer2 = 	r'$Fe-Si$'
slayer1 = 	r'$W$'
substrate = r'$SiSi0_2$'

# sample dimensions in cm
sample_l = 1.0
sample_w = 1.0

# ferromagnetic layer thickness in nm
sample_fml_d = 30.0

# ferromagnetic layer volume in cm^3
sample_fml_vol = (sample_l*sample_w*sample_fml_d*1e+2)*cm**3 

# calibration coefficient for the perp & inpl 
cal_coeff = 4.961 / 2.808

# number of decimal points to display in tick labels
xprec = 0
yprec = 0

# set axes range to some nice round value
xmin = -50
xmax = 50
ymin = -10000
ymax = 10000

deskew_points = 10
deskew_fit_range = np.arange(-2.0, 2.0, 0.01)

# number of points for linear fit of Hc from OO for each quadrant pair
Hc_fit_points_q23 = 8
Hc_fit_points_q14 = 8
Hc_creep_q14 = 35
Hc_creep_q23 = 35

# plot Hc fit line in a specific range 
Hc_fit_range_q23 = np.arange(-1.0, 1.0, 0.01) # start, stop, step
Hc_fit_range_q14 = np.arange(-3.0, 3.0, 0.01) # start, stop, step

# number of points for linear fit of Ms from OO for each quadrant pair
Ms_fit_points_q12 = 8
Ms_fit_points_q34 = 8
Ms_creep_q12 = 0
Ms_creep_q34 = 0

# plot Ms fit line in a specific range 
Ms_fit_range_q12 = np.arange(-3.0, 3.0, 0.01) # start, stop, step
Ms_fit_range_q34 = np.arange(-1.0, 1.0, 0.01) # start, stop, step

# ===================================================================

# output pdf
out_pdf = 'VSM_hcms_plot_' + hdir + '_' + sample_id + '.pdf'
out_svg = 'VSM_hcms_plot_' + hdir + '_' + sample_id + '.svg'

# plot legend lables
label_inpl = r'$\parallel$'
label_perp = r'$\perp$'
sdelimiter = r'$\rfloor\lfloor$'

# axes labels for the mu0M vs. mu0H plots 
axis_label_mu0M = r'$\mu_0 M\, / \, G$'
axis_label_mu0H = r'$\mu_0 H\, / \, G$'
axis_label_mu0Ms = r'$\mu_0 M_s\, / \, T$'
axis_label_mu0Hk = r'$\mu_0 H_k\, / \, T$'
axis_label_th = r'$thickness\, / \, nm$'

# position and vertical separation of entries
mshk_legend_x = -1.45
mshk_legend_y = 1.35
mshk_legend_y_sep = 0.3

def vsm_open_in(sample_mdate, sample_id, sample_no):

	file_inpl = sample_mdate + sample_id + '-ii.txt'
	#file_perp = sample_mdate + sample_id + '-p.txt'
	
	# load raw data from text file, skiprows=12 --> gets rid of the vsm file headder
	x1, y1 = np.loadtxt(file_inpl, skiprows=12 ,unpack=True)
	#x2, y2 = np.loadtxt(file_perp, skiprows=12 ,unpack=True)
	
	# recalculate units to get mu0*M vs. mu0*H plot in teslas
	tx1 = x1
	ty1 = y1 * 4*np.pi*1e+3 / sample_fml_vol

	#tx2 = x2 / tesla  
	#ty2 = y2 * 4*np.pi*1e+3 / sample_fml_vol / tesla * cal_coeff 

	return np.array([tx1, ty1]);

def gen_plot_title(sample_no):
	# sample label describing the layers
	sample_label = '[' + substrate + sdelimiter + slayer1 + sdelimiter + slayer2 + sdelimiter + slayer3 + ']'

	# plot title using sample label
	plt_title = sample_label + '	id: ' + sample_id 
	
	return plt_title;
	
def custom_axis_formater(custom_title, custom_x_label, custom_y_label, xmin, xmax, ymin, ymax, xprec, yprec):
	
	# get axes and tick from plot 
	ax = plt.gca()
	major_tx_no_x = 5
	major_tx_no_y = 8
	# set the number of major and minor bins for x,y axes
	# prune='lower' --> remove lowest tick label from x axis
	xmajorLocator = MaxNLocator(major_tx_no_x, prune='lower') 
	xmajorFormatter = FormatStrFormatter('%.'+ np.str(xprec) + 'f')
	xminorLocator = MaxNLocator(5*major_tx_no_x) 
	
	ymajorLocator = MaxNLocator(major_tx_no_y) 
	ymajorFormatter = FormatStrFormatter('%.'+ np.str(yprec) + 'f')
	yminorLocator = MaxNLocator(5*major_tx_no_y)
	
	# format major and minor ticks width, length, direction 
	ax.tick_params(which='both', width=2, direction='in', labelsize=24)
	ax.tick_params(which='major', length=6)
	ax.tick_params(which='minor', length=4)

	# set axes thickness
	ax.spines['top'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.spines['right'].set_linewidth(2)
	ax.spines['left'].set_linewidth(2)

	ax.xaxis.set_major_locator(xmajorLocator)
	ax.yaxis.set_major_locator(ymajorLocator)

	ax.xaxis.set_major_formatter(xmajorFormatter)
	ax.yaxis.set_major_formatter(ymajorFormatter)

	# for the minor ticks, use no labels; default NullFormatter
	ax.xaxis.set_minor_locator(xminorLocator)
	ax.yaxis.set_minor_locator(yminorLocator)

	# grid and axes are drawn below the data plot
	ax.set_axisbelow(True)

	# add x,y grids to plot area
	ax.xaxis.grid(True, zorder=0, color='lightgray', linestyle='-', linewidth=1)
	ax.yaxis.grid(True, zorder=0, color='lightgray', linestyle='-', linewidth=1)

	# set axis labels
	ax.set_xlabel(custom_x_label, fontsize=24)
	ax.set_ylabel(custom_y_label, fontsize=24)

	# set plot title
	ax.set_title(custom_title, loc='right', fontsize=24)

	return;

def gen_mshk_legend(mshk_array):
	# get axes and tick from plot 
	ax = plt.gca()

	# add to plot calculated values of M_k and H_k
	ax.text(mshk_legend_x, mshk_legend_y, r'$\parallel \mu_0 M_s =$' + np.str(np.round(mshk_array[4], 2)) + r'$T$')
	ax.text(mshk_legend_x, mshk_legend_y - mshk_legend_y_sep, r'$\parallel \mu_0 H_k =$' + np.str(np.round(mshk_array[6], 2)) + r'$T$')
	ax.text(mshk_legend_x, mshk_legend_y - 2.0*mshk_legend_y_sep, r'$\perp \mu_0 M_s =$' + np.str(np.round(mshk_array[5], 2)) + r'$T$')
	ax.text(mshk_legend_x, mshk_legend_y - 3.0*mshk_legend_y_sep, r'$\perp \mu_0 H_k =$' + np.str(np.round(mshk_array[7], 2)) + r'$T$')
	return;

def vsm_fit_hst_Hc(vsm_data, Hc_fit_points_q23, Hc_fit_points_q14, Hc_creep_q14, Hc_creep_23):
	# liear fit of the hysteresis loops for each of quadrant pairs
	tx_half = (vsm_data.shape[1] - 1) / 2
	ty_half = (vsm_data.shape[1] - 1) / 2

	# pick a range of points from the calculated data definde by fit_points_inpl/perp
	xq14 = vsm_data[0, np.int_(tx_half - Hc_creep_q23 - Hc_fit_points_q23):np.int_(tx_half - Hc_creep_q23)] 
	xq23 = vsm_data[0, np.int_(2*tx_half - Hc_creep_q14 - Hc_fit_points_q14):np.int_(2*tx_half - Hc_creep_q14)]

	yq14 = vsm_data[1, np.int_(ty_half - Hc_creep_q23 - Hc_fit_points_q23):np.int_(ty_half - Hc_creep_q23)] 
	yq23 = vsm_data[1, np.int_(2*ty_half - Hc_creep_q14 - Hc_fit_points_q14):np.int_(2*ty_half - Hc_creep_q14)]
	
	# calculate linear fit parameters
	slope_q14, intercept_q14, r_value_q14, p_value_q14, std_err_q14 = stats.linregress(xq14, yq14)
	slope_q23, intercept_q23, r_value_q23, p_value_q23, std_err_q23 = stats.linregress(xq23, yq23)

	#print(slope_q14, slope_q23)
	#print(intercept_q14, intercept_q23)

	print('=================================')
	print('Hc1: ', intercept_q14 / slope_q14, 'Hc2: ', intercept_q23 / slope_q23, 'mean Hc:', 0.5*(np.absolute(intercept_q14 / slope_q14) + np.absolute(intercept_q23 / slope_q23)))
	print('---------------------------------')
	print('slope_q12: ', slope_q14)
	print('intercept_q12: ', intercept_q14)
	print('r_value_q12: ', r_value_q14)
	print('std_err_q12:', std_err_q14)
	print('---------------------------------')
	print('slope_q34: ', slope_q23)
	print('intercept_q34: ', intercept_q23)
	print('r_value_q34', r_value_q23)
	print('std_err_q34', std_err_q23)

	plt.figtext(0.2, 0.04, r'$H_c^{(q14)}:$ ' + '	a: ' + np.str(slope_q14) + '	b: ' + np.str(intercept_q14) + '	' + r'$R^2:$' + np.str(r_value_q14*r_value_q14) + '	S: ' + np.str(std_err_q14), size=14)
	plt.figtext(0.2, 0.02, r'$H_c^{(q23)}:$' + '	a: ' + np.str(slope_q23) + '	b: ' + np.str(intercept_q23) + '	' + r'$R^2:$ ' + np.str(r_value_q23*r_value_q23) + '	S: ' + np.str(std_err_q23), size=14)

	return np.array([slope_q14, slope_q23, intercept_q14, intercept_q23]);

def vsm_fit_hst_Ms(vsm_data, Ms_fit_points_q12, Ms_fit_points_q34, Ms_creep_q12, Ms_creep_34):
	# liear fit of the hysteresis loops for each of quadrant pairs
	tx_half = (vsm_data.shape[1] - 1) / 2
	ty_half = (vsm_data.shape[1] - 1) / 2

	# pick a range of points from the calculated data definde by fit_points_inpl/perp
	xq12 = vsm_data[0, np.int_(tx_half - Ms_creep_q34 - Ms_fit_points_q34):np.int_(tx_half - Ms_creep_q34)] 
	xq34 = vsm_data[0, np.int_(2*tx_half - Ms_creep_q12 - Ms_fit_points_q12):np.int_(2*tx_half - Ms_creep_q12)]

	yq12 = vsm_data[1, np.int_(ty_half - Ms_creep_q34 - Ms_fit_points_q34):np.int_(ty_half - Ms_creep_q34)] 
	yq34 = vsm_data[1, np.int_(2*ty_half - Ms_creep_q12 - Ms_fit_points_q12):np.int_(2*ty_half - Ms_creep_q12)]
	
	# calculate linear fit parameters
	slope_q12, intercept_q12, r_value_q12, p_value_q12, std_err_q12 = stats.linregress(xq12, yq12)
	slope_q34, intercept_q34, r_value_q34, p_value_q34, std_err_q34 = stats.linregress(xq34, yq34)

	print('=================================')
	print('Ms12: ', intercept_q12, 'Ms34: ', intercept_q34, 'Ms_mean: ', 0.5*(np.absolute(intercept_q12) + np.absolute(intercept_q34)))
	print('---------------------------------')
	print('slope_q12: ', slope_q12)
	print('intercept_q12: ', intercept_q12)
	print('r_value_q12: ', r_value_q12)
	print('std_err_q12:', std_err_q12)
	print('---------------------------------')
	print('slope_q34: ', slope_q34)
	print('intercept_q34: ', intercept_q34)
	print('r_value_q34', r_value_q34)
	print('std_err_q34', std_err_q34)

	plt.figtext(0.2, 0.08, r'$M_s^{(q12)}:$ ' + '	a: ' + np.str(slope_q12) + '	b: ' + np.str(intercept_q12) + '	' + r'$R^2:$' + np.str(r_value_q12*r_value_q12) + '	S: ' + np.str(std_err_q12), size=14)
	plt.figtext(0.2, 0.06, r'$M_s^{(q34)}:$' + '	a: ' + np.str(slope_q34) + '	b: ' + np.str(intercept_q34) + '	' + r'$R^2:$ ' + np.str(r_value_q34*r_value_q34) + '	S: ' + np.str(std_err_q34), size=14)

	return np.array([slope_q12, slope_q34, intercept_q12, intercept_q34]);

def vsm_plot_anno_Hc(fit_params):
	
	Hc1 = -1.0*(fit_params[2] / fit_params[0])
	Hc2 = -1.0*(fit_params[3] / fit_params[1])

	Hc_mean = 0.5*( np.absolute(Hc1) + np.absolute(Hc2) )
	
	# plot annotations
	ax = plt.gca()
	#ax.text(Hc1-1.0, 0, r'$H_{c1}: $' + np.str(np.round(Hc1, 3)) , fontsize=15)

	ax.annotate(r'$H_{c1}: $' + np.str(np.round(Hc1, 3)),
            xy=(Hc1, 0),  # theta, radius
            xytext=(0.2, 0.5),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle="simple", color='blue'),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=20
            )

	ax.annotate(r'$H_{c2}: $' + np.str(np.round(Hc2, 3)),
            xy=(Hc2, 0),  # theta, radius
            xycoords='data',
            xytext=(0.8, 0.5),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle="simple", color='red'),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=20
            )
	ax.annotate(r'$H_c: $' + np.str(np.round(Hc_mean, 3)),
            xy=(0, 0),  # theta, radius
            xycoords='data',
            xytext=(0.95, 0.5),    # fraction, fraction
            textcoords='axes fraction',
            #arrowprops=dict(facecolor='red', width=1.0, shrink=0.05),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=20
            )

	return;

def vsm_plot_anno_Ms(fit_params):
	
	Ms1 = fit_params[2]
	Ms2 = fit_params[3]

	Ms_mean = 0.5*( np.absolute(Ms1) + np.absolute(Ms2) )
	
	# plot annotations
	ax = plt.gca()
	#ax.text(Hc1-1.0, 0, r'$H_{c1}: $' + np.str(np.round(Hc1, 3)) , fontsize=15)

	ax.annotate(r'$M_{s1}: $' + np.str(np.round(Ms1, 3)),
            xy=(-6000, Ms1),  # theta, radius
            xytext=(0.165, 0.25),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle="simple", color='green'),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=20
            )
	ax.annotate(r'$M_{s2}: $' + np.str(np.round(Ms2, 3)),
            xy=(6000, Ms2),  # theta, radius
            xycoords='data',
            xytext=(0.83, 0.75),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle="simple", color='magenta'),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=20
            )
	ax.annotate(r'$M_s: $' + np.str(np.round(Ms_mean, 3)),
            xy=(0, 0),  # theta, radius
            xycoords='data',
            xytext=(0.95, 0.95),    # fraction, fraction
            textcoords='axes fraction',
            #arrowprops=dict(facecolor='red', width=1.0, shrink=0.05),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=20
            )

	return;

def vsm_data_deskew(vsm_data, deskew_points):

	tx_quater = (vsm_data.shape[1] - 1) / 4
	ty_quater = (vsm_data.shape[1] - 1) / 4

	xi1 = np.concatenate((vsm_data[0, 0:np.int_(deskew_points)], vsm_data[0, np.int_(vsm_data.shape[1] - 1 - deskew_points):np.int_(vsm_data.shape[1] - 1)]))

	yi1 = np.concatenate((vsm_data[1, 0:np.int_(deskew_points)], vsm_data[1, np.int_(vsm_data.shape[1] - 1 - deskew_points):np.int_(vsm_data.shape[1] - 1)]))

	# calculate linear fit parameters
	slope_deskew, intercept_deskew, r_value_deskew, p_value_deskew, std_err_deskew = stats.linregress(xi1,yi1)
	print(slope_deskew, intercept_deskew)

	xd1 = vsm_data[0]
	yd1 = vsm_data[1] - slope_deskew*vsm_data[0]
	return np.array([xd1, yd1]);

###################################### plot this  and that ############################################
def vsm_hst_plot(vsm_data, fit_params_Hc, fit_params_Ms):
	# plot the data
	# 'ro-' -> red circles with solid line
	#if np.size(fit_params_Hc) == None and np.size(fit_params_Ms) == None:
	#	tx1, = plt.plot(vsm_data[0], vsm_data[1], 'co-', label=label_hdir)
	#	plt.legend([tx1], [label_hdir], loc='upper left', fontsize=20 , frameon=False)
	#
	#elif np.size(fit_params_Hc) != None and np.size(fit_params_Ms) == None:
	#	tx1, = plt.plot(vsm_data[0], vsm_data[1], 'co-', label=label_hdir)
	#	tx2, = plt.plot(vsm_data[0], fit_params_Hc[0]*vsm_data[0] + fit_params_Hc[2], 'b--')
	#	tx3, = plt.plot(vsm_data[0], fit_params_Hc[1]*vsm_data[0] + fit_params_Hc[3], 'r--')
	#	plt.legend([tx1, tx2, tx3], [label_hdir, r'$fit$', r'$fit$'], loc='upper left', fontsize=20 , #frameon=False)

	#elif np.size(fit_params_Hc) == None and np.size(fit_params_Ms) != None:
	#	tx1, = plt.plot(vsm_data[0], vsm_data[1], 'co-', label=label_hdir)
	#	tx2, = plt.plot(vsm_data[0], fit_params_Ms[0]*vsm_data[0] + fit_params_Ms[2], 'g--')
	#	tx3, = plt.plot(vsm_data[0], fit_params_Ms[1]*vsm_data[0] + fit_params_Ms[3], 'm--')
	#	# display the legend for the defined labels
	#	plt.legend([tx1, tx2, tx3], [label_hdir, r'$fit$', r'$fit$'], loc='upper left', fontsize=20 , frameon=False)

	#elif np.size(fit_params_Hc) != None and np.size(fit_params_Ms) != None:
	tx1, = plt.plot(vsm_data[0], vsm_data[1], 'co-', label=label_hdir)
	tx2, = plt.plot(vsm_data[0], fit_params_Hc[0]*vsm_data[0] + fit_params_Hc[2], 'b-.')
	tx3, = plt.plot(vsm_data[0], fit_params_Hc[1]*vsm_data[0] + fit_params_Hc[3], 'r-.')
	tx4, = plt.plot(vsm_data[0], fit_params_Ms[0]*vsm_data[0] + fit_params_Ms[2], 'g-.')
	tx5, = plt.plot(vsm_data[0], fit_params_Ms[1]*vsm_data[0] + fit_params_Ms[3], 'm-.')
	# display the legend for the defined labels
	plt.legend([tx1, tx2, tx3, tx4, tx5], [label_hdir, r'$fit\;q23$', r'$fit\;q14$', r'$fit\;q12$', r'$fit\;q34$'], loc='lower right', fontsize=20 , frameon=True)

	# fitted line for the inplane loop
	# plt.plot(deskew_fit_range, fit_params[0]*deskew_fit_range + fit_params[1],'g-', label=label_inpl + 'fit')
	# set x,y limits
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	
	return;

# Create a new figure of size 8x8 inches, using 100 dots per inch
# protip: A4 is 8.3 x 11.7 inches
fig = plt.figure(figsize=(16, 16), dpi=72)
fig.canvas.set_window_title('hst_hcms_plot_' + sample_mdate + sample_id)
# verison name text
plt.figtext(0.84, 0.99, version_name, size=14)

spec = gridspec.GridSpec(ncols=1, nrows=1)

# load raw data from measuremnt files for sample_no: 3 
vsm_files_in = vsm_open_in(sample_mdate, sample_id, sample_no)
vsm_data_deskewed = vsm_data_deskew(vsm_files_in, deskew_points)

vsm_fit_params_Hc = vsm_fit_hst_Hc(vsm_data_deskewed, Hc_fit_points_q23, Hc_fit_points_q14, Hc_creep_q14, Hc_creep_q23)
vsm_fit_params_Ms = vsm_fit_hst_Ms(vsm_data_deskewed, Ms_fit_points_q12, Ms_fit_points_q34, Ms_creep_q12, Ms_creep_q34)
xy1 = fig.add_subplot(spec[0,0])
# set title of the plot 
plt_title = gen_plot_title(0)
# plot hysteresis loops and fited data

vsm_hst_plot(vsm_data_deskewed, vsm_fit_params_Hc, vsm_fit_params_Ms)
# format axis and add labels
vsm_plot_anno_Hc(vsm_fit_params_Hc)
vsm_plot_anno_Ms(vsm_fit_params_Ms)
custom_axis_formater(plt_title, axis_label_mu0H, axis_label_mu0M, xmin, xmax, ymin, ymax, xprec, yprec)

plt.subplots_adjust(left=0.15, bottom=0.15, wspace=0.0, hspace=0.0)
fig.tight_layout(pad=11.0, w_pad=0.0, h_pad=0.0)

# Write a pdf file with fig and close
pp = PdfPages(out_pdf)
pp.savefig(fig)
pp.close()

fig = plt.savefig(out_svg)

# Show plot preview
plt.show()
