# -*- coding: utf-8 -*-
'''
This script refines raw data from a weather balloon. The
different data streams are separated and converted to physical units, stored,
filtered, processed and presented graphically for preliminary analysis.

The code presented below is meant to serve as a starting point for customized
data analysis, and should be expanded upon and edited as needed.

Andøya Space Education

Created on Tue Jan 25 2022 at 09:39:00.
Last modified [dd.mm.yyyy]: 07.02.2022
@author: bjarne.ådnanes.bergtun
(Partly based on an earlier MATLAB code.
The idea of using the tkinter file dialog solution is due to
odd-einar.cedervall.nervik)
'''

import tkinter as tk # GUI
from tkinter import filedialog as fd # file dialogs
import dask.dataframe as dd # import of data
import matplotlib.pyplot as plt # plotting
import numpy as np # maths
import pandas as pd # data handling


# Setup of imported libraries

pd.options.mode.chained_assignment = None
plt.style.use('seaborn')


###############################################################################
########################### User defined parameters ###########################

# Logical switches.
# If using Spyder, you can avoid needing to load the data every time by going
# to "Run > Configuration per file" and activating the option "Run in console's
# namespace instead of an empty one".

load_data = True
CRC_filter = 3 # Only rows with CRC >= CRC_filter are loaded
sanitize_GPS = True # If true, satellite number is used to clean the GPS-data
convert_data = True
process_data = True
plot_data = True
export_data = False
export_kml = False


# Quantization parameters

U_main = 3.3
standard_wordlength = 12
standard_dt = 4 # [s]


# NTC parameters
# If using an NTC with a 1k reference resistance above 25 deg. C, set R_NTC
# equal to the string '1k_warm'; otherwise R_NTC should be the NTC reference
# resistance in ohms, e.g. R_NTC_ext = 470.

R_fixed_ext = 1e4
R_NTC_ext = 470

R_fixed_int = 2e4
R_NTC_int = 1e4


# Antenna position parameters (pro tip: these can be found using Google Maps
# or a GPS unit). The antenna is assumed to be stationary.

antenna_lat = 69.29601298387487
antenna_long = 16.03062325486236
antenna_height = 9 # height above ground in m.


# Channel spesification & naming.
# The channels will be named according to this list, so both numbering and
# ordering of items must be in accordance with the data file!

channels = [
    't',
    'framecounter',
    'lat',
    'long',
    'height',
    'GPS_satellites',
    'temp',
    'voltage',
    'temp_int',
    'pressure',
    'humidity_rel',
    'RSSI',
    'CRC'
    ]

###############################################################################
################################ Load CSV data ################################

if load_data:

    # First a root window is created and put on top of all other windows.

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    # On top of the root window, a filedialog is opened to get the CSV file
    # from the file explorer.

    data_file = fd.askopenfilename(
        title = 'Select weather balloon data to import',
        filetypes = (
            ('Log files','*.log'),
            ('Data files','*.dat'),
            ('All files','*.*')
            ),
        parent = root)

    # Use dask to load the file, saving only the lines with t >= t_0 into
    # memory, and only the colons listed in the channels dictionary defined
    # above.

    raw_data = dd.read_csv(
        data_file,
        sep = ',',
        header = 0,
        names = channels,
        dtype = 'float64',
        na_values = ' nan'
        )
    raw_data = raw_data[raw_data['CRC']>=CRC_filter]
    raw_data = raw_data.compute()

    # Sanitize GPS-data
    if sanitize_GPS:
        mask = raw_data['GPS_satellites'] < 3
        mask2 = raw_data['GPS_satellites'] == 3
        raw_data.loc[mask, ['lat', 'long', 'height']] = np.nan
        raw_data.loc[mask2, 'height'] = np.nan

    # For the sake of plotting, we want to insert one nan-value every time we
    # have missing data. Since the PTU is sending at a regular interval
    # (specified above as standard_dt), missing data can be identified by
    # looking at the change in time, t.diff().
    dt = raw_data['t'].diff()

    dt_max = standard_dt*1.5 # Let's introduce a bit of tolerance

    # Copy the lines where missing data needs to be inserted.
    missing_data = raw_data.loc[dt > dt_max].copy()

    # Create column of missing data, and fill the missing data to be inserted
    nan_column = np.full(len(missing_data['t']), np.nan)
    for i in channels:
        if i != 't':
            missing_data[i] = nan_column

    # When dt is larger than standard_dt, this means that a missing data line
    # needs to be inserted *before* the current line.
    missing_data['t'] -= standard_dt
    missing_data.index -= 0.5

    # Insert the missing data, and sort according to the index.
    raw_data = raw_data.append(missing_data).sort_index()

    # The indices needs to be reset for iloc[] to work as expected.
    raw_data.reset_index(drop=True, inplace=True)


###############################################################################
################################ Convert data #################################

if convert_data:

    print('Pre-processing ...')

    # Physical constants

    T_0 = 273.15 # 0 celsius degrees in kelvin


    # Steinhart--Hart parameter dictionary. This dictionary simplifies the
    # needed user input when using different NTC-sensors. Make sure it is up to
    # date!

    SH_dictionary = {
        10e3: (
            3.354016e-3,
            2.569850e-4,
            2.620131e-6,
            6.383091e-8
            ),
        1e3: (
            3.354016e-3,
            2.909670e-4,
            1.632136e-6,
            7.192200e-8
            ),
        '1k_warm': (
            3.354016e-3,
            2.933908e-4,
            3.494314e-6,
            -7.71269e-7
            ),
        470: (
            3.354075e-3,
            2.972812e-4,
            3.629995e-6,
            1.977197e-7
            )
        }


    # Conversion formulas

    def volt(bit_value, wordlength, U=U_main):
        Z = 2**wordlength - 1
        return U*bit_value/Z

    def NTC(U_bit, R_ref, R_fixed, wordlength=standard_wordlength): # unit: celsius degrees
        A_1, B_1, C_1, D_1 = SH_dictionary[R_ref]
        if R_ref == '1k_warm':
            R_rel = R_fixed/1000
        else:
            R_rel = R_fixed/R_ref
        R = R_rel*U_bit/(2**wordlength-1-U_bit)
        R[R<=0] = np.nan # avoids complex logarithms
        ln_R = np.log(R)
        T = 1/(A_1 + B_1*ln_R + C_1*ln_R**2 + D_1*ln_R**3) - T_0
        return T

    def power(U_bit): # unit: volts
        U = volt(U_bit,8)
        return 2*U


    # Convert data units

    processed_data = raw_data.copy()

    processed_data['t'] = raw_data['t'] - raw_data['t'].iloc[0]

    processed_data['height'] = raw_data['height']/1000 # Converts to km
    processed_data['pressure'] = raw_data['pressure']/1000 # Converts to kPa

    processed_data['temp'] = NTC(
        raw_data['temp'],
        R_NTC_ext,
        R_fixed_ext)
    processed_data['temp_int'] = NTC(
        raw_data['temp_int'],
        R_NTC_int,
        R_fixed_int,
        wordlength=8)
    processed_data['voltage'] = power(raw_data['voltage'])


###############################################################################
############################### Processes data ################################

if process_data:

    print('Processing ...')

    ############################### Utility ###############################

    # Smoothing function

    def smooth(data, r_tol=0.02):
        """
        Smooths a data series by requiring that the change between one datapoint and the next is below a certain relative treshold (the default is 2 % of the total range of the values in the dataseries). If this is not the case, the value is replaced by NaN. This gives a 'rhougher' output than a more sophisticated smoothing algorithm (see for example statsmodels.nonparametric.smoothers_lowess.lowess from the statmodels module), but has the advantage of being very quick. If more sophisticated methods are needed, this algorithm can be used to trow out obviously erroneous data before the data is sent through a more traditional filter.

        Parameters
        ----------
        data : pandas.DataSeries
            The data series to be smoothed

        r_tol : float, optional
            Tolerated change between one datapoint and the next, relative to the full range of values in DATA. The default is 0.02.

        Returns
        -------
        data_smooth : pandas.DataSeries
            Smoothed data series.

        """
        print('Smoothing data ...')
        valid_data = data[data.notna()]
        if len(valid_data) == 0:
            data_range = np.inf
        else:
            data_range = np.ptp(valid_data)
        tol = r_tol*data_range
        data_smooth = data.copy()
        data_interpol = data_smooth.interpolate()
        data_smooth[np.abs(data_interpol.diff()) > tol] = np.nan
        return data_smooth


    ############ Coordinate transformations & antenna distance ############

    # Constants used in WGS 84

    r_a = 6378137. # Semi-major axis in meters.
    r_b = 6356752.314245 # Approximate semi-minor axis in meters.


    # Step 0: Convert angles to radians, as this makes for easier calculations

    processed_data['lat'] = np.radians(raw_data['lat'])
    processed_data['long'] = np.radians(raw_data['long'])

    antenna_lat = np.radians(antenna_lat)
    antenna_long = np.radians(antenna_long)


    # Step 1: Find spherical coordinates relative Earth's centre

    def r_E(latitude, height): # height must be in meters!
        r_a_cos = r_a * np.cos(latitude)
        r_b_sin = r_b * np.sin(latitude)
        return np.sqrt(r_a_cos**2 + r_b_sin**2) + height # distance to Earth's centre in meters


    # Step 2: Convert from spherical to cartesian coordinates, with axes
    # oriented such that conversion is easy -- z pointing north and x pointing
    # from the center of the Earth to the Greenwich meridian.

    def cartesian(latitude,longitude, height): # height must be in meters!
        r = r_E(latitude, height)
        r_cos = r * np.cos(latitude)
        x = r_cos * np.cos(longitude)
        y = r_cos * np.sin(longitude)
        z = r * np.sin(latitude)
        return (x, y, z)

    x_antenna, y_antenna, z_antenna = cartesian(
        antenna_lat, # radians
        antenna_long, # radians
        antenna_height # meters
        )


    # Step 3: Calculate distance using pytagoras

    def antenna_distance(latitude,longitude,height): # height must be in meters!
        x, y, z = cartesian(latitude,longitude, height)
        Dx = x - x_antenna
        Dy = y - y_antenna
        Dz = z - z_antenna
        r = np.sqrt(Dx**2 + Dy**2 + Dz**2)
        return r


    processed_data['r'] = antenna_distance(
        processed_data['lat'], # radians
        processed_data['long'], # radians
        raw_data['height'] # meters
        )/1000 # Unit: km


    ####################### Speed, wind & direction #######################

    # When calculating wind speed, we are interested in the movement parallel
    # and perpendicular to the Earth's surface. Hence, we will be using
    # localized coordinates, with the z-axis always pointing upwards (towards
    # increasing height), and the x-axis pointing northwards (towards incre-
    # asing latitude). Hence, the y-axis will be pointing westwards (towards
    # decreasing longitude).
    # Obviously, this choice of directions will not work at the poles, but it
    # is highly unlikely that our weather balloon will end up at either pole.

    # Useful constants

    tau = np.pi*2


    # We start with calculating the horizontal speed, as this is straigth-
    # forward.

    def speed(position, time):
        dx = position.diff()
        dt = time.diff()
        return dx / dt

    processed_data['v_z'] = speed(
        raw_data['height'], # meters
        processed_data['t'] # seconds
        )


    # Then we turn to the wind.

    # To get sensible plots of wind direction,* we will need a function to
    # remove discontinuities caused by the multi-valued nature of arctan().
    # This can be done by ensuring that the "jump" from one measurement to the
    # next never exceeds pi.
    #
    # * This is only needed if we use cartesian plots. If we instead use
    #   spherical plots, this sanitizing is not necessary.

    def sanitize_wind_direction(direction_data):
        # increments = direction_data.diff()
        # for i in np.arange(len(direction_data)):
        #     if increments.iloc[i] > np.pi:
        #         direction_data.iloc[i] -= tau
        #         increments.iloc[i+1] += tau
        #     elif increments.iloc[i] < -np.pi:
        #         direction_data.iloc[i] += tau
        #         increments.iloc[i+1] -= tau
        return direction_data

    # The wind direction is defined according to where the wind is coming
    # from, with 0 degrees being North, and 90 degrees being East.
    # Hence the direction is opposite of the direction of travel, and measured
    # clockwise rather than anti-clockwise.
    # The wind direction is measured in radians, while it's speed is in m/s.

    def horizontal_speed(latitude,longitude,height,time): # height must be in meters!
        r = r_E(latitude,height)
        dt = time.diff()
        dphi = latitude.diff()
        dtheta = longitude.diff()
        dx = r * dphi
        dy = -r * dtheta
        speed = np.sqrt(dx**2 + dy**2)/dt
        direction = sanitize_wind_direction(-np.arctan2(-dy,-dx))
        return (speed, direction)

    processed_data['wind'], processed_data['wind_direction'] = horizontal_speed(
        processed_data['lat'], # radians
        processed_data['long'], # radians
        raw_data['height'], # meters
        processed_data['t'] # seconds
        )


    ################### Other atmospherical calculations ##################

    # Constants used in the Arden Buck equation

    a = 6.1115 # Unit: hPa
    b = 23.036
    c = 279.82 # Unit: degrees celsius
    d = 333.7 # Unit: degrees celsius

    # Auxillary function used in the Arden Buck equation
    def temp_term(temp):
        return (b-temp/d)*(temp/(c+temp))

    # Vapor pressure
    def vapor_pressure(temp, hum_rel):
        return a*hum_rel*np.exp(temp_term(temp))/1000 # Unit: kPa

    # Dew point
    def dew_point(temp, hum_rel):
        gamma = np.log(hum_rel/100) + temp_term(temp)
        return c*gamma/(b-gamma)


    processed_data['vapor_pressure'] = vapor_pressure(
        processed_data['temp'],
        processed_data['humidity_rel']
        )

    processed_data['dew_point'] = dew_point(
        processed_data['temp'],
        processed_data['humidity_rel']
        )


    ###################### Link analysis (telemetry) ######################

    # Usefull constants
    f = 433.5e6  # hz
    c = 299792.458  # kps
    P_tx = 20  # dBm
    G = -5  # dB

    R_min = 0.01  # km
    R_max = 20  # km
    n = int(1e4)  # Measurments

    R = np.linspace(R_min, R_max, n)

    L_0 = 20 * np.log10(4 * np.pi * R * f / c)

    P_rx = P_tx + G - L_0

    ###################### Preparation for plotting #######################

    # Identify the point at which the balloon bursts, and split the data set
    # accordingly

    burst_index = raw_data['height'].idxmax()

    up_data = processed_data.iloc[:burst_index]
    down_data = processed_data.iloc[burst_index:]


###############################################################################
################################## Plot data ##################################

if plot_data:

    # Close any plots left open from earlier sessions.

    plt.close('all')
    print('Plotting ...')

    plt.rcParams['legend.frameon'] = 'True'

    ###################### Custom plotting functions ######################

    # Custom parameters

    standard_linewidth = 0.5

    # First some auxillary functions containing some often-needed lines of
    # code for custom plots.

    def plot_data(x, y, label=''):
        if label != '':
            down_label = label + ' (descending)'
            up_label = label + ' (ascending)'
        else:
            down_label = 'Descending'
            up_label = 'Ascending'
        down, = plt.plot(
            down_data[x],
            down_data[y],
            'r-',
            linewidth=standard_linewidth,
            )
        up, = plt.plot(
            up_data[x],
            up_data[y],
            'b-',
            linewidth=standard_linewidth,
            )
        plots = [up, down]
        labels = [up_label, down_label]
        return plots, labels


    # Similar to plot_data(), but this function allows plotting two data sets.
    # Data set 2 will be plotted behind data set 1, but the legend will order
    # the data sets according to whether the balloon is going up- or downwards.
    def plot_dataX2(data1_label, x1, y1, data2_label, x2, y2):
        down1_label = data1_label + ' (descending)'
        down2_label = data2_label + ' (descending)'
        up1_label = data1_label + ' (ascending)'
        up2_label = data2_label + ' (ascending)'
        down2, = plt.plot(
            down_data[x2],
            down_data[y2],
            'c-',
            linewidth=standard_linewidth,
            )
        up2, = plt.plot(
            up_data[x2],
            up_data[y2],
            'g-',
            linewidth=standard_linewidth,
            )
        down1, = plt.plot(
            down_data[x1],
            down_data[y1],
            'r-',
            linewidth=standard_linewidth,
            )
        up1, = plt.plot(
            up_data[x1],
            up_data[y1],
            'b-',
            linewidth=standard_linewidth,
            )
        plots = [up1, up2, down1, down2]
        labels = [up1_label, up2_label, down1_label, down2_label]
        return plots, labels


    # Plot labels and the legend, and reveal the figure
    def finalize_figure(x_label, y_label, plots, plot_labels):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(
            plots,
            plot_labels,
            facecolor = 'white',
            framealpha = 1
            )
        plt.tight_layout()
        plt.show()


    # This is a single function providing a simple interface for standard
    # graphs, as well as serving as an example of how the auxillary functions
    # above might be utilized.

    def plot_graph(figure_name, x, y, x_label, y_label, data_label=''):
        plt.figure(figure_name)
        data_plots, data_labels = plot_data(
            x,
            y,
            label = data_label
            )
        finalize_figure(
            x_label,
            y_label,
            data_plots,
            data_labels
            )


    ########################### Specific plots ############################

    # Vertical speed v. height

    plt.figure('Vertical speed')
    line = plt.axvline(
        x = 5,
        color = 'k',
        linestyle = '--',
        linewidth = 0.85
        )
    data_plot, data_label = plot_data(
        'v_z',
        'height'
        )
    plots = data_plot + [line]
    plot_labels = data_label + ['Nominal ascending speed']
    finalize_figure(
        'Vertical speed [m/s]',
        'Height [km]',
        plots,
        plot_labels
        )


    # Wind speed v. height

    plot_graph(
        'Wind speed',
        'wind',
        'height',
        'Wind speed [m/s]',
        'Height [km]'
        )


    # Wind direction v. height
    # We will use a fancy polar plot for this.

    plt.figure('Wind direction')
    ax = plt.subplot(111, polar=True)
    data_plots, data_labels = plot_data(
        'wind_direction',
        'height',
        label = 'Wind direction'
        )
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        (0, 30, 60,
         90, 120, 150,
         180, 210, 240,
         270, 300, 330),
        labels = (
            'N',
            u'30\N{DEGREE SIGN}',
            u'60\N{DEGREE SIGN}',
            'E',
            u'120\N{DEGREE SIGN}',
            u'150\N{DEGREE SIGN}',
            'S',
            u'210\N{DEGREE SIGN}',
            u'240\N{DEGREE SIGN}',
            'W',
            u'300\N{DEGREE SIGN}',
            u'330\N{DEGREE SIGN}'
            )
        )
    legend_angle = np.radians(53)
    plt.legend(
        data_plots,
        data_labels,
        facecolor = 'white',
        framealpha = 1,
        loc = 'lower left',
        bbox_to_anchor = (
            0.555 + np.cos(legend_angle)/2,
            0.555 + np.sin(legend_angle)/2
            )
        )
    plt.tight_layout()
    plt.show()


    # Pressure v. height

    plot_graph(
        'Pressure',
        'pressure',
        'height',
        'Pressure [kPa]',
        'Height [km]'
        )


    # Temperature v. height

    plt.figure('Temperature')
    data_plots, data_labels = plot_dataX2(
        'Temperature',
        'temp',
        'height',
        'Dew point',
        'dew_point',
        'height'
        )
    finalize_figure(
        u'Temperature [\N{DEGREE SIGN}C]',
        'Height [km]',
        data_plots,
        data_labels
        )


    # Humidity v. height

    plot_graph(
        'Humidity',
        'humidity_rel',
        'height',
        'Relative humidity [%]',
        'Height [km]'
        )


    # Internal temperature v. height

    plot_graph(
        'Internal temperature',
        'temp_int',
        'height',
        u'Internal temperature [\N{DEGREE SIGN}C]',
        'Height [km]'
        )


    # Battery

    plot_graph(
        'Battery',
        't',
        'voltage',
        '$t$ [s]',
        'Battery voltage [V]'
        )


    # Frame counter

    plot_graph(
        'Frame counter',
        't',
        'framecounter',
        '$t$ [s]',
        'Frame number'
        )


    # Distance v. time

    plot_graph(
        'PTU–antenna distance',
        't',
        'r',
        '$t$ [s]',
        'Distance between PTU and antenna [km]'
        )


    # RSSI:

    plt.figure('RSSI')
    theoretical_plot, = plt.plot(
        R,
        P_rx,
        'k--',
        linewidth = 0.85
        )
    data_plot, data_label = plot_data(
        'r',
        'RSSI'
        )
    plots = data_plot + [theoretical_plot]
    plot_labels = data_label + ['Theoretical curve']
    finalize_figure(
        'Distance between PTU and antenna [km]',
        'Received power [dBm]',
        plots,
        plot_labels
        )


###############################################################################
############################ Export processed data ############################

# Much like before, a filedialog is opened, this time to allow the user to
# specify the name and storage location of the processed data.
# The processed data is stored using Pandas' .to_csv()

if export_data:
    print('Exporting processed data ...')

    datafile = fd.asksaveasfilename(
        title = 'Save the processed data as ...',
        filetypes = (('CSV files','*.csv'),),
        defaultextension = (('CSV files','*.csv'),),
        parent = root)

    processed_data.to_csv(
        datafile,
        sep = ';',
        decimal = ',',
        index = False)


# Create and export a kml-file which can be opened in Google Earth.

if export_kml:
    print('Exporting kml ...')

    # kml-coordinates needs to be in degrees for longitude and latitude, and
    # meters for the height. Hence, we will take our data from raw_data:
    notna_indices = (
        raw_data['lat'].notna() &
        raw_data['long'].notna() &
        raw_data['height'].notna()
        )
    kml_lat = raw_data['lat'][notna_indices].copy().to_numpy()
    kml_long = raw_data['long'][notna_indices].copy().to_numpy()
    kml_height = raw_data['height'][notna_indices].copy().to_numpy()

    # To avoid having to install a kml-library, we will instead (ab)use a numpy
    # array and savetxt()-function to save our kml file.
    # Unfortunately, this means that we need to hard-code the kml-file ...
    kml_header = (
'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Document>
<name>Paths</name>
<description> Weather balloon path. Note that the tessellate tag is by
default set to 0. If you want to create tessellated lines, they must be
authored (or edited) directly in KML.</description>
<Style id="yellowLineGreenPoly">
<LineStyle>
<color>7f00ffff</color>
<width>4</width>
</LineStyle>
<PolyStyle>
<color>7f00ff00</color>
</PolyStyle>
</Style>
<Placemark>
<name>Absolute Extruded</name>
<description>Transparent green wall with yellow outlines</description>
<styleUrl>#yellowLineGreenPoly</styleUrl>
<LineString>
<extrude>0</extrude>
<tessellate>0</tessellate>
<altitudeMode>absolute</altitudeMode>
<coordinates>''')

    kml_body = np.array([kml_long, kml_lat, kml_height]).transpose()

    kml_footer = (
'''</coordinates>
</LineString>
</Placemark>
</Document>
</kml>''')

    datafile = fd.asksaveasfilename(
        title = 'Save the kml data as ...',
        filetypes = (('KML files','*.kml'),),
        defaultextension = (('KML files','*.kml'),),
        parent = root)

    np.savetxt(
        datafile,
        kml_body,
        fmt = '%.6f',
        delimiter = ',',
        header = kml_header,
        footer = kml_footer,
        comments = ''
        )