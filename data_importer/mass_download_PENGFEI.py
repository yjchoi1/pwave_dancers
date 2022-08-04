'''
This file is used to massively download the data
update: 07/25/2018, add second into the event info
'''

# change directory
import os
main_dir = "E:\UCLA_research\IRIS\NorthCarolina_Eqk_mseed"
os.chdir(main_dir)

# import modules for writing a flatfile
import csv
import sys

# the networks in California
import numpy as np
networks_cal = np.array(['N4','NE','NM','NP','NQ','NW','OH','PE','PN','PO','SE','SS','US','WU'])

# get events
from obspy import UTCDateTime
from obspy import read_inventory
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from obspy.clients.fdsn import Client

from obspy.clients.fdsn.header import URL_MAPPINGS
URL_MAPPINGS.keys()

mdl = MassDownloader()
client = Client("IRIS")
starttime = UTCDateTime("2020-08-09")
endtime = UTCDateTime("2020-08-10")
# starttime = UTCDateTime("2011-08-01")  # the more recent recordings of NGA-W2 is 2011/07/06
# endtime = UTCDateTime("2012-12-31")

# starttime = UTCDateTime("2000-08-01")  # this is for South Napa earthquake sequence
# endtime = UTCDateTime("2017-09-01")

# the bounds for events, draw a rectangle on Wilber3 website
minlatitude_event = 36
maxlatitude_event = 37
minlongitude_event = -82
maxlongitude_event = -81.0
# a rectangle only for Napa region
# minlatitude_event = 38.10
# maxlatitude_event = 38.30
# minlongitude_event = -122.5
# maxlongitude_event = -122.0

# the bounds for stations, obtain from R library "map"
minlatitude_sta = 27
maxlatitude_sta = 46
minlongitude_sta = -93
maxlongitude_sta = -72
domain = RectangularDomain(minlatitude=minlatitude_sta, maxlatitude=maxlatitude_sta,
                           minlongitude=minlongitude_sta, maxlongitude=maxlongitude_sta)


Cal_cat = client.get_events(starttime=starttime, minmagnitude=4, endtime=endtime,
                            minlatitude=minlatitude_event, maxlatitude=maxlatitude_event,
                            minlongitude=minlongitude_event, maxlongitude=maxlongitude_event)

print(Cal_cat)


for i in range(len(Cal_cat)):
    # for i in range(50, len(Cal_cat)):
    ######## get event information
    event = Cal_cat[i]
    origins = str(event.origins)
    descriptions = str(event.event_descriptions)
    mags = str(event.magnitudes)

    # key words
    key_ori = np.array(['UTCDateTime', 'longitude', 'latitude', 'depth'])
    key_desp = 'type'
    key_mags = np.array(['mag', 'magnitude_type'])

    # check and extract origins info
    if key_ori[0] in origins:
        if key_ori[1] in origins:
            time = eval(origins.partition('UTCDateTime(')[-1].rpartition('), longitude')[0])
        elif key_ori[2] in origins:
            time = eval(origins.partition('UTCDateTime(')[-1].rpartition('), latitude')[0])
        elif key_ori[3] in origins:
            time = eval(origins.partition('UTCDateTime(')[-1].rpartition('), depth')[0])
        else:
            time = eval(origins.partition('UTCDateTime(')[-1].rpartition(', creation_info')[0])
    else:
        time = 'NA'

    if key_ori[1] in origins:
        if key_ori[2] in origins:
            longitude = float(origins.partition('longitude=')[-1].rpartition(', latitude')[0])
        elif key_ori[3] in origins:
            longitude = float(origins.partition('longitude=')[-1].rpartition(', depth')[0])
        else:
            longitude = float(origins.partition('longitude=')[-1].rpartition(', creation_info')[0])
    else:
        longitude = 'NA'

    if key_ori[2] in origins:
        if key_ori[3] in origins:
            latitude = float(origins.partition('latitude=')[-1].rpartition(', depth')[0])
        else:
            latitude = float(origins.partition('latitude=')[-1].rpartition(', creation_info')[0])
    else:
        latitude = 'NA'

    if key_ori[3] in origins:
        try:
            depth = float(origins.partition('depth=')[-1].rpartition(', creation_info')[0])
        except Exception as e:
            depth = float(origins.partition('depth=')[-1].rpartition(')]')[0])
    else:
        depth = 'NA'

    # check and extract region type info
    if key_desp in descriptions:
        region_type = eval(descriptions.partition('type=')[-1].rpartition(')]')[0])
    else:
        region_type = 'NA'

    # check and extract mag info
    if key_mags[0] in mags:
        if key_mags[1] in mags:
            magnitude = float(mags.partition('mag=')[-1].rpartition(', magnitude_type')[0])
        else:
            magnitude = float(mags.partition('mag=')[-1].rpartition(', creation_info')[0])
    else:
        magnitude = 'NA'

    if key_mags[1] in mags:
        try:
            mag_type = eval(mags.partition('magnitude_type=')[-1].rpartition(', creation_info')[0])
        except Exception as e:
            mag_type = eval(mags.partition('magnitude_type=')[-1].rpartition(', origin_id')[0])
    else:
        mag_type = 'NA'


    ########## get time series
    if len(time) == 7:
        t = UTCDateTime(time[0], time[1], time[2], time[3], time[4], time[5], time[6])
    else:
        t = UTCDateTime(time[0], time[1], time[2], time[3], time[4], time[5], 000000)


    ########## go through all networks
    for k in range(len(networks_cal)):

        os.chdir(main_dir)
        ########## set restrictions
        #### channel code
        #### the first letter is for Band/sample rate:
        # B-10~80Hz; S-10~80Hz; H-80~250Hz; E-80~250Hz; C-250~1000Hz; D-250~1000Hz; G-1000~5000Hz; F-1000~5000Hz;
        # we only use broadband, which includes B, H, C, F
        #### the second letter is for instrument:
        # H-High Gain Seismometer; L-Low Gain Seismometer; M-Mass Position Seismometer; N-Accelerometer;
        # we only use, H, L, N
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                            network=networks_cal[k], channel_priorities=["BH?"]) # broadband(10~80Hz), high gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                              network=networks_cal[k], channel_priorities=["BL?"]) # broadband(10~80Hz), low gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["BN?"]) # broadband(10~80Hz), accelerometer

        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["HH?"])  # broadband(80~250Hz), high gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["HL?"])  # broadband(80~250Hz), low gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["HN?"])  # broadband(80~250Hz), accelerometer

        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["CH?"])  # broadband(250~1000Hz), high gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["CL?"])  # broadband(250~1000Hz), low gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["CN?"])  # broadband(250~1000Hz), accelerometer

        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                              network=networks_cal[k], channel_priorities=["FH?"])  # broadband(1000~5000Hz), high gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["FL?"])  # broadband(1000~5000Hz), low gain
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["FN?"])  # broadband(1000~5000Hz), accelerometer

        restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
                                    network=networks_cal[k], channel_priorities=["H**"])  # high gain seismometer
        # restrictions = Restrictions(starttime=t - 60, endtime=t + 60 * 10, reject_channels_with_gaps=True,
        #                             network=networks_cal[k], channel_priorities=["?N?"])  # accelormeter

        mseed_storage = 'waveforms_event' + str(i+1) + '_network_' + networks_cal[k]
        stationxml_storage = 'station_event' + str(i+1) + '_network_' + networks_cal[k]
        mdl.download(domain, restrictions, mseed_storage=mseed_storage, stationxml_storage=stationxml_storage)

        ############ write event/station info into a spreadsheet, the info includes
        # event: start time, longitude, latitude, depth of hypocenter, region_type, magnitude, mag_type,
        year = time[0]
        month = time[1]
        day = time[2]
        hr = time[3]
        mins = time[4]
        sec = time[5]

        waveforms_dir = main_dir + '/' + mseed_storage
        try:
            os.chdir(waveforms_dir)

            # list = os.listdir(station_dir)
            # t_file = str(t).replace("-","")[0:8]
            # station_cur = networks_cal[k] + '*' + '__' + t_file + '*' + '.mseed'
            # os.chdir(main_dir)
            f = open("eventinfo.csv", 'wt')
            try:
                writer = csv.writer(f)
                writer.writerow(('Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Longitude', 'Latitude',
                                 'Depth(m)', 'Region_type', ' Magnitude', 'Mag_type'))
                writer.writerow((year, month, day, hr, mins, sec, longitude, latitude, depth, region_type,
                                 magnitude, mag_type))
                # for i in range(10):
                #     writer.writerow((i + 1, chr(ord('a') + i), '08/%02d/07' % (i + 1)))
            finally:
                f.close()
        except Exception as e:
            print(e)
            continue


        # # check if the data are available or not
        # try:
        #     # start time is 1min/60sec before the event starts, and end time is 1min/60sec after the event starts
        #     st1 = client.get_waveforms(networks_cal[k], "*", "*", "B*", t - 60, t + 60 * 10,
        #                                attach_response=True)  # 10~80Hz
        #
        #     ########### get time series
        #     for j in range(len(st1)):
        #         stream = st1[j]
        #         a = stream.data
        #         # stream.remove_response(output = 'ACC') # unit of accel is m/s
        #         # stream.plot()
        #         # acc = stream.data
        #         network = str(stream.stats.network)
        #         station = str(stream.stats.station)
        #         channel = str(stream.stats.channel)
        #         sampling_rate = stream.stats.sampling_rate
        #         npts = stream.stats.npts
        #
        #         ########### get station info
        #         # path1 = "/path/to/"
        #         # path2 = network + "_" + station
        #         # path3 = ".xml"
        #         # path = path1 + path2 + path3
        #         # inv = read_inventory(path, format='STATIONXML')
        #         inv = client.get_stations(network = 'IU', station = 'ANMO')
        #         sta = inv[0]
        #         cha = sta[0]
        #         sta_longitude = sta.longitude
        #         sta_latitude = sta.latitude
        #         sta_elevation = sta.elevation # unit is meter
        #
        #
        #         ############ create a file and save it
        #         filename = network + "_" + station + "_" + channel + "_" + str(stream.stats.starttime) + ".ascii"
        #         stream.write(filename, format='SH_ASC')
        #
        #         ############ put this recording into summary matrix
        # except Exception as e:
        #     print e
        #     continue
        #
        # # different sampling rate
        # try:
        #     st2 = client.get_waveforms(networks_cal[k], "*", "*", "H*", t - 60, t + 60 * 10,
        #                                attach_response=True)  # 80~250Hz
        # except Exception as e:
        #     print e
        #     continue
        # # st3 = client.get_waveforms("CI", "*", "*", "S*", t, t + 60 * 10, attach_response=True) # 10~80Hz
        # # st4 = client.get_waveforms("CI", "*", "*", "E*", t, t + 60 * 10, attach_response=True) # 80~250Hz
        # # st5 = client.get_waveforms("CI", "*", "*", "C*", t, t + 60 * 10, attach_response=True) # 250~1000Hz
        # # st6 = client.get_waveforms("CI", "*", "*", "D*", t, t + 60 * 10, attach_response=True) # 250~1000Hz
        # # st7 = client.get_waveforms("CI", "*", "*", "G*", t, t + 60 * 10, attach_response=True) # 1000~5000Hz
        # # st8 = client.get_waveforms("CI", "*", "*", "F*", t, t + 60 * 10, attach_response=True) # 1000~5000Hz
        #















