# Import functions and libraries
from __future__ import division
import numpy as np, matplotlib.pyplot as plt
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from rtlsdr import RtlSdr
import threading,time, queue

from bokeh.plotting import figure, show
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Oval, Text, Circle
from bokeh.tile_providers import CARTODBPOSITRON
import math
from IPython.display import clear_output
import sys


output_notebook()

def bit2byte( bits ):
    msg = np.zeros( MODES_LONG_MSG_BYTES, dtype='int' )

    # Pack bits into bytes */
    for i in r_[:MODES_LONG_MSG_BITS:8]:
        for j in r_[:8]:
            msg[ i // 8] = msg[ i // 8] + (int(bits[i+j]) << (7-j))
    return msg


class Plane:
    addr = -1
    flightnum = 'UNKNOWN'
    position = (-1,-1)
    planetype = 0
    lat0 = 0
    lat1 = 0
    lon0 = 0
    lat0 = 0
    time0 = -1
    time1 = -1
    heading = 0
    
    
    def __init__( self, addr):
        self.addr = addr;
        
    def addplanetype( self, planetype ):
        self.planetype = planetype;
    
    def addflightnum( self, flightnum ):
        self.flightnum = flightnum;
        
    def addposition( self, position ):
        self.position = position;
        
MODES_LONG_MSG_BITS = 112
MODES_SHORT_MSG_BITS = 56
MODES_LONG_MSG_BYTES  = (112//8)
MODES_SHORT_MSG_BYTES = (56//8)


modes_checksum_table = [
0x3935ea, 0x1c9af5, 0xf1b77e, 0x78dbbf, 0xc397db, 0x9e31e9, 0xb0e2f0, 0x587178,
0x2c38bc, 0x161c5e, 0x0b0e2f, 0xfa7d13, 0x82c48d, 0xbe9842, 0x5f4c21, 0xd05c14,
0x682e0a, 0x341705, 0xe5f186, 0x72f8c3, 0xc68665, 0x9cb936, 0x4e5c9b, 0xd8d449,
0x939020, 0x49c810, 0x24e408, 0x127204, 0x093902, 0x049c81, 0xfdb444, 0x7eda22,
0x3f6d11, 0xe04c8c, 0x702646, 0x381323, 0xe3f395, 0x8e03ce, 0x4701e7, 0xdc7af7,
0x91c77f, 0xb719bb, 0xa476d9, 0xadc168, 0x56e0b4, 0x2b705a, 0x15b82d, 0xf52612,
0x7a9309, 0xc2b380, 0x6159c0, 0x30ace0, 0x185670, 0x0c2b38, 0x06159c, 0x030ace,
0x018567, 0xff38b7, 0x80665f, 0xbfc92b, 0xa01e91, 0xaff54c, 0x57faa6, 0x2bfd53,
0xea04ad, 0x8af852, 0x457c29, 0xdd4410, 0x6ea208, 0x375104, 0x1ba882, 0x0dd441,
0xf91024, 0x7c8812, 0x3e4409, 0xe0d800, 0x706c00, 0x383600, 0x1c1b00, 0x0e0d80,
0x0706c0, 0x038360, 0x01c1b0, 0x00e0d8, 0x00706c, 0x003836, 0x001c1b, 0xfff409,
0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000
]

        
        
def modesChecksum(msg, bits):
    crc = 0;

    if (bits == 112):
        offset = 0
    else:
        offset = 112 - 56

    for j in r_[:bits]:

        byte = j//8;
        bit = j%8;
        bitmask = 1 << (7 - bit);

        # If bit is set, xor with corresponding table entry.
        if (msg[byte] & bitmask):
            crc ^= modes_checksum_table[j+offset];
    
    return crc; # 24 bit checksum. 


def modesMessageLenByType( type ):
    if ( type == 16 or type == 17 or type == 19 or type == 20 or type == 21):
        return MODES_LONG_MSG_BITS
    else:
        return MODES_SHORT_MSG_BITS
    
def fixSingleBitErrors( msg, bits ):
    for j in r_[:8]:
        byte = j // 8
        bitmask = 1 << (7 - (j%8))
        
        aux = msg[:bits//8]
        
        crc1 = (aux[(bits//8)-3] << 16) |(aux[(bits//8)-2] << 8) |aux[(bits//8)-1];
        crc2 = modesChecksum(aux,bits)
        
        crcok = (crc1 == crc2)
        if ( crcok ):
            for i in r_[:bits//8]:
                msg[i] = aux[i]
                
        return crcok
    
def NL( rlat ):
    # A.1.7.2.d (page 9)
    NZ = 15
    return np.floor( 2 * np.pi / 
                    (np.arccos( 1 - (1 - np.cos( np.pi / (2 * NZ )) ) 
                               / np.cos( np.pi / 180 * abs(rlat) ) ** 2 )))


def cprN( lat,  isodd):
    nl = NL(lat) - isodd;
    if (nl < 1):
        nl = 1;
    return nl;

def Dlon( lat,  isodd):
    return 360.0 / cprN(lat, isodd)

        
def cprmod( a, b ):
    res = a % b;
    if (res < 0):
        res = res + b;
    return res;


def decodeCPR( plane ):
        AirDlat0 = 360.0 / 60
        AirDlat1 = 360.0 / 59
            
        lat0 = plane.lat0
        lat1 = plane.lat1
        lon0 = plane.lon0
        lon1 = plane.lon1
            
        j = np.floor(((59*lat0 - 60*lat1) / 131072) + 0.5)
            
        rlat0 = AirDlat0 * (cprmod(j,60) + lat0 / 131072)
        rlat1 = AirDlat1 * (cprmod(j,59) + lat1 / 131072)
            
        if (rlat0 >= 270):
            rlat0 = rlat0 - 360
                
        if (rlat1 >= 270):
            rlat1 = rlat1 - 360
                
        if (NL(rlat0) != NL(rlat1)):
            return;
            
        if (plane.time0 > plane.time1) :

            # Use even packet.
            ni = cprN(rlat0,0);
            m = np.floor((((lon0 * (NL(rlat0)-1)) -
                        (lon1 * NL(rlat0))) / 131072) + 0.5);
            lon = Dlon(rlat0,0) * (cprmod(m,ni)+lon0/131072);
            lat = rlat0;
        else:
            # Use odd packet
            ni = cprN(rlat1,1);
            m = np.floor((((lon0 * (NL(rlat1)-1)) - 
                        (lon1 * NL(rlat1))) / 131072) + 0.5);
            lon = Dlon(rlat1,1) * (cprmod(m,ni)+lon1/131072);
            lat = rlat1;
        if ( lon > 180 ):
            lon = lon - 360;
            
        plane.addposition( (lat, lon) )

def decodeModesMessage( msg, plane_list, log ):
    
    ais_charset = np.array( list("?ABCDEFGHIJKLMNOPQRSTUVWXYZ????? ???????????????0123456789??????"))
    
    #  Get the message type ASAP as other operations depend on this
    msgtype = msg[0] >> 3
    msgbits = modesMessageLenByType(msgtype)
    
    
    # Get checksum.  CRC is always the last three bytes.
    crc = (msg[(msgbits//8)-3] << 16) |  (msg[(msgbits//8)-2] << 8) | msg[(msgbits//8)-1];
    crc2 = modesChecksum(msg,msgbits)
    crcok = (crc == crc2)
    
    # Correct 1-bit error
    if (not crcok):
        crcok = fixSingleBitErrors( msg, msgbits )
    
        
    # ICAO address ( airplane address )
    aa1 = msg[1]
    aa2 = msg[2]
    aa3 = msg[3]
    
    
    # Get DF 17 (ADSB) extended squitter types
    metype = msg[4] >> 3 # extended squitter message type
    mesub = msg[4] & 7
    
    
    # Decode extended squitter, ignore all other messages
    if ( (msgtype == 17) and crcok ):
        
        strp = time.strftime("%H:%M:%S", time.localtime()) + " found DF-17 packet\n"
        sys.__stdout__.write(strp)
        log.write(strp)
        
        addr = (aa1 << 16 ) | (aa2 << 8) | aa3
        
        # Add plane address to plane_list
        if ( addr in plane_list ):
            plane = plane_list[addr]
        else:
            plane = Plane(addr)
            plane_list[addr] = plane
            
            strp = time.strftime("%H:%M:%S", time.localtime()) + " found new plane (ICAO: %x)!\n" % addr
            sys.__stdout__.write(strp)
            log.write(strp)
            if ( len(plane_list) == 1):
                print("Found %d plane" % len(plane_list))
                
            else:
                print("Found %d planes" % len(plane_list))
        
        if ( metype >=1 and metype <= 4 ):
            aircraft_type = metype - 1
            flight_index = np.array( [ msg[5] >> 2,
                            ((msg[5]&3)<<4)|(msg[6]>>4),
                            ((msg[6]&15)<<2)|(msg[7]>>6),
                            msg[7]&63,
                            msg[8]>>2,
                            ((msg[8]&3)<<4)|(msg[9]>>4),
                            ((msg[9]&15)<<2)|(msg[10]>>6),
                            msg[10]&63] )
            
            flightnum = ais_charset[ flight_index ]
            
            strp = time.strftime("%H:%M:%S", time.localtime()) + " found flight number %s for plane %x!\n" % (''.join(flightnum), addr)
            
            sys.__stdout__.write(strp)
            log.write(strp)
            
            plane.addplanetype( aircraft_type )
            plane.addflightnum( "".join( flightnum ) )
        elif ( metype >= 9 and metype <= 18 ):
            
            # latitude and longitude are in CPR format
            # here we implement the global decoding described in section 5.3.1:
            # http://adsb.tc.faa.gov/WG3_Meetings/Meeting29/1090-WP29-07-Draft_CPR101_Appendix.pdf
            # see also:
            # https://sites.google.com/site/adsbreceiver/
            # http://www.lll.lu/~edward/edward/adsb/DecodingADSBposition.html
            # http://aviation.stackexchange.com/questions/3707/ads-b-compact-position-report-nl-function
            
            
            isodd = msg[6] & (1<<2);
            lat_enc = ((msg[6] & 3) << 15) | (msg[7] << 7) | (msg[8] >> 1); 
            lon_enc = ((msg[8] & 1) << 16) | (msg[9] << 8) | msg[10];
            
            if (isodd):
                plane.lat1 = lat_enc
                plane.lon1 = lon_enc
                plane.time1 = time.time()
                
                if (plane.time0 == -1):
                    return
            else:
                plane.lat0 = lat_enc
                plane.lon0 = lon_enc
                plane.time0 = time.time()
                
                if (plane.time1 == -1):
                    return
            
            if ( abs( plane.time0 - plane.time1 ) <= 10):
                decodeCPR( plane )
                strp = time.strftime("%H:%M:%S", time.localtime()) + " found position (%f,%f) for plane %x!\n" % (plane.position[0], plane.position[1], addr)
                sys.__stdout__.write(strp)
                log.write(strp)
                sys.stdout.flush()
                
        elif ( metype == 19 and mesub >=1 and mesub <= 4 ):
            if  ( mesub == 1 or mesub == 2):
                ew_dir = (msg[5]&4) >> 2;
                ew_velocity = ((msg[5]&3) << 8) | msg[6];
                ns_dir = (msg[7]&0x80) >> 7;
                ns_velocity = ((msg[7]&0x7f) << 3) | ((msg[8]&0xe0) >> 5);
                vert_rate_source = (msg[8]&0x10) >> 4;
                vert_rate_sign = (msg[8]&0x8) >> 3;
                vert_rate = ((msg[8]&7) << 6) | ((msg[9]&0xfc) >> 2);
                # Compute velocity and angle from the two speed components. 
                velocity = sqrt(ns_velocity*ns_velocity+ew_velocity*ew_velocity);
                if (velocity):
                    ewv = ew_velocity;
                    nsv = ns_velocity;

                    if (ew_dir): ewv *= -1;
                    if (ns_dir): nsv *= -1;
                    heading = -arctan2(ewv,nsv);
                    
                    # We don't want negative values but a 0-360 scale. 
                    if (heading < 0):
                        heading += 2 * np.pi;
                else:
                    heading = 0;
                plane.heading = heading
                    
            if ( mesub == 3 or mesub == 4):
                plane.heading = (360 / 128) * (((msg[5] & 3) << 5) | (msg[6] >> 3)) * np.pi / 180;
                
            strp = time.strftime("%H:%M:%S", time.localtime()) + " found plane angle %f degree for plane %x!\n" % (plane.heading * 180.0 / np.pi, addr)
            sys.__stdout__.write(strp)
            log.write(strp)
            sys.stdout.flush()
                

def sdr_read( Qin, sdr, N_samples, stop_flag ):
    
    t0 = time.time()
    while (  not stop_flag.is_set() ):
        data_chunk = abs(sdr.read_samples(N_samples))   # get samples 
        Qin.put( data_chunk ) # append to list

    sdr.close()

    
def signal_process( Qin, source, functions, plot, log, stop_flag  ):

    detectPreamble = functions[0]
    data2bit = functions[1]
    
    b = 0
    plane_list = {};
    while(  not stop_flag.is_set() ):
        
        # Get streaming chunk
        chunk = Qin.get();
        
        strp = time.strftime("%H:%M:%S", time.localtime()) + " looking for packets...\n"
        sys.__stdout__.write(strp)
        log.write(strp)
        
        idx_preamble = detectPreamble(chunk)
        
        for n in idx_preamble:
            bits = data2bit(chunk[n:(n+16+MODES_LONG_MSG_BITS*2)])
            msg = bit2byte( bits )
            decodeModesMessage( msg, plane_list, log )
                   
        # Update map:
        lat = []
        lon = []
        heading = []
        flightnum = []
        for addr in plane_list:
            plane = plane_list[addr]
            (mx, my) = LatLonToMeters(plane.position[0],plane.position[1])
            lat.append( my )
            lon.append( mx )
            heading.append( plane.heading )
            flightnum.append( plane.flightnum )
            
        source.data['lat'] = lat
        source.data['lon'] = lon
        source.data['heading'] = heading
        source.data['flightnum'] = flightnum
        
        push_notebook()
        Qin.queue.clear()
        
    log.close()
    
    
def LatLonToMeters(lat, lon ):
    #"Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"
    originShift = 2 * math.pi * 6378137 / 2.0
    mx = lon * originShift / 180.0
    my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

    my = my * originShift / 180.0
    return mx, my

                
def rt_flight_radar( fs, center_freq, gain, N_samples, pos_ref, functions ):

    #clear_output()
    #time.sleep(1)
    # create an input output FIFO queues
    Qin = queue.Queue()
    
    # create a pyaudio object
    sdr = RtlSdr()
    sdr.sample_rate = fs    # sampling rate
    sdr.center_freq = center_freq   # 1090MhZ center frequency
    sdr.gain = gain
    
    # initialize map

    # Berkeley (lat, lon) = (37.871853, -122.258423)
    
    (mx_d, my_d) = LatLonToMeters(pos_ref[0]-0.2, pos_ref[1]-0.2)
    (mx_u, my_u) = LatLonToMeters(pos_ref[0]+0.2, pos_ref[1]+0.2)
    
    plot = figure(x_range=(mx_d, mx_u), y_range=(my_d, my_u),
           x_axis_type="mercator", y_axis_type="mercator")
    plot.add_tile(CARTODBPOSITRON)


    plot.title.text = "Flight Radar"

    # create lat, longitude source
    source = ColumnDataSource(
        data=dict(
            lat=[],
            lon=[],
            heading = [],
            flightnum = []
        )
    )
    
    # create plane figure
    oval1 = Oval( x = "lon", y = "lat", width=3000, height=700, angle= "heading", fill_color="blue", line_color="blue")
    oval2 = Oval( x = "lon", y = "lat", width=1000, height=7000, angle= "heading", fill_color="blue", line_color="blue")
    text = Text( x = "lon", y = "lat", text_font_size="10pt", text="flightnum", angle= "heading", text_color="red")
    

    
    plot.add_glyph(source, oval1)
    plot.add_glyph(source, oval2)
    plot.add_glyph(source, text)


    output_notebook()
    handle = show(plot,notebook_handle=True)
    
    # initialize write file
    log = open('rtadsb_log','a')
    
    # initialize stop_flag
    stop_flag = threading.Event()

    # initialize threads
    t_sdr_read = threading.Thread(target = sdr_read,   args = (Qin, sdr, N_samples, stop_flag  ))
    t_signal_process = threading.Thread(target = signal_process, args = ( Qin, source, functions, plot, log, stop_flag))
    
    

    # start threads
    t_sdr_read.start()
    t_signal_process.start()


    return stop_flag
