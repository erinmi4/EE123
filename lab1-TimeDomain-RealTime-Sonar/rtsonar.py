import numpy as np, matplotlib.pyplot as plt
from numpy import *
from scipy import signal
from scipy import interpolate
import sounddevice as sd
import queue
import threading
from time import sleep
import matplotlib.cm as cm

import bokeh.plotting as bk
from bokeh.models import GlyphRenderer
from bokeh.io import push_notebook
from IPython.display import clear_output

bk.output_notebook()


def put_data( Qout, ptrain, stop_flag):
    while( not stop_flag.is_set() ):
        if ( Qout.qsize() < 2 ):
            Qout.put( ptrain )
            
       
def signal_process( Qin, Qdata, pulse_a, Nseg, Nplot, dbf, fs, maxdist, temperature, functions, stop_flag  ):
    # Signal processing function for real-time sonar
    # Takes in streaming data from Qin and process them using the functions defined above
    # Uses the first 2 pulses to calculate for delay
    # Then for each Nseg segments calculate the cross correlation (uses overlap-and-add)
    # Inputs:
    # Qin - input queue with chunks of audio data
    # Qdata - output queue with processed data
    # pulse_a - analytic function of pulse
    # Nseg - length between pulses
    # Nplot - number of samples for plotting
    # fs - sampling frequency
    # maxdist - maximum distance
    # temperature - room temperature

    crossCorr = functions[2]
    findDelay = functions[3]
    dist2time = functions[4]
    
    # initialize Xrcv 
    Xrcv = zeros( 2 * Nseg, dtype='complex' );
    cur_idx = 0; # keeps track of current index
    found_delay = False;
    maxsamp = min(int(dist2time( maxdist, temperature) * fs), Nseg); # maximum samples corresponding to maximum distance
    
    while(  not stop_flag.is_set() ):
        # Get streaming chunk
        chunk = Qin.get();
        chunk = chunk.reshape(len(chunk),)

        Xchunk =  crossCorr( chunk, pulse_a ) 
        
        # Overlap-and-add
        Xrcv[cur_idx:(cur_idx+len(chunk)+len(pulse_a)-1)] += Xchunk;
        cur_idx += len(chunk)
        
            
        idx = findDelay( abs(Xrcv), Nseg );

        Xrcv = np.roll(Xrcv, -idx );
        Xrcv[-idx:] = 0;

        # crop a segment from Xrcv and interpolate to Nplot
        Xrcv_seg = (abs(Xrcv[:maxsamp].copy()) / abs( Xrcv[0] )) ** 0.5 ;
        interp = interpolate.interp1d(r_[:maxsamp], Xrcv_seg)
        Xrcv_seg = interp( r_[:maxsamp-1:(Nplot*1j)] )

        # remove segment from Xrcv
        Xrcv = np.roll(Xrcv, -Nseg );
        Xrcv[-Nseg:] = 0
        cur_idx = 0;
        
        # offset parameters
        eps = 10.0**(-dbf/20.0)  # minimum signal
        
        # find maximum
        Xrcv_seg_max = Xrcv_seg.max()

        # compute 20*log magnitude, scaled to the max
        Xrcv_seg_log = 20.0 * np.log10( (Xrcv_seg / Xrcv_seg_max)*(1-eps) + eps )

        # rescale image intensity to 256
        img = 256*(Xrcv_seg_log + dbf)/dbf - 1

        Qdata.put( Xrcv_seg );
            

            
def image_update( Qdata, fig, maxrep, Nplot, stop_flag):
    renderer = fig.select(dict(name='echos', type=GlyphRenderer))
    source = renderer[0].data_source
    img = source.data['image'][0];
    
    while(  not stop_flag.is_set() ):
        new_line = Qdata.get();
            
        img = np.roll( img, 1, 0);
        view = img.view(dtype=np.uint8).reshape((maxrep, Nplot, 4))
        view[0,:,:] = cm.gray(new_line) * 255;
    
        source.data['image'] = [img]
        push_notebook()
        Qdata.queue.clear();
        

    
def rtsonar( f0, f1, fs, Npulse, Nseg, maxrep, Nplot, dbf, maxdist, temperature, functions ):
    
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        Qin.put( indata )  # Global queue

        try:
            data = Qout.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)

        outdata[:] = data.reshape(len(data),1)
    
    Nrep=1
    clear_output()
    genChirpPulse = functions[0]
    genPulseTrain = functions[1]
    
    pulse_a = genChirpPulse(Npulse, f0,f1,fs) * np.hanning(Npulse)
    pulse = np.real(pulse_a)
    ptrain = genPulseTrain(pulse, Nrep, Nseg)
    
    # create an input output FIFO queues
    Qin = queue.Queue()
    Qout = queue.Queue()
    Qdata = queue.Queue()
    
    img = np.zeros((maxrep,Nplot), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((maxrep, Nplot, 4))
    view[:,:,3] = 255;

    # initialize plot
    fig = bk.figure(title = 'Sonar',  y_axis_label = "Time [s]", x_axis_label = "Distance [m]",
                    x_range=(0, maxdist/100), y_range=(0, maxrep * Nseg / fs ) , 
                    plot_height = 400, plot_width = 800 )
    fig.image_rgba( image = [ img ], x=[0], y=[0], dw=[maxdist/100], dh=[maxrep * Nseg / fs ], name = 'echos' )
    bk.show(fig,notebook_handle=True)
    
    # initialize stop_flag
    stop_flag = threading.Event()   
    
    # initialize threads
    
    t_put_data = threading.Thread(target = put_data, args = (Qout, ptrain, stop_flag  ))
    st = sd.Stream( device=(2,2), samplerate = fs, blocksize=len(ptrain), channels=1, callback=audio_callback)
    t_signal_process = threading.Thread(target = signal_process, args = ( Qin, Qdata, pulse_a, Nseg, Nplot, dbf, fs, maxdist, temperature, functions, stop_flag))
    t_update_image = threading.Thread(target = image_update, args = (Qdata, fig, maxrep, Nplot, stop_flag) )

    # start threads
    t_put_data.start()
    st.start()
    t_signal_process.start()
    t_update_image.start()
    
    return (stop_flag, st)
