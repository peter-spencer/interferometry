# Process an AVI movie collected by a white light interferometer
# Peter Spencer Feb 2022

## Input file information
filename = "output.avi"
colour_channel = 0              # 0 for greyscale AVI data                                  [channel number]
z_step = 20e-9                  # Step size between scan frames                             [metres]

## Output file settings
output_filename = 'output.gwy'
gwy_palette = 'Spectral'
gwy_palette_layers = 'Gray'

## Initial estimates to start fitting process with
wavelength = 650e-9             # Interference fringe dominant wavelength                   [metres]
wavelength_lowest = 550e-9      # Lower bound on dominant wavelength                        [metres]
wavelength_highest = 650e-9     # Upper bound on dominant wavelength                        [metres]
width_estimate = 600e-9         # Coherence length of intensity envelope                    [metres]
width_lowest = 250e-9           # Lower bound on coherence length                           [metres]
width_highest = 1.00e-6         # Upper bound on coherence length                           [metres]
intensity_lowest = 0            # Lower bound on peak intensity                             [arbitrary units]
intensity_highest = 255         # Upper bound on peak intensity                             [arbitrary units]
offset_lowest = 0               # Lower bound on background intensity                       [arbitrary units]
offset_highest = 20             # Upper bound on background intensity                       [arbitrary units]
layer_detect_low = 0.5          # Fraction of `wavelength` for start of layout detection    [ratio]
layer_detect_high = 1.5         # Fraction of `wavelength` for end of layout detection      [ratio]
layer_detect_prominence = 0.1   # Minimum dip between Fourier peaks                         [ratio]
layer_detect_height = 0.3       # Minimum height of a relevant Fourier peak                 [ratio]

## Analysis parameters
refractive_index = 1.65         # Optical constant of thin-film between reflections         [dimensionless]

## Other settings
verbose = False                 # Descriptive text messages (True) or progress bar (False)  [boolean]
analysis_steps = 10             # Number of waypoints in the analysis process               [number]

###########################################################################################
#### Import libraries and define helper functions                                      ####
###########################################################################################

import time
import numpy as np
from scipy.signal import find_peaks
from gwyfile.objects import GwyContainer, GwyDataField
import cv2
import pygpufit.gpufit as gf
import cupy as cp

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# The hilbert transform cupy kernel for cuda, taken from cusignal source code
_hilbert_kernel = cp.ElementwiseKernel(
    "",
    "T h",
    """
    if ( !odd ) {
        if ( ( i == 0 ) || ( i == bend ) ) {
            h = 1.0;
        } else if ( i > 0 && i < bend ) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    } else {
        if ( i == 0 ) {
            h = 1.0;
        } else if ( i > 0 && i < bend) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    }
    """,
    "_hilbert_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)


def gpu_clear():
    cp.fft.config.get_plan_cache().clear()

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

def split_hilbert(x, d=10):
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the last axis. Taken from
    cusignal library and adapted. Splits the input into d
    batches to handle larger input datasets within GPU memory.
    Parameters
    ----------
    x : array_like
        Signal data.  Must be real and at least 2D. First 
        dimension must have length divisible by `d`
    d : Number of blocks to split data into. Default is 10. The
        input must be divisible by this number.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along last axis
    """
    rdata = cp.asarray(x)
    
    if cp.iscomplexobj(rdata):
        raise ValueError("x must be real.")
    if x.shape[0] % d > 0:
        raise ValueError("length of first dimension of x must divide by d.")

    h = cp.empty((rdata.shape[1],), dtype=rdata.dtype)
    _hilbert_kernel(h)

    ind = [cp.newaxis] * 2
    ind[-1] = slice(None)
    h = h[tuple(ind)]

    a = x.shape[0] // 10
    hdata = np.zeros_like(x)

    for b in range(d):
        st = b*a
        sp = (b+1)*a

        Xf = cp.fft.fft(rdata[st:sp])
        hdata[st:sp] = cp.asnumpy(cp.abs(cp.fft.ifft(Xf * h)))

    gpu_clear()

    return hdata

###########################################################################################################
###########################################################################################################
###########################################################################################################

######################################### Main Script starts here #########################################

###########################################################################################################
###########################################################################################################
###########################################################################################################

all_start = time.perf_counter()

###########################################################################################
#### Load the data into memory                                                         ####
###########################################################################################

# Open the source data file: it is an AVI movie, where each frame is a different z-height
cap = cv2.VideoCapture(filename)

# Get the number of frames to be processed (i.e. number of Z data points)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Reading video file into memory...")

# Read the first frame
success,image_raw = cap.read()

# Measure the image dimensions
frame_width = image_raw.shape[0]
frame_height = image_raw.shape[1]
pixels = frame_width * frame_height

# Allocate buffer variables
image_buffer = np.zeros((frame_width,frame_height,length), dtype='f4')

# Loop over every frame in the source file
idx = 0
while success:

    # Update progress
    printProgressBar(idx, length, prefix="Read", suffix="frames", decimals=0)

    # Store the processed frame in the image buffer
    image_buffer[:,:,idx] = image_raw[:,:,colour_channel]

    # Read the next frame
    idx += 1
    success,image_raw = cap.read()

printProgressBar(idx, length, prefix="Read", decimals=0)
print('Video file loaded.')

###########################################################################################
#### Prepare the data and subtract the background signal                               ####
###########################################################################################

if verbose:
    print('Normalising scan data and subtracting background...')
else:
    analysis_step = 0
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

# Organise the data into a shape suitable for fitting to each pixel in parallel
data = np.reshape(image_buffer,(pixels,length))

# Get an array of the average intensities of each pixel
norm = np.mean(data, axis=1, keepdims=True)

# Normalise each pixel to have the same average intensity
data_normed = data / norm

# Calculate the median pixel at each Z height to calculate the common background signal
med = np.median(data_normed, axis=0, keepdims=True)

# Subtract the common background signal from each pixel in the original data
data -= med * norm

###########################################################################################
#### Get the layer count                                                               ####
###########################################################################################

if verbose:
    print('Starting Fourier transform...')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

fstart = time.perf_counter()

# Take the real-valued Fourier transform along Z for each pixel
fdata = cp.asnumpy(cp.fft.rfft(cp.asarray(data)))

fend = time.perf_counter()

if verbose:
    print("Fourier transform took %g seconds"%(fend-fstart))
    print('Starting peak finding for layer count determination...')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

fstart = time.perf_counter()

f = np.linspace(0,data.shape[1]*z_step/4 / 1e-9,num=501, endpoint=True)

ifstart = np.int32(wavelength*layer_detect_low / z_step * 2)
ifend = np.int32(wavelength*layer_detect_high / z_step * 2)

fringex = f[ifstart:ifend]
fringes = fdata[:,ifstart:ifend]

layer_count = np.zeros(data.shape[0])
peaky = []
for idx in range(data.shape[0]):
    this_fringe = np.abs(fringes[idx,:])
    this_fringe /= np.max(this_fringe)
    peaks, _ = find_peaks(this_fringe, prominence=layer_detect_prominence, height=layer_detect_height)
    peaky.append(peaks)
    layer_count[idx] = np.where(len(peaks) > 1, 2, 1)

fend = time.perf_counter()

if verbose:
    print("Peak finding took %g seconds"%(fend-fstart))
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)


###########################################################################################
#### Calculate the intensity envelope of the interferogram                             ####
###########################################################################################

if verbose:
    print("Starting Hilbert transform...")

hstart = time.perf_counter()

# This calculates the Hilbert analytic, the absolute value is the intensity envelope and
# the phase angle can be used to find the "instantaneous frequency", which we can use here
# to detect drifting and nonlinearity in the z-scan.
hs = split_hilbert(data)

hend = time.perf_counter()

if verbose:
    print("Hilbert transform took %g seconds"%(hend-hstart))
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

###########################################################################################
#### Analyse the effect of stray light on the interferogram                            ####
###########################################################################################

if verbose:
    print('Analysing stray light contribution...')

fstart = time.perf_counter()

# This Fourier transform is being calculated to detect interference fringes that were not
# removed from the intensity envelope (this "bleed-through" happend because of stray light)
hf = cp.asnumpy(cp.fft.rfft(cp.asarray(hs)))

ibf = np.sum(np.abs(fdata)[:,ifstart:ifend], axis=1)
ihf = np.sum(np.abs(hf)[:,ifstart:ifend], axis=1)
stray_ratio = ihf / ibf                                 # Estimate of the stray light fraction

gpu_clear()

fend = time.perf_counter()

if verbose:
    print("Stray light correction took %g seconds"%(fend-fstart))
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

###########################################################################################
#### Simultaneously fit Gaussian functions to the two peaks                            ####
###########################################################################################

if verbose:
    print('Fit to both peaks simultaneously...')

intensity_first = np.argmax(hs, axis=1)

# initial parameters estimation
g_positionA = np.float32(intensity_first)
g_amplitudeA = np.float32(np.max(hs, axis=1))
g_widthA = np.float32(np.tile(width_estimate/z_step, (pixels,)))

g_positionB = np.float32(intensity_first + 3e-6/z_step)
g_amplitudeB = np.float32(np.max(hs, axis=1))
g_widthB = np.float32(np.tile(width_estimate/z_step, (pixels,)))

g_offset = np.float32(np.zeros((pixels,)))

initial_parameters = np.stack([g_amplitudeA, g_positionA, g_widthA, g_offset, g_amplitudeB, g_positionB, g_widthB],axis=-1)

if verbose:
    print('Starting GPU fitting on envelope...')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

# run Gpufit
to_fit = np.int32(np.array([1,1,1,0,1,1,1]))

constraints_each = np.float32(np.array([    intensity_lowest, intensity_highest,
                                            0, length,
                                            width_lowest/z_step, width_highest/z_step,
                                            offset_lowest, offset_highest,
                                            intensity_lowest, intensity_highest,
                                            0, length,
                                            width_lowest/z_step, width_highest/z_step]))

constraints_type = np.int32(np.array([3,3,3,3,3,3,3]))

constraints = np.tile(constraints_each, (data.shape[0],1))

parameters, states, chi_squares_height, number_iterations, execution_time = gf.fit_constrained(hs, None,
    gf.ModelID.GAUSS_1D_TWIN, initial_parameters, parameters_to_fit=to_fit,
    constraints=constraints, constraint_types=constraints_type)

gpu_clear()

if verbose:
    print('GPU fitting completed (envelope).')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

###########################################################################################
#### Complete fit to the first peak's interference pattern to extract the phase        ####
###########################################################################################

## Re-fit the data directly to extract the phase
# initial parameters estimation
g_position   = np.float32(parameters[:,1])                  # From Gaussian fit to hilbert envelope
g_amplitude  = np.float32(parameters[:,0])                  # From Gaussian fit to hilbert envelope
g_width      = np.float32(2*np.sqrt(2)*parameters[:,2])     # From Gaussian fit to hilbert envelope
g_wavelength = np.float32(np.tile(wavelength/z_step, (pixels,)))
g_phase      = np.float32(np.zeros((pixels,)))              # Assume envelope peak == maximum intensity
g_offset     = np.float32(stray_ratio)                      # From FFT residual fringing of hilbert envelope

initial_parameters = np.stack([g_amplitude, g_position, g_wavelength, g_width, g_phase, g_offset],axis=-1)

if verbose:
    print('Starting GPU fitting to get first peak\'s phase...')

to_fit = np.int32(np.array([0,0,1,0,1,0]))

constraints_each = np.float32(np.array([    intensity_lowest, intensity_highest,
                                            0 ,length,
                                            wavelength_lowest/z_step,wavelength_highest/z_step,
                                            2*np.sqrt(2)*width_lowest/z_step ,2*np.sqrt(2)*width_highest/z_step,
                                            -np.pi, np.pi,
                                            0 , 1]))

constraints_type = np.int32(np.array([3,3,3,3,3,3]))

constraints = np.tile(constraints_each, (data.shape[0],1))

parameters_two, states, chi_squares_height, number_iterations, execution_time = gf.fit_constrained(data, None,
    gf.ModelID.INTERFEROGRAM_1DB, initial_parameters, parameters_to_fit=to_fit,
    constraints=constraints, constraint_types=constraints_type)

image_first = parameters_two[:,1] + parameters_two[:,2]*parameters_two[:,4]/4/np.pi

gpu_clear()

if verbose:
    print('GPU fitting completed (first peak\'s phase).')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

###########################################################################################
#### Complete fit to the second peak's interference pattern to extract the phase       ####
###########################################################################################

## Re-fit the data directly to extract the phase
# initial parameters estimation
g_position   = np.float32(parameters[:,5])                  # From Gaussian fit to hilbert envelope
g_amplitude  = np.float32(parameters[:,4])                  # From Gaussian fit to hilbert envelope
g_width      = np.float32(2*np.sqrt(2)*parameters[:,6])     # From Gaussian fit to hilbert envelope
g_wavelength = np.float32(np.tile(wavelength/z_step, (pixels,)))
g_phase      = np.float32(np.zeros((pixels,)))              # Assume envelope peak == maximum intensity
g_offset     = np.float32(stray_ratio)                      # From FFT residual fringing of hilbert envelope

initial_parameters = np.stack([g_amplitude, g_position, g_wavelength, g_width, g_phase, g_offset],axis=-1)

if verbose:
    print('Starting GPU fitting to get second peak\'s phase...')

to_fit = np.int32(np.array([0,0,1,0,1,0]))

constraints_each = np.float32(np.array([    intensity_lowest, intensity_highest,
                                            0 ,length,
                                            wavelength_lowest/z_step,wavelength_highest/z_step,
                                            2*np.sqrt(2)*width_lowest/z_step ,2*np.sqrt(2)*width_highest/z_step,
                                            -np.pi, np.pi,
                                            0 , 1]))

constraints_type = np.int32(np.array([3,3,3,3,3,3]))

constraints = np.tile(constraints_each, (data.shape[0],1))

parameters_three, states, chi_squares_height, number_iterations, execution_time = gf.fit_constrained(data, None,
    gf.ModelID.INTERFEROGRAM_1DB, initial_parameters, parameters_to_fit=to_fit,
    constraints=constraints, constraint_types=constraints_type)

image_second = parameters_three[:,1] + parameters_three[:,2]*parameters_three[:,4]/4/np.pi

gpu_clear()

if verbose:
    print('GPU fitting completed (second peak\'s phase).')
else:
    analysis_step += 1
    printProgressBar(analysis_step, analysis_steps, prefix="Analysis", decimals=0)

###########################################################################################
#### Sort the results and save the data                                                ####
###########################################################################################

# Sort the layers by height

layer_top = np.zeros_like(image_first)
layer_bottom = np.zeros_like(image_first)

for idx in range(pixels):
    if layer_count[idx] == 1:
        layer_top[idx] = image_first[idx]
        layer_bottom[idx] = image_first[idx]
    elif image_first[idx] > image_second[idx]:
        layer_top[idx] = image_first[idx]
        layer_bottom[idx] = image_second[idx]
    else:
        layer_top[idx] = image_second[idx]
        layer_bottom[idx] = image_first[idx]

# Prepare the image arrays for saving
layer_top = np.float64(np.reshape(layer_top*z_step,(frame_width,frame_height)))
layer_bottom = np.float64(np.reshape(layer_bottom*z_step,(frame_width,frame_height)))
layer_count = np.float64(np.reshape(layer_count,(frame_width,frame_height)))

# Derived measurements
layer_delta = layer_top - layer_bottom
substrate = layer_top - layer_delta/refractive_index

# Write the analysis results to the Gwyddion file

# Create a Gwyddion file object to save results into
obj = GwyContainer()

obj['/0/data/title'] = 'Top reflection height'
obj['/0/base/palette'] = gwy_palette
obj['/0/data'] = GwyDataField(layer_top, xreal=frame_width, si_unit_xy='m', yreal=frame_height, si_unit_z='m')

obj['/1/data/title'] = 'Bottom reflection height'
obj['/1/base/palette'] = gwy_palette
obj['/1/data'] = GwyDataField(layer_bottom, xreal=frame_width, si_unit_xy='m', yreal=frame_height, si_unit_z='m')

obj['/2/data/title'] = 'Layer count'
obj['/2/base/palette'] = gwy_palette_layers
obj['/2/data'] =  GwyDataField(layer_count, xreal=frame_width, si_unit_xy='m', yreal=frame_height, si_unit_z='')

obj['/4/data/title'] = 'Optical thickness'
obj['/4/base/palette'] = gwy_palette
obj['/4/data'] =  GwyDataField(layer_delta, xreal=frame_width, si_unit_xy='m', yreal=frame_height, si_unit_z='m')

obj['/5/data/title'] = 'Substrate surface (n=%g)'%(refractive_index)
obj['/5/base/palette'] = gwy_palette
obj['/5/data'] =  GwyDataField(substrate, xreal=frame_width, si_unit_xy='m', yreal=frame_height, si_unit_z='m')

# Export the data results
obj.tofile(output_filename)

###########################################################################################
#### Analysis completed                                                                ####
###########################################################################################

printProgressBar(analysis_steps, analysis_steps, prefix="Analysis", decimals=0)
print('')

all_done = time.perf_counter()
print('Processing completed in %g seconds'%(all_done-all_start))
