# interferometry
 Processing data from a white light interferometer

This repository currently consists of a Python script to process interferometer data into a 2D height map of a surface.

This surface is either a single reflective surface, or one with two reflections because of a layer of transparent material. This was developed for measuring features on surfaces covered, or partially covered, with S1813 photoresist.

## Prerequisites

Written and tested in Python 3.7 on Windows 10. The following Python libraries are used:

- Numpy
- Scipy
- OpenCV
- cuPy
- Gpufit
- Gwyfile

The image processing uses custom fit models in Gpufit and therefore Gpufit must be compiled from modified source code to work; all the other libraries are used as-is and will work without modification.

The modified Gpufit files are in the gpufit subdirectory.

The Hilbert transform function is derived from the source code of the cusignal library, altered to split up the operation into smaller chunks for better memory handling; it is not required to install cusignal.

Gwyddion is suggested as the software viewer for the output files, and by default the native .gwy file format is used.

## What it works with

The script was developed and tested with monochrome data taken from the red colour channel of a Sony mirrorless SLR camera through an Olympus IMT-2 microscope with Epi illumination and a 10X Nikon DI Mirau objective. The samples were scanned in the Z-axis using a Physik Instrumente closed-loop piezo stage. Measurement data was encoded at a resolution of 640 x 480, for a total of 1001 Z-height positions covering a range of 20 microns. My computer could handle scans of this size in memory but would struggle if the resolution was increased any further.

My experience to date has shown that the measurements can be performed successfully with some compromises on equipment, although more stable apparatus gets much more consistent data and is a worthwhile investment. I experimented with some 3D printed brackets and adapters and achieved decent success but they tended to drift in Z over the course of a scan due to air currents and handling of the microscope (although the new experimental drift correction appears to successfully correct the data); this drift only became a problem in my experience when accuracy of 10% or better was needed in the Z axis.

The resolution limits outlined above for X, Y, and Z were perfectly suited for my application but improvements can be made. For example, by detecting reflections during the initial data loading the interferograms could be isolated and non-useful data discarded: This would allow longer and/or finer-stepped Z-scans to be processed efficiently. Higher XY resolution can be handled by batch processing sub-regions of the image, or interleaved pixels. These may be added as future improvements.

For development a closed-loop piezo stage was essential but now that Z drift measurement and compensation is feasible, I believe that an open loop Z positioning system would be practical under the right circumstances. It’s even possible that successful scans could be made with a setup that allows the sample to slowly drift through the scan range of its own accord, rather than a deterministic process.

## Theory of operation

The data from the interferometer is stored in a AVI movie format for efficiency and because the losses due to compression have negligible impact on the data quality.

The analysis script reads in the AVI file data into a 3-dimensional XYZ matrix where each element is the intensity recorded by the camera.

First, the script determines the common background signal along the z-axis by taking the 2D median average of all XY data points for each Z. This background signal is subtracted from the data.

Next, the number of layers is estimated: A Fourier transform is made, along Z, for each data point. The wavelength region around the interference fringe peak(s) is isolated and the number of distinct peaks is counted. If a single peak is found at a given XY point, then there is only a single resolved reflection signal; multiple and/or split peaks in the Fourier signal indicate additional reflections and the layer count is set to two.

The Hilbert analytic is then calculated and its absolute value taken to extract the intensity envelope of the one or two interferograms along Z at each XY point.

The Hilbert process removes the interference fringes, except where stray light in the optical system has caused a slight offset because of the difference in illumination pattern when the interference is taking place. The fraction of stray light is estimated by taking the Fourier transform of the absolute value of the Hilbert analytic, and measuring the signal at the interference fringe wavelength.

An initial guess for the Z height of each point in XY is then made by finding the maximum along Z for each point in XY.

This initial guess for Z height is then refined by least-squares fitting two Gaussian peaks to the Hilbert signal, where the second potential reflection is constrained to not overlap with the first reflection signal. The goal of the least-squares fit is to determine the centre of the reflection to somewhat better than a quarter-wavelength of the dominant wavelength in the interference pattern, e.g. to within ~100 nm if the dominant wavelength is 600 nm is good target.

Finally, the least-squares estimate is refined by another least-squares fit to a complete model of the interferogram's central portion that includes the interference fringes. This final fit is made against the original data after background subtraction. The central wavelength, amplitude, coherence length (width of the envelope), and stray light coefficient are fixed parameters from the least-sqaures envelope fit and only the dominant wavelength and phase are fitting parameters.

This last fit is performed separately for each of the two reflection estimates from the least-square envelope fit because this has given the best results during testing.

With the final fit, the Z height estimate can be refined by including the phase difference from the envelope centre. This is the same as fixing the phase shift on reflection to zero (relative to the reference mirror). This results in a dramatic improvement in the resolution and noise immunity of the output Z height.

The weakness of including phase is that the algorithm will be unstable if the envelope centre is off by approximately one interference fringe. If this happens, discrete jumps in Z height by a quarter of the dominant wavelength will be observed in the output data.

With the above algorithm, a 600 nm dominant wavelength interferogram with around 1000 nm coherence length, scanned at 20 nm intervals in Z, gives an RMS output noise of around 2 nm.