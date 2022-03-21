#ifndef GPUFIT_INTERFEROGRAM1D_CUH_INCLUDED
#define GPUFIT_INTERFEROGRAM1D_CUH_INCLUDED

/* Description of the calculate_interferogram1d function
* ==============================================
*
* This function calculates the values of one-dimensional interference
* pattern model functions and their partial derivatives with respect
* to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* There are three possibilities regarding the X values:
*
*   No X values provided: 
*
*       If no user information is provided, the (X) coordinate of the 
*       first data value is assumed to be (0.0).  In this case, for a 
*       fit size of M data points, the (X) coordinates of the data are 
*       simply the corresponding array index values of the data array, 
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then 
*       the same X values will be used for all fits.  In this case, the 
*       size of the user_info array (in bytes) must equal 
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array 
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: wavelength
*             p[3]: coherence length
*             p[4]: phase
*             p[5]: offset
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. 
*
* user_info_size: The size of user_info in bytes. 
*
* Calling the calculate_interferogram1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_interferogram1da(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL * user_info_float = (REAL*)user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters

    /* parameters: An input vector of model parameters.
    *             p[0]: amplitude
    *             p[1]: center coordinate
    *             p[2]: wavelength
    *             p[3]: coherence length
    *             p[4]: phase
    *             p[5]: offset
    */
    REAL const * p = parameters;
    
    // value

    REAL const argx = 4 * (x - p[1]) * (x - p[1]) / (p[3] * p[3]);
    REAL const ex = exp(-argx);
    //REAL const argt = 4 * 3.14159265358979323846 * (x - p[1]) / p[2] - p[4];
    REAL const argt = 4 * 3.14159265358979323846 * (x - p[1]) / p[2] - p[4];
    REAL const tr = cos(argt);
    value[point_index] = p[0] * ex * tr + p[5];
    //value[point_index] = p[0] * ex * tr + p[4];

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points]  = ex * tr;    // done.

    current_derivative[1 * n_points]  = p[0] * ex * tr * 8 * (x - p[1]) / (p[3] * p[3])
                                      + p[0] * ex * 4 * 3.14159265358979323846 * sin(argt)  /  p[1];            // done.
    current_derivative[2 * n_points]  = p[0] * ex * 4 * 3.14159265358979323846 * (x - p[1]) * sin(argt)  / (p[2] * p[2]); // done.
    current_derivative[3 * n_points]  = p[0] * ex * tr * 8 * (x - p[1]) * (x - p[1]) / (p[3] * p[3] * p[3]);    // done.
    current_derivative[4 * n_points]  = p[0] * ex * sin(argt);                                                  // done.

    current_derivative[5 * n_points]  = 1;          // done.
    //current_derivative[4 * n_points]  = 1;          // done.
}

__device__ void calculate_interferogram1db(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL * user_info_float = (REAL*)user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters

    /* parameters: An input vector of model parameters.
    *             p[0]: amplitude
    *             p[1]: center coordinate
    *             p[2]: wavelength
    *             p[3]: coherence length
    *             p[4]: phase
    *             p[5]: offset
    */
    REAL const * p = parameters;
    
    // value

    // REAL const argx = 4 * (x - p[1]) * (x - p[1]) / (p[3] * p[3]);
    // REAL const ex = exp(-argx);
    // REAL const argt = 4 * 3.14159265358979323846 * (x - p[1]) / p[2] - p[4];
    // REAL const tr = cos(argt);
    // value[point_index] = p[0] * ex * tr + p[5];

    REAL const argx = 4 * (x - p[1]) * (x - p[1]) / (p[3] * p[3]);
    REAL const ex = exp(-argx);
    REAL const argt = 4 * 3.14159265358979323846 * (x - p[1]) / p[2] - p[4];
    REAL const tr = cos(argt);
    value[point_index] = (1 + p[5]) * p[0] * ex * (tr - p[5]);
    // value[point_index] = (1 + p[5]) * p[0] * ex * tr     -     (1 + p[5]) * p[5] * p[0] * ex;
    // value[point_index] = p[0] * ex * (tr - p[5] + tr*p[5] - p[5]^2);

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points]  = (1 + p[5]) * ex * (tr - p[5]);    // done.

    current_derivative[1 * n_points]  = (1 + p[5]) * p[0] * ex * (tr - p[5]) * 8 * (x - p[1]) / (p[3] * p[3])
                                      + (1 + p[5]) * p[0] * ex * 4 * 3.14159265358979323846 * sin(argt)  /  p[1];   // done.
    current_derivative[2 * n_points]  = (1 + p[5]) * p[0] * ex * 4 * 3.14159265358979323846 * (x - p[1]) * sin(argt)  / (p[2] * p[2]); // done.
    current_derivative[3 * n_points]  = (1 + p[5]) * p[0] * ex * (tr - p[5]) * 8 * (x - p[1]) * (x - p[1]) / (p[3] * p[3] * p[3]);    // done.
    current_derivative[4 * n_points]  = (1 + p[5]) * p[0] * ex * sin(argt);     // done.

    current_derivative[5 * n_points]  = p[0] * ex * (tr - 1 - 2*p[5]);          // done.
}

#endif
