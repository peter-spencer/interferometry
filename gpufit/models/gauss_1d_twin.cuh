#ifndef GPUFIT_GAUSS1D_TWIN_CUH_INCLUDED
#define GPUFIT_GAUSS1D_TWIN_CUH_INCLUDED

/* Description of the calculate_gauss1d function
* ==============================================
*
* This function calculates the values of one-dimensional gauss model functions
* and their partial derivatives with respect to the model parameters. 
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
*             p[2]: width (standard deviation)
*             p[3]: offset
* second Gaussian:
*             p[4]: amplitude
*             p[5]: center coordinate
*             p[6]: width (standard deviation)
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
* Calling the calculate_gauss1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_gauss1dtwin(
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

    REAL const * p = parameters;
    
    // value

    REAL const argxA = (x - p[1]) * (x - p[1]) / (2 * p[2] * p[2]);
    REAL const exA = exp(-argxA);
    REAL const argxB = (x - p[5]) * (x - p[5]) / (2 * p[6] * p[6]);
    REAL const exB = exp(-argxB);
    value[point_index] = p[0] * exA + p[4] * exB + p[3];

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points]  = exA;
    current_derivative[1 * n_points]  = p[0] * exA * (x - p[1]) / (p[2] * p[2]);
    current_derivative[2 * n_points]  = p[0] * exA * (x - p[1]) * (x - p[1]) / (p[2] * p[2] * p[2]);
    current_derivative[3 * n_points]  = 1;
    current_derivative[4 * n_points]  = exB;
    current_derivative[5 * n_points]  = p[4] * exB * (x - p[5]) / (p[6] * p[6]);
    current_derivative[6 * n_points]  = p[4] * exB * (x - p[5]) * (x - p[5]) / (p[6] * p[6] * p[6]);
}

#endif
