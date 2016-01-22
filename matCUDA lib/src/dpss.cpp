#include "array.cuh"

struct InputData
{
    //! host side representation of diagonal
    float  *a;
    //! host side representation superdiagonal
    float  *b;

    //! device side representation of diagonal
    float  *g_a;
    //! device side representation of superdiagonal
    float  *g_b;
    //! helper variable pointing to the mem allocated for g_b which provides
    //! space for one additional element of padding at the beginning
    float  *g_b_raw;
};

struct ResultDataSmall
{
    //! eigenvalues (host side)
    float *eigenvalues;

    // left interval limits at the end of the computation
    float *g_left;

    // right interval limits at the end of the computation
    float *g_right;

    // number of eigenvalues smaller than the left interval limit
    unsigned int *g_left_count;

    // number of eigenvalues bigger than the right interval limit
    unsigned int *g_right_count;

    //! flag if algorithm converged
    unsigned int *g_converged;

    // helper variables

    unsigned int mat_size_f;
    unsigned int mat_size_ui;

    float         *zero_f;
    unsigned int  *zero_ui;
};

struct ResultDataLarge
{
    // number of intervals containing one eigenvalue after the first step
    unsigned int *g_num_one;

    // number of (thread) blocks of intervals containing multiple eigenvalues
    // after the first step
    unsigned int *g_num_blocks_mult;

    //! left interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_left_one;

    //! right interval limits of intervals containing one eigenvalue after the
    //! first iteration step
    float *g_right_one;

    //! interval indices (position in sorted listed of eigenvalues)
    //! of intervals containing one eigenvalue after the first iteration step
    unsigned int *g_pos_one;

    //! left interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_left_mult;

    //! right interval limits of intervals containing multiple eigenvalues
    //! after the first iteration step
    float *g_right_mult;

    //! number of eigenvalues less than the left limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_left_count_mult;

    //! number of eigenvalues less than the right limit of the eigenvalue
    //! intervals containing multiple eigenvalues
    unsigned int *g_right_count_mult;

    //! start addresses in g_left_mult etc. of blocks of intervals containing
    //! more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult;

    //! accumulated number of intervals in g_left_mult etc. of blocks of
    //! intervals containing more than one eigenvalue after the first step
    unsigned int  *g_blocks_mult_sum;

    //! eigenvalues that have been generated in the second step from intervals
    //! that still contained multiple eigenvalues after the first step
    float *g_lambda_mult;

    //! eigenvalue index of intervals that have been generated in the second
    //! processing step
    unsigned int *g_pos_mult;
};

template<class T>
inline void freePtr(T *&ptr)
{
    if (NULL != ptr)
    {
        free(ptr);
        ptr = NULL;
    }
}

void initInputData(InputData &input, char *exec_path, const unsigned int mat_size, const unsigned int user_defined)
{
    // allocate memory
    input.a = (float *) malloc(sizeof(float) * mat_size);
    input.b = (float *) malloc(sizeof(float) * mat_size);

    if (1 == user_defined)
    {

        // initialize diagonal and superdiagonal entries with random values
        srand(278217421);

        // srand( clock());
        for (unsigned int i = 0; i < mat_size; ++i)
        {
            input.a[i] = (float)(2.0 * (((double)rand()
                                         / (double) RAND_MAX) - 0.5));
            input.b[i] = (float)(2.0 * (((double)rand()
                                         / (double) RAND_MAX) - 0.5));
        }

        // the first element of s is used as padding on the device (thus the
        // whole vector is copied to the device but the kernels are launched
        // with (s+1) as start address
        input.b[0] = 0.0f;
    }
    else
    {

        // read default matrix
        unsigned int input_data_size = mat_size;
        char *diag_path = sdkFindFilePath("diagonal.dat", exec_path);
        assert(NULL != diag_path);
        sdkReadFile(diag_path, &(input.a), &input_data_size, false);

        char *sdiag_path = sdkFindFilePath("superdiagonal.dat", exec_path);
        assert(NULL != sdiag_path);
        sdkReadFile(sdiag_path, &(input.b), &input_data_size, false);

        free(diag_path);
        free(sdiag_path);
    }

    // allocate device memory for input
    CUDA_CALL(cudaMalloc((void **) &(input.g_a)    , sizeof(float) * mat_size));
    CUDA_CALL(cudaMalloc((void **) &(input.g_b_raw), sizeof(float) * mat_size));

    // copy data to device
    CUDA_CALL(cudaMemcpy(input.g_a    , input.a, sizeof(float) * mat_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(input.g_b_raw, input.b, sizeof(float) * mat_size, cudaMemcpyHostToDevice));

    input.g_b = input.g_b_raw + 1;
}

void computeGerschgorin(float *d, float *s, unsigned int n, float &lg, float &ug)
{

    lg = FLT_MAX;
    ug = -FLT_MAX;

    // compute bounds
    for (unsigned int i = 1; i < (n - 1); ++i)
    {

        // sum over the absolute values of all elements of row i
        float sum_abs_ni = fabsf(s[i-1]) + fabsf(s[i]);

        lg = min(lg, d[i] - sum_abs_ni);
        ug = max(ug, d[i] + sum_abs_ni);
    }

    // first and last row, only one superdiagonal element

    // first row
    lg = min(lg, d[0] - fabsf(s[0]));
    ug = max(ug, d[0] + fabsf(s[0]));

    // last row
    lg = min(lg, d[n-1] - fabsf(s[n-2]));
    ug = max(ug, d[n-1] + fabsf(s[n-2]));

    // increase interval to avoid side effects of fp arithmetic
    float bnorm = max(fabsf(ug), fabsf(lg));

    // these values depend on the implementation of floating count that is
    // employed in the following
    float psi_0 = 11 * FLT_EPSILON * bnorm;
    float psi_n = 11 * FLT_EPSILON * bnorm;

    lg = lg - bnorm * 2 * n * FLT_EPSILON - psi_0;
    ug = ug + bnorm * 2 * n * FLT_EPSILON + psi_n;

    ug = max(lg, ug);
}

void initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size)
{

    // helper variables to initialize memory
    unsigned int zero = 0;
    unsigned int mat_size_f = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float *tempf = (float *) malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *) malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i)
    {
        tempf[i] = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    CUDA_CALL(cudaMalloc((void **) &result.g_num_one,
                               sizeof(unsigned int)));
    CUDA_CALL(cudaMemcpy(result.g_num_one, &zero, sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    CUDA_CALL(cudaMalloc((void **) &result.g_num_blocks_mult,
                               sizeof(unsigned int)));
    CUDA_CALL(cudaMemcpy(result.g_num_blocks_mult, &zero,
                               sizeof(unsigned int),
                               cudaMemcpyHostToDevice));


    CUDA_CALL(cudaMalloc((void **) &result.g_left_one, mat_size_f));
    CUDA_CALL(cudaMalloc((void **) &result.g_right_one, mat_size_f));
    CUDA_CALL(cudaMalloc((void **) &result.g_pos_one, mat_size_ui));

    CUDA_CALL(cudaMalloc((void **) &result.g_left_mult, mat_size_f));
    CUDA_CALL(cudaMalloc((void **) &result.g_right_mult, mat_size_f));
    CUDA_CALL(cudaMalloc((void **) &result.g_left_count_mult,
                               mat_size_ui));
    CUDA_CALL(cudaMalloc((void **) &result.g_right_count_mult,
                               mat_size_ui));

    CUDA_CALL(cudaMemcpy(result.g_left_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(result.g_right_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(result.g_pos_one, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(result.g_left_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(result.g_right_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(result.g_left_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(result.g_right_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **) &result.g_blocks_mult, mat_size_ui));
    CUDA_CALL(cudaMemcpy(result.g_blocks_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &result.g_blocks_mult_sum, mat_size_ui));
    CUDA_CALL(cudaMemcpy(result.g_blocks_mult_sum, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **) &result.g_lambda_mult, mat_size_f));
    CUDA_CALL(cudaMemcpy(result.g_lambda_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &result.g_pos_mult, mat_size_ui));
    CUDA_CALL(cudaMemcpy(result.g_pos_mult, tempf, mat_size_ui,
                               cudaMemcpyHostToDevice));
}

void computeEigenvaluesLargeMatrix(const InputData &input, const ResultDataLarge &result,
                              const unsigned int mat_size, const float precision,
                              const float lg, const float ug,
                              const unsigned int iterations)
{
    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);

    StopWatchInterface *timer_step1 = NULL;
    StopWatchInterface *timer_step2_one = NULL;
    StopWatchInterface *timer_step2_mult = NULL;
    StopWatchInterface *timer_total = NULL;
    sdkCreateTimer(&timer_step1);
    sdkCreateTimer(&timer_step2_one);
    sdkCreateTimer(&timer_step2_mult);
    sdkCreateTimer(&timer_total);

    sdkStartTimer(&timer_total);

    // do for multiple iterations to improve timing accuracy
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {

        sdkStartTimer(&timer_step1);
        bisectKernelLarge<<< blocks, threads >>>
        (input.g_a, input.g_b, mat_size,
         lg, ug, 0, mat_size, precision,
         result.g_num_one, result.g_num_blocks_mult,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_blocks_mult, result.g_blocks_mult_sum
        );

        getLastCudaError("Kernel launch failed.");
        CUDA_CALL(cudaDeviceSynchronize());
        sdkStopTimer(&timer_step1);

        // get the number of intervals containing one eigenvalue after the first
        // processing step
        unsigned int num_one_intervals;
        CUDA_CALL(cudaMemcpy(&num_one_intervals, result.g_num_one,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        dim3 grid_onei;
        grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 threads_onei;
        // use always max number of available threads to better balance load times
        // for matrix data
        threads_onei.x = MAX_THREADS_BLOCK;

        // compute eigenvalues for intervals that contained only one eigenvalue
        // after the first processing step
        sdkStartTimer(&timer_step2_one);

        bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
        (input.g_a, input.g_b, mat_size, num_one_intervals,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         precision
        );

        getLastCudaError("bisectKernelLarge_OneIntervals() FAILED.");
        CUDA_CALL(cudaDeviceSynchronize());
        sdkStopTimer(&timer_step2_one);

        // process intervals that contained more than one eigenvalue after
        // the first processing step

        // get the number of blocks of intervals that contain, in total when
        // each interval contains only one eigenvalue, not more than
        // MAX_THREADS_BLOCK threads
        unsigned int  num_blocks_mult = 0;
        CUDA_CALL(cudaMemcpy(&num_blocks_mult, result.g_num_blocks_mult,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        // setup the execution environment
        dim3  grid_mult(num_blocks_mult, 1, 1);
        dim3  threads_mult(MAX_THREADS_BLOCK, 1, 1);

        sdkStartTimer(&timer_step2_mult);

        bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
        (input.g_a, input.g_b, mat_size,
         result.g_blocks_mult, result.g_blocks_mult_sum,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_lambda_mult, result.g_pos_mult,
         precision
        );


        getLastCudaError("bisectKernelLarge_MultIntervals() FAILED.");
        CUDA_CALL(cudaDeviceSynchronize());
        sdkStopTimer(&timer_step2_mult);

    }

    sdkStopTimer(&timer_total);

    printf("Average time step 1: %f ms\n",
           sdkGetTimerValue(&timer_step1) / (float) iterations);
    printf("Average time step 2, one intervals: %f ms\n",
           sdkGetTimerValue(&timer_step2_one) / (float) iterations);
    printf("Average time step 2, mult intervals: %f ms\n",
           sdkGetTimerValue(&timer_step2_mult) / (float) iterations);

    printf("Average time TOTAL: %f ms\n",
           sdkGetTimerValue(&timer_total) / (float) iterations);

    sdkDeleteTimer(&timer_step1);
    sdkDeleteTimer(&timer_step2_one);
    sdkDeleteTimer(&timer_step2_mult);
    sdkDeleteTimer(&timer_total);
}

bool processResultDataLargeMatrix(const InputData &input, const ResultDataLarge &result,
                             const unsigned int mat_size,
                             const char *filename,
                             const unsigned int user_defined, char *exec_path)
{
    bool bCompareResult = false;
    const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
    const unsigned int mat_size_f  = sizeof(float) * mat_size;

    // copy data from intervals that contained more than one eigenvalue after
    // the first processing step
    float *lambda_mult = (float *) malloc(sizeof(float) * mat_size);
    CUDA_CALL(cudaMemcpy(lambda_mult, result.g_lambda_mult,
                               sizeof(float) * mat_size,
                               cudaMemcpyDeviceToHost));
    unsigned int *pos_mult =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    CUDA_CALL(cudaMemcpy(pos_mult, result.g_pos_mult,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost));

    unsigned int *blocks_mult_sum =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    CUDA_CALL(cudaMemcpy(blocks_mult_sum, result.g_blocks_mult_sum,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost));

    unsigned int num_one_intervals;
    CUDA_CALL(cudaMemcpy(&num_one_intervals, result.g_num_one,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    unsigned int sum_blocks_mult = mat_size - num_one_intervals;


    // copy data for intervals that contained one eigenvalue after the first
    // processing step
    float *left_one = (float *) malloc(mat_size_f);
    float *right_one = (float *) malloc(mat_size_f);
    unsigned int *pos_one = (unsigned int *) malloc(mat_size_ui);
    CUDA_CALL(cudaMemcpy(left_one, result.g_left_one, mat_size_f,
                               cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(right_one, result.g_right_one, mat_size_f,
                               cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(pos_one, result.g_pos_one, mat_size_ui,
                               cudaMemcpyDeviceToHost));

    // extract eigenvalues
    float *eigenvals = (float *) malloc(mat_size_f);

    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i)
    {

        eigenvals[pos_mult[i] - 1] = lambda_mult[i];
    }

    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index)
    {

        eigenvals[pos_one[i] - 1] = left_one[i];
    }

    if (1 == user_defined)
    {
        // store result
        writeTridiagSymMatlab(filename, input.a, input.b+1, eigenvals, mat_size);
        // getLastCudaError( sdkWriteFilef( filename, eigenvals, mat_size, 0.0f));

        printf("User requests non-default argument(s), skipping self-check!\n");
        bCompareResult = true;
    }
    else
    {

        // compare with reference solution

        float *reference = NULL;
        unsigned int input_data_size = 0;

        char *ref_path = sdkFindFilePath("reference.dat", exec_path);
        assert(NULL != ref_path);
        sdkReadFile(ref_path, &reference, &input_data_size, false);
        assert(input_data_size == mat_size);

        // there's an imprecision of Sturm count computation which makes an
        // additional offset necessary
        float tolerance = 1.0e-5f + 5.0e-6f;

        if (sdkCompareL2fe(reference, eigenvals, mat_size, tolerance) == true)
        {
            bCompareResult = true;
        }
        else
        {
            bCompareResult = false;
        }

        free(ref_path);
        free(reference);
    }

    freePtr(eigenvals);
    freePtr(lambda_mult);
    freePtr(pos_mult);
    freePtr(blocks_mult_sum);
    freePtr(left_one);
    freePtr(right_one);
    freePtr(pos_one);

    return bCompareResult;
}

void cleanupResultDataLargeMatrix(ResultDataLarge &result)
{

    CUDA_CALL(cudaFree(result.g_num_one));
    CUDA_CALL(cudaFree(result.g_num_blocks_mult));
    CUDA_CALL(cudaFree(result.g_left_one));
    CUDA_CALL(cudaFree(result.g_right_one));
    CUDA_CALL(cudaFree(result.g_pos_one));
    CUDA_CALL(cudaFree(result.g_left_mult));
    CUDA_CALL(cudaFree(result.g_right_mult));
    CUDA_CALL(cudaFree(result.g_left_count_mult));
    CUDA_CALL(cudaFree(result.g_right_count_mult));
    CUDA_CALL(cudaFree(result.g_blocks_mult));
    CUDA_CALL(cudaFree(result.g_blocks_mult_sum));
    CUDA_CALL(cudaFree(result.g_lambda_mult));
    CUDA_CALL(cudaFree(result.g_pos_mult));
}

void cleanupInputData(InputData &input)
{

    freePtr(input.a);
    freePtr(input.b);

    CUDA_CALL(cudaFree(input.g_a));
    input.g_a = NULL;
    CUDA_CALL(cudaFree(input.g_b_raw));
    input.g_b_raw = NULL;
    input.g_b = NULL;
}