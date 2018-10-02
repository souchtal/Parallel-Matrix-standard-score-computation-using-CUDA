#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
/*Matrix size N*/
#define N  20

#define CHECK_ERR(x)                                    \
if (x != cudaSuccess) {                               \
fprintf(stderr,"%s in %s at line %d\n",             \
cudaGetErrorString(err),__FILE__,__LINE__);     \
exit(-1);                                               \
}                                                     \
/*host variables for matrices*/
float h_A[N][N];
float h_B[N][N];

__global__ void func ( float *temp_d, float *mean_d, float *SD_d, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n )
        /*d_A[j][i] = temp_d[i];
        /*computations*/
        if(SD_d[i] == 0.0){
            temp_d[i] = 0.0;
        }else{
            temp_d[i] = (temp_d[i] - mean_d[i]) / SD_d[i];
    }
}
/*initializing matrix with random values*/
void initialize_inputs() {
    int row, col;
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            h_A[row][col] = row+1;
            h_B[row][col] = 0.0;
        }
    }
}

int main() {

    cudaError_t err;
    int i,j;

    /*host variables*/
    float transpose_A[N][N];
    float temp2[N];
    float temp_h[N];

    /* Mean and Standard Deviation variables */
    float mean[N];
    float SD[N];
    int row, col;
    float mu, sigma;

    /*device variables*/
    float d_A[N][N];
    float out_A[N][N];
    float *temp_d;
    float *mean_d;
    float *SD_d;

    /*timing variables*/
    struct timeval start, stop;  
    struct timezone tzdummy;
    unsigned long long runtime;
    unsigned long long total_time = 0;

    /*Program begins*/
    initialize_inputs();

    /*compute sd anad mean for every column*/
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += h_A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(h_A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);

        mean[col] = mu;
        SD[col] = sigma;
    }
    /*testing */
    printf("\nAll the means :\n");
    for(i=0;i<N;i++){
        printf("%.2f ",mean[i]);
    }
    printf("\nAll the SDs :\n");
    for(i=0;i<N;i++){
        printf("%.2f ",SD[i]);
    }

    /*transpose matrix so that cols are rows*/
    for (i = 0; i < N; i++){
      for(j = 0 ; j < N ; j++){
         transpose_A[j][i] = h_A[i][j];
      }
    }
    /**testing*/
    printf("\nThe transpose is\n");
    for (i = 0; i < N; i++){
      for(j = 0 ; j < N ; j++){
         printf("%.2f ",transpose_A[i][j]);
      }
    }
    /*********************Allocate memory on device**********************/
    /*allocate temp_d on device*/
    err = cudaMalloc((void **) &temp_d, sizeof(float)*N);
    CHECK_ERR(err);

    /*allocate memory for every column on device*/
    for(i=0;i<N;i++){
        err = cudaMalloc((void **) &d_A[i], sizeof(float)*N);
        CHECK_ERR(err);
    }

    /*allocate memory for means array on device*/
    err = cudaMalloc((void **) &mean_d, sizeof(float)*N);
    CHECK_ERR(err);
    /*allocate memory for SDs array on device*/
    err = cudaMalloc((void **) &SD_d, sizeof(float)*N);
    CHECK_ERR(err);

    /**********************Send data to GPU********************************/
    /*sending means array */
    err = cudaMemcpy(mean_d, mean, sizeof(float)*N, cudaMemcpyHostToDevice);
    CHECK_ERR(err);
    /*sending SDs array */
    err = cudaMemcpy(SD_d, SD, sizeof(float)*N, cudaMemcpyHostToDevice);
    CHECK_ERR(err);
    /*sending columns of the matrix*/

    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            temp_h[j] = transpose_A[i][j];
        }
        err = cudaMemcpy(temp_d, temp_h, sizeof(float)*N, cudaMemcpyHostToDevice);
        CHECK_ERR(err);
        j=0;
        /********** Start Clock ***********/
        cudaDeviceSynchronize();
        gettimeofday(&start, &tzdummy);

        func<<<ceil(N/256.0), 256>>>(temp_d,mean_d,SD_d,N);
        /************stop clock************/
        cudaDeviceSynchronize();
        gettimeofday(&stop, &tzdummy);
        runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);

        total_time = total_time + runtime;
        err = cudaMemcpy(temp2, temp_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

        for(j=0;j<N;j++){
            out_A[i][j] = temp2[j];
        }
    }
    /*************************Print output**********************************/
    printf("\nAfter Normalization :\n");
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            /*printf("%f  ", out_A[i][j] );*/
            printf("%5.2f%s", out_A[j][i], (j < N-1) ? ", " : ";\n\t");
        }
    }

    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)total_time/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");

}
