// the subroutine for GPU code can be found in several separated text file from the Brightspace. 
// You can add these subroutines to this main code.
////////////////////////////////////////////


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"


const int BLOCK_SIZE =64;  // number of threads per block

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_Lamda = NULL;
float* d_Lamda = NULL;
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;

// Variables to change
int GlobalSize =2000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
int BlockSize = BLOCK_SIZE;            // number of threads in each block
const float EPS = 0.000005;    // tolerence of the error
int max_iteration = 100;       // the maximum iteration steps


// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  ParseArguments(int, char**);
void checkCardVersion(void);
void checkCudaError(cudaError_t, const char[], int);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N); 
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float * g_Lamda,int N);


void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
			
		}
	}
}

void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];

	normW = sqrt(normW);

	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];
	
	return lamda;
}

void RunCPUPowerMethod()
{
	printf("*************************************\n");
	float oldLamda =0;
	float lamda=0;
	
	//AvProduct
	CPU_AvProduct();
	
	//power loop
    int i;
	for (i=0;i<max_iteration;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;	
	
	}
	printf("CPU lamda at %d: %f \n", i, lamda);
	printf("*************************************\n");
	
}


// Host code
int main(int argc, char** argv)
{

    struct timespec t_start,t_end;
    double runtime;
    ParseArguments(argc, argv);
		
    int N = GlobalSize;
    printf("Matrix size %d X %d ||Blocksize: %i \n", N, N, BlockSize);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);
  
    // Allocate normalized value in host memory
    h_NormW = (float*)malloc(norm_size);
    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);
    // Allocate lamda value in host memory
    h_Lamda = (float *)malloc(norm_size);


    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);

    printf("Power method in CPU starts\n");	   
    clock_gettime(CLOCK_REALTIME,&t_start);
    RunCPUPowerMethod();   // the lamda is already solved here
    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    printf("CPU: run time = %f secs.\n",runtime);
    printf("Power method in CPU is finished\n");
    
    
    /////////////////////////////////////////////////
    // This is the starting points of GPU
    printf("Power method in GPU starts\n");
    checkCardVersion();

    // Initialize input matrix
    InitOne(h_VecV,N);

    clock_gettime(CLOCK_REALTIME,&t_start);  // Here I start to count

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;   
    int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared   
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    


    cudaError_t cuda_err;
    // Allocate matrix and vectors in device memory
    cuda_err = cudaMalloc((void**)&d_MatA, mat_size);
    checkCudaError(cuda_err, "Error Allocating Matrix", 1);
    cuda_err = cudaMalloc((void**)&d_VecV, vec_size);
    checkCudaError(cuda_err, "Error Allocating Vector", 1);
    cuda_err = cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
    checkCudaError(cuda_err, "Error Allocating Normal Vector", 1);
    cuda_err = cudaMalloc((void**)&d_NormW, norm_size); 
    checkCudaError(cuda_err, "Error Value for Normallised Eigen Vector", 1);
    cuda_err = cudaMalloc((void**)&d_Lamda, norm_size);
    checkCudaError(cuda_err, "Error Allocating Lamda", 1);


    cuda_err = cudaMemset(d_VecW, 0, vec_size);
    checkCudaError(cuda_err, "Error Setting Vector Size", 1);

    //Copy from host memory to device memory
    cuda_err = cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
    checkCudaError(cuda_err, "Error Copying host matrix to device", 1);
    cuda_err = cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
    checkCudaError(cuda_err, "Error Copying host vector to device", 1);
	// cutilCheckError(cutStopTimer(timer_mem));
	  
   //Power method loops
    float OldLamda =0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float timeWithoutMemoryCopy = 0;
    float timeGpuMilliSec = 0;
    
    cudaEventRecord(start);
    Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGpuMilliSec, start, stop);
    timeWithoutMemoryCopy += timeGpuMilliSec;

    cuda_err = cudaGetLastError();
    checkCudaError(cuda_err, "Sync Error with Av_Product", 1);
    cuda_err = cudaDeviceSynchronize();
    checkCudaError(cuda_err, "Async Error with Av_Product", 1);

    int idx;
    for (idx = 0; idx < max_iteration; idx++) {

        cuda_err = cudaMemset(d_NormW, 0, norm_size);
        
        cudaEventRecord(start);
        FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize >>> (d_VecW, d_NormW, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeGpuMilliSec, start, stop);
        timeWithoutMemoryCopy += timeGpuMilliSec;

        cuda_err = cudaGetLastError();
        checkCudaError(cuda_err, "Sync Error with FindNormW", 1);
        cuda_err = cudaThreadSynchronize();
        checkCudaError(cuda_err, "Async Error with FindNormW", 1);  
        cuda_err = cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
        checkCudaError(cuda_err, "Error copying NormW to Host", 1);

        
        h_NormW[0] = sqrt(h_NormW[0]);
        cuda_err = cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
        checkCudaError(cuda_err, "Error Setting new value of NormW on Device", 1);

        cudaEventRecord(start);
        NormalizeW<<<blocksPerGrid, threadsPerBlock, sharedMemSize >>> (d_VecW, d_NormW, d_VecV, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeGpuMilliSec, start, stop);
        timeWithoutMemoryCopy += timeGpuMilliSec;

        cuda_err = cudaGetLastError();
        checkCudaError(cuda_err, "Sync Error with Normalize W", 1);
        cuda_err = cudaThreadSynchronize();
        checkCudaError(cuda_err, "Async Error with NormalizeW", 1);

        cudaEventSynchronize(start);        
        Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
        cudaEventSynchronize(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeGpuMilliSec, start, stop);
        timeWithoutMemoryCopy += timeGpuMilliSec;

        cuda_err = cudaGetLastError();
        checkCudaError(cuda_err, "Sync Error with Av_Product", 1);
        cuda_err = cudaDeviceSynchronize();
        checkCudaError(cuda_err, "Async Error with Av_Product", 1);
        
        cuda_err = cudaMemset(d_Lamda, 0, norm_size);
        checkCudaError(cuda_err, "Error Setting value of lamda to zero", 1);
        
        cudaEventSynchronize(start);
        ComputeLamda<<<blocksPerGrid, threadsPerBlock, sharedMemSize >>>  (d_VecV, d_VecW, d_Lamda, N);
        cudaEventSynchronize(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeGpuMilliSec, start, stop);
        timeWithoutMemoryCopy += timeGpuMilliSec;
        cuda_err = cudaGetLastError();
        checkCudaError(cuda_err, "Sync Error with Compute Lamda", 1);

        cuda_err = cudaThreadSynchronize();
        checkCudaError(cuda_err, "Async Error with Compute Lamda", 1);

        cuda_err = cudaMemcpy(h_Lamda, d_Lamda, norm_size, cudaMemcpyDeviceToHost);
        checkCudaError(cuda_err, "Error copying device lamda to host", 1);


        if(abs(OldLamda - h_Lamda[0]) < EPS)
			break;
        OldLamda = h_Lamda[0];
    }
	
    printf("GPU lamda at %d: %f \n", idx, h_Lamda[0]);
    
    

    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    printf("GPU: run time = %f secs.\n",runtime);

    printf("GPU: Run time without MemCopy = %f secs.\n", timeWithoutMemoryCopy/1000);

    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	if (d_NormW)
		  cudaFree(d_NormW);
    if (d_Lamda)
        cudaFree(d_Lamda);
		
    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
    if (h_NormW)
        free(h_NormW);
    if (h_Lamda)
        free(h_Lamda);
    
    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++)
    {
    	data[i] = (int) (3*rand() % (int)(101));//1;//value;
	    value ++; if(value>n) value =1;
      // data[i] = 1;
    }
}

// Obtain program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) 
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
            GlobalSize = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
        {
            max_iteration = atoi(argv[i+1]);
		    i = i + 1;
        }
    }
}


void checkCardVersion()
{
   cudaDeviceProp prop;
   
   cudaGetDeviceProperties(&prop, 0);
   
   printf("This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}

/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows:

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory,
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In eacg step, two subvectors with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/


__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;

    int aEnd   = aBegin + N - 1;
    int step  = BLOCK_SIZE;

    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {

        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

        __shared__ float bs[BLOCK_SIZE];


        for (int aa = 0; aa < BLOCK_SIZE;aa+= 1)
        {
            aIndex = a+tx+aa*N;
            if( aIndex < N*N)
        	    As[tx+aa*BLOCK_SIZE] = g_MatA[aIndex];
		        else
        	    As[tx+aa*BLOCK_SIZE] = 0;
        }

        bIndex = b+tx;
   	    if(bIndex<N)
		      bs[tx] = g_VecV[bIndex];
	      else
		      bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[k+tx*BLOCK_SIZE] * bs[k];
        }//}
        __syncthreads();
    }

    g_VecW[ BLOCK_SIZE * bx + tx] = Csub;
}


__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdataVW[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdataVW[tid] =  g_VecV[globalid] * g_VecW[globalid];
  }
  else {
     sdataVW[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdataVW[tid] = sdataVW[tid] + sdataVW[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_Lamda,sdataVW[0]);
}


__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sNormData[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;


  if(tid==0) sNormData[0] =  g_NormW[0];
  __syncthreads();

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid]/sNormData[0];
  }

}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] =  g_VecW[globalid];
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_NormW,sdata[0]);
}

void checkCudaError(cudaError_t cuda_err, const char mesg[], int terminate) {
    bool isError = cuda_err != cudaSuccess;
    if (isError) {
        printf("Reason for Error: %s\n", cudaGetErrorString(cuda_err));
        printf("%s\n", mesg);
    }


    if (isError && terminate) {
        exit(1);
    }
}