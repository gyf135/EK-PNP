/* This code accompanies
*   Two relaxation time lattice Boltzmann method coupled to fast Fourier transform Poisson solver: Application to electroconvective flow, Journal of Computational Physics
*	 https://doi.org/10.1016/j.jcp.2019.07.029
*	 Numerical analysis of electroconvection in cross-flow with unipolar charge injection, Physical Review Fluids
*	 https://doi.org/10.1103/PhysRevFluids.4.103701
*
*   Yifei Guan, Igor Novosselov
* 	 University of Washington
*
* Author: Yifei Guan
*
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include "LBM.h"
#include <device_functions.h>
#define RAD 1
__global__ void gpu_poisson(double*, double*,double*);
__global__ void gpu_efield(double*, double*, double*);
__global__ void odd_extension(double*, cufftDoubleComplex*, double*);
__global__ void gpu_derivative(double*, double*, cufftDoubleComplex*);
__global__ void odd_extract(double*, cufftDoubleComplex*);
__global__ void gpu_bc(double*);
__device__ __forceinline__ size_t gpu_s_scalar_index(unsigned int x, unsigned int y)
{
	return (2*RAD + nThreads)*y + x;
}

__host__
void poisson_phi(double *charge_gpu, double *phi_gpu)
{
	// blocks in grid
	dim3  grid(NX / nThreads, NY, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	unsigned int it = 0;
	double MAX_ITERATIONS = 1.0E6;
	double TOLERANCE = 1.0e-9;
	double *Res = (double*)malloc(mem_size_scalar);
	double error = 0.0;
	double *R;
	checkCudaErrors(cudaMalloc((void**)&R, mem_size_scalar));
	for (it = 0; it < MAX_ITERATIONS; ++it) {
		error = 0.0;
		gpu_poisson << < grid, threads >> > (charge_gpu, phi_gpu, R);
		checkCudaErrors(cudaMemcpy(Res, R, mem_size_scalar, cudaMemcpyDeviceToHost));
		for (unsigned int y = 0; y < NY; ++y) {
			for (unsigned int x = 0; x < NX; ++x) {
				//if (it % 1000 == 1) 	printf("%g\n", error);
				if (error < Res[scalar_index(x, y)]) error = Res[scalar_index(x, y)];
			}
		}
		if (error < TOLERANCE) break;
	}
	checkCudaErrors(cudaFree(R));
	free(Res);

	//printf("%g\n", error);
	if (it == MAX_ITERATIONS) {
		printf("Poisson solver did not converge!\n");
		printf("Residual = %g\n", error);
		system("pause");
		//exit(-1);
	}
	getLastCudaError("Poisson solver kernel error");
}

__global__ void gpu_poisson(double *c, double *fi,double *R){
	unsigned int y   = blockIdx.y;
	unsigned int x   = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int s_y = threadIdx.y + RAD;
	unsigned int s_x = threadIdx.x + RAD;
	unsigned int xp1 = (x + blockDim.x) % NX;
	unsigned int yp1 = (y + blockDim.y) % NY;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	__shared__ double s_in[(2*RAD + nThreads)*3];
	// load to shared memory (regular cells)
	s_in[gpu_s_scalar_index(s_x,s_y)] = fi[gpu_scalar_index(x, y)];

	// load halo cells
	if (threadIdx.x < RAD) {
		s_in[gpu_s_scalar_index(s_x - RAD, s_y)] = fi[gpu_scalar_index(xm1, y)];
		s_in[gpu_s_scalar_index(s_x + blockDim.x, s_y)] = fi[gpu_scalar_index(xp1, y)];
	}
	if (threadIdx.y < RAD) {
		s_in[gpu_s_scalar_index(s_x, s_y - RAD)] = fi[gpu_scalar_index(x, ym1)];
		s_in[gpu_s_scalar_index(s_x, s_y + blockDim.y)] = fi[gpu_scalar_index(x, yp1)];
	}
	// Boundary conditions
	if (y == 0) {
		fi[gpu_scalar_index(x, y)] = voltage;
		return;
	}
	if (y == NY - 1) {
		fi[gpu_scalar_index(x, y)] = voltage2;
		return;
	}
	__syncthreads();

	double charge    = c[gpu_scalar_index(x, y)];
	//double phi       = fi[gpu_scalar_index(x, y)];
	//double phiL      = fi[gpu_scalar_index(xm1, y)];
	//double phiR      = fi[gpu_scalar_index(xp1, y)];
	//double phiU      = fi[gpu_scalar_index(x, yp1)];
	//double phiD      = fi[gpu_scalar_index(x, ym1)];

	double phi  = s_in[gpu_s_scalar_index(s_x, s_y)];
	double phiL = s_in[gpu_s_scalar_index(s_x-1, s_y)];
	double phiR = s_in[gpu_s_scalar_index(s_x+1, s_y)];
	double phiU = s_in[gpu_s_scalar_index(s_x, s_y+1)];
	double phiD = s_in[gpu_s_scalar_index(s_x, s_y-1)];

	double source    = (charge / eps) * dx *dx; // Right hand side of the equation
	double phi_old   = phi;
	phi = 0.25 * (phiL + phiR + phiU + phiD + source);
	// Record the error
	R[gpu_scalar_index(x, y)] = fabs(phi - phi_old);
	
	//__syncthreads();
	fi[gpu_scalar_index(x, y)] = phi;
	//if (x == 5 && y == 5) printf("%g\n", phi);
}

__host__
void efield(double *phi_gpu, double *Ex_gpu, double *Ey_gpu) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	gpu_efield << < grid, threads >> > (phi_gpu, Ex_gpu, Ey_gpu);
	gpu_bc << <grid, threads >> > (Ey_gpu);
	getLastCudaError("Efield kernel error");

}

__global__ void gpu_efield(double *fi, double *ex, double *ey){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	double phi  = fi[gpu_scalar_index(x, y)];
	double phiL = fi[gpu_scalar_index(xm1, y)];
	double phiR = fi[gpu_scalar_index(xp1, y)];
	double phiU = fi[gpu_scalar_index(x, yp1)];
	double phiD = fi[gpu_scalar_index(x, ym1)];
	ex[gpu_scalar_index(x, y)] = 0.5*(phiL - phiR) / dx;
	ey[gpu_scalar_index(x, y)] = 0.5*(phiD - phiU) / dy;
}
__global__ void gpu_bc(double *ey) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y == 0) {
		//ex[gpu_scalar_index(x, 0)] = ex[gpu_scalar_index(x, 1)];
		ey[gpu_scalar_index(x, 0)] = ey[gpu_scalar_index(x, 1)];
		return;
	}
	if (y == NY - 1) {
		//ex[gpu_scalar_index(x, NY - 1)] = ex[gpu_scalar_index(x, NY - 2)];
		ey[gpu_scalar_index(x, NY - 1)] = ey[gpu_scalar_index(x, NY - 2)];
		return;
	}
}


// =========================================================================
// Fast poisson solver domain extension
// =========================================================================
__host__ void fast_Poisson(double *charge_gpu, double *T_gpu, double *kx, double *ky, cufftHandle plan) {

	checkCudaErrors(cudaMalloc((void**)&freq_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));
	checkCudaErrors(cudaMalloc((void**)&charge_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));
	checkCudaErrors(cudaMalloc((void**)&T_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));
	// Extend the domain
	extension(charge_gpu, charge_gpu_ext, T_gpu);

	// Execute a real-to-complex 2D FFT
	CHECK_CUFFT(cufftExecZ2Z(plan, charge_gpu_ext, freq_gpu_ext, CUFFT_FORWARD));

	// Execute the derivatives in frequency domain
	derivative(kx, ky, freq_gpu_ext);

	// Execute a complex-to-complex 2D IFFT
	CHECK_CUFFT(cufftExecZ2Z(plan, freq_gpu_ext, phi_gpu_ext, CUFFT_INVERSE));

	// Extraction of phi from extended domain phi_gpu_ext
	extract(phi_gpu, phi_gpu_ext);

	// Calculate electric field strength
	efield(phi_gpu, Ex_gpu, Ey_gpu);

	checkCudaErrors(cudaFree(charge_gpu_ext));
	checkCudaErrors(cudaFree(phi_gpu_ext));
	checkCudaErrors(cudaFree(freq_gpu_ext));
	checkCudaErrors(cudaFree(T_gpu_ext));
}


__host__ void extension(double *c, cufftDoubleComplex *c_ext, double *T) {
	// blocks in grid
	dim3  grid(NX / nThreads, NE, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extension << < grid, threads >> > (c, c_ext, T);
	getLastCudaError("Odd Extension error");
}

__global__ void odd_extension(double *charge, cufftDoubleComplex *charge_ext, double *T) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y == 0) {
		charge_ext[gpu_scalar_index(x, y)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y == 1) {
		charge_ext[gpu_scalar_index(x, y)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y)] - T[gpu_scalar_index(x, y)]) / eps - voltage / dy / dy;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y > 1 && y < NY - 2) {
		charge_ext[gpu_scalar_index(x, y)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y)] - T[gpu_scalar_index(x, y)]) / eps;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y == NY - 2) {
		charge_ext[gpu_scalar_index(x, y)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y)] - T[gpu_scalar_index(x, y)]) / eps - voltage2 / dy / dy;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y == NY - 1) {
		charge_ext[gpu_scalar_index(x, y)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y == NY) {
		charge_ext[gpu_scalar_index(x, y)].x = convertCtoCharge*(charge[gpu_scalar_index(x, 1)] - T[gpu_scalar_index(x, 1)]) / eps + voltage2 / dy / dy;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y > NY && y<NE-1) {
		charge_ext[gpu_scalar_index(x, y)].x = convertCtoCharge*(charge[gpu_scalar_index(x, NE - y)] - T[gpu_scalar_index(x, NE - y)]) / eps;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
	if (y == NE - 1) {
		charge_ext[gpu_scalar_index(x, y)].x = convertCtoCharge*(charge[gpu_scalar_index(x, 1)] - T[gpu_scalar_index(x, 1)]) / eps + voltage / dy / dy;
		charge_ext[gpu_scalar_index(x, y)].y = 0.0;
		return;
	}
}

__host__ void derivative(double *kx, double *ky, cufftDoubleComplex *source) {
	// blocks in grid
	dim3  grid(NX / nThreads, NE, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	gpu_derivative << < grid, threads >> > (kx, ky, source);
	getLastCudaError("Gpu derivative error");
}
 
__global__ void gpu_derivative(double *kx, double *ky, cufftDoubleComplex *source) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	double I = kx[x];
	double J = ky[y];
	double mu = (4.0 / dy / dy)*(sin(J*dy*0.5)*sin(J*dy*0.5)) + I*I;
	if (y == 0 && x == 0) mu = 1.0;
	source[gpu_scalar_index(x, y)].x = -source[gpu_scalar_index(x, y)].x / mu;
	source[gpu_scalar_index(x, y)].y = -source[gpu_scalar_index(x, y)].y / mu;
}

__host__ void extract(double *fi, cufftDoubleComplex *fi_ext) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extract << < grid, threads >> > (fi, fi_ext);
	getLastCudaError("Odd Extension error");
}

__global__ void odd_extract(double *phi, cufftDoubleComplex *phi_ext) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y == 0) {
		phi[gpu_scalar_index(x, y)] = voltage;
		return;
	}
	if (y == NY-1) {
		phi[gpu_scalar_index(x, y)] = voltage2;
		return;
	}
	phi[gpu_scalar_index(x, y)] = phi_ext[gpu_scalar_index(x, y)].x/SIZE;
}