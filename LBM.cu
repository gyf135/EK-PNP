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

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>
//#include "poisson.cu"
#include "LBM.h"
#include <cuda_runtime.h>
#define MAX(a, b) (((a) > (b)) ? (a) : (b)) 



__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d)
{
    return (NX*(NY*(d-1)+y)+x);
}

#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// forward declarations of kernels
__global__ void gpu_initialization(double*, double*, double*, double*, double*, double*, double*, double*);
//__global__ void gpu_taylor_green(unsigned int,double*,double*,double*);
__global__ void gpu_init_equilibrium(double*,double*,double*,double*,double*, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void gpu_collide_save(double*,double*,double*,double*,double*,double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*,double,double*);
__global__ void gpu_boundary(double*, double*, double*, double*, double*, double*,double*, double*, double*, double*);
__global__ void gpu_stream(double*, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void gpu_bc_charge(double*, double*, double*, double*, double*, double*);
__global__ void gpu_PBE(double*, double*, double*);
__global__ void gpu_PBE_phi(double*, double*);

/*
__device__ void taylor_green_eval(unsigned int t, unsigned int x, unsigned int y, double *r, double *u, double *v)
{
    double kx = 2.0*M_PI/NX;
    double ky = 2.0*M_PI/NY;
    double td = 1.0/(nu*(kx*kx+ky*ky));
    
    double X = x+0.5;
    double Y = y+0.5;
    double ux = -u_max*sqrt(ky/kx)*cos(kx*X)*sin(ky*Y)*exp(-1.0*t/td);
    double uy =  u_max*sqrt(kx/ky)*sin(kx*X)*cos(ky*Y)*exp(-1.0*t/td);
    double P = -0.25*rho0*u_max*u_max*((ky/kx)*cos(2.0*kx*X)+(kx/ky)*cos(2.0*ky*Y))*exp(-2.0*t/td);
    double rho = rho0+3.0*P;
    
    *r = rho;
    *u = ux;
    *v = uy;
}
*/
__host__ void initialization(double *r, double *c, double *fi, double *u, double *v, double *ex, double *ey, double *temp, double *kx, double *ky, cufftHandle plan)
{
	// blocks in grid
	dim3 grid(NX / nThreads, NY, 1);

	// threads in block
	dim3 threads(nThreads, 1, 1);

	gpu_initialization << <grid, threads >> > (r, c, fi, u, v, ex, ey, temp);
	getLastCudaError("gpu_taylor_green kernel error");

	// Use PB equation as the charge density and electric potential initial conditions
	checkCudaErrors(cudaMalloc((void**)&phi_old_gpu, mem_size_scalar));
	double *phi_old_host = (double*)malloc(mem_size_scalar);

	CHECK(cudaMemcpy(phi_old_host, fi,
		mem_size_scalar, cudaMemcpyDeviceToHost));

	CHECK(cudaMemcpy(phi_old_gpu, phi_old_host,
		mem_size_scalar, cudaMemcpyHostToDevice));

	for (unsigned int i = 0; i <= 500; ++i) {
		gpu_PBE << <grid, threads >> > (c, fi, temp);
		// =========================================================================
		// Fast poisson solver
		// =========================================================================
		fast_Poisson(charge_gpu, T_gpu, kx, ky, plan);
		gpu_PBE_phi << <grid, threads >> > (fi, phi_old_gpu);
		CHECK(cudaMemcpy(phi_old_host, fi,
			mem_size_scalar, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(phi_old_gpu, phi_old_host,
			mem_size_scalar, cudaMemcpyHostToDevice));
	}
	free(phi_old_host);
	checkCudaErrors(cudaFree(phi_old_gpu));
}

__global__ void gpu_PBE_phi(double *fi, double *phi_old) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t sidx = gpu_scalar_index(x, y);
	fi[sidx] = PB_omega*fi[sidx] + (1.0 - PB_omega)*phi_old[sidx];
}

__global__ void gpu_PBE(double *c, double *fi, double *temp) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t sidx = gpu_scalar_index(x, y);
	c[sidx] = chargeinf*exp(-electron*fi[sidx] / kB / roomT);
	temp[sidx] = chargeinf*exp(electron*fi[sidx] / kB / roomT);
}

__global__ void gpu_initialization(double *r, double *c, double *fi, double *u, double *v, double *ex, double *ey, double *temp)
{
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t sidx = gpu_scalar_index(x, y);
	r[sidx]  = rho0;
	c[sidx] = 0.0;// charge0;
	fi[sidx] = voltage;
	u[sidx]  = 0.0;
	v[sidx]  = 0.0;
	ex[sidx] = 0.0;
	ey[sidx] = 0.0;
	temp[sidx] = 0.0;// charge0n;
}
/*
__host__ void taylor_green(unsigned int t, double *r, double *u, double *v)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_taylor_green<<< grid, threads >>>(t,r,u,v);
    getLastCudaError("gpu_taylor_green kernel error");
}

__global__ void gpu_taylor_green(unsigned int t, double *r, double *u, double *v)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    
    size_t sidx = gpu_scalar_index(x,y);
    
    taylor_green_eval(t,x,y,&r[sidx],&u[sidx],&v[sidx]);
}
*/
__host__ void init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *temp0, double *temp1, double *r, double *c,
								double *u, double *v, double *ex, double *ey, double *temp)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_init_equilibrium<<< grid, threads >>>(f0,f1,h0,h1,temp0,temp1, r,c,u,v,ex,ey,temp);
    getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *temp0, double *temp1, double *r, double *c,
										double *u, double *v, double *ex, double *ey, double *temp)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    
    double rho    = r[gpu_scalar_index(x,y)];
    double ux     = u[gpu_scalar_index(x,y)];
    double uy     = v[gpu_scalar_index(x,y)];
	double charge = c[gpu_scalar_index(x, y)];
	double Ex     = ex[gpu_scalar_index(x, y)];
	double Ey     = ey[gpu_scalar_index(x, y)];
	double Temp   = temp[gpu_scalar_index(x, y)];

    // load equilibrium
    // feq_i  = w_i rho [1 + 3(ci . u) + (9/2) (ci . u)^2 - (3/2) (u.u)]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u) + (1/2) (ci . 3u)^2]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u){ 1 + (1/2) (ci . 3u) }]
    
    // temporary variables
    double w0r = w0*rho;
    double wsr = ws*rho;
    double wdr = wd*rho;
	double w0c = w0*charge;
	double wsc = ws*charge;
	double wdc = wd*charge;
	double w0t = w0*Temp;
	double wst = ws*Temp;
	double wdt = wd*Temp;

    double omusq   = 1.0 - 0.5*(ux*ux+uy*uy)/cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey)) / cs_square;
	double omusq_cn = 1.0 - 0.5*((ux + Kn*Ex)*(ux + Kn*Ex) + (uy + Kn*Ey)*(uy + Kn*Ey)) / cs_square;

    
    double tux   = ux / cs_square / CFL;
    double tuy   = uy / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;
	double tux_cn = (ux + Kn*Ex) / cs_square / CFL;
	double tuy_cn = (uy + Kn*Ey) / cs_square / CFL;


    
	// zero weight
    f0[gpu_field0_index(x,y)]    = w0r*(omusq);
	h0[gpu_field0_index(x,y)]    = w0c*(omusq_c);
	temp0[gpu_field0_index(x, y)] = w0t*(omusq_cn);

    
	// adjacent weight
	// flow
    double cidot3u = tux;
    f1[gpu_fieldn_index(x,y,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy;
    f1[gpu_fieldn_index(x,y,2)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux;
    f1[gpu_fieldn_index(x,y,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    f1[gpu_fieldn_index(x,y,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	// charge
	cidot3u = tux_c;
	h1[gpu_fieldn_index(x, y, 1)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	h1[gpu_fieldn_index(x, y, 2)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	h1[gpu_fieldn_index(x, y, 3)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	h1[gpu_fieldn_index(x, y, 4)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	// Temperature
	cidot3u = tux_cn;
	temp1[gpu_fieldn_index(x, y, 1)] = wst*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_cn;
	temp1[gpu_fieldn_index(x, y, 2)] = wst*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_cn;
	temp1[gpu_fieldn_index(x, y, 3)] = wst*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_cn;
	temp1[gpu_fieldn_index(x, y, 4)] = wst*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
    
	// diagonal weight
	// flow
    cidot3u = tux+tuy;
    f1[gpu_fieldn_index(x,y,5)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy-tux;
    f1[gpu_fieldn_index(x,y,6)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -(tux+tuy);
    f1[gpu_fieldn_index(x,y,7)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux-tuy;
    f1[gpu_fieldn_index(x,y,8)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));

	// charge
	cidot3u = tux_c + tuy_c;
	h1[gpu_fieldn_index(x, y, 5)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	h1[gpu_fieldn_index(x, y, 6)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_c + tuy_c);
	h1[gpu_fieldn_index(x, y, 7)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	h1[gpu_fieldn_index(x, y, 8)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// Temperature
	cidot3u = tux_cn + tuy_cn;
	temp1[gpu_fieldn_index(x, y, 5)] = wdt*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_cn - tux_cn;
	temp1[gpu_fieldn_index(x, y, 6)] = wdt*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_cn + tuy_cn);
	temp1[gpu_fieldn_index(x, y, 7)] = wdt*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_cn - tuy_cn;
	temp1[gpu_fieldn_index(x, y, 8)] = wdt*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));

}

__host__ void stream_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2, double *r, double *c,
	double *u, double *v, double *ex, double *ey, double *Temp, double t,double *f0bc)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);


    gpu_collide_save<<< grid, threads >>>(f0,f1,f2, h0, h1, h2, temp0, temp1, temp2, r, c, u,v, ex, ey, Temp, t, f0bc);

	//double *test = (double*)malloc(sizeof(double));
	//CHECK(cudaMemcpy(test, test_gpu, sizeof(double), cudaMemcpyDeviceToHost));
	//printf("%g\n", *test);


	gpu_boundary << < grid, threads >> >(f0, f1, f2, h0, h1, h2, temp0, temp1, temp2, f0bc);
	gpu_stream << < grid, threads >> >(f0, f1, f2, h0, h1, h2, temp0, temp1, temp2);
	gpu_bc_charge << < grid, threads >> >(h0, h1, h2, temp0, temp1, temp2);


    getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2, double *r, double *c,
	double *u, double *v, double *ex, double *ey, double *Temperature, double t,double *f0bc)
{

	// useful constants
	double omega_plus = 1.0 / (nu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_minus = 1.0 / (V / (nu / cs_square / dt) + 1.0 / 2.0) / dt;
	double omega_c_minus = 1.0 / (diffu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_c_plus = 1.0 / (VC / (diffu / cs_square / dt) + 1.0 / 2.0) / dt;
	double omega_T_minus = 1.0 / (D / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_T_plus = 1.0 / (VT / (D / cs_square / dt) + 1.0 / 2.0) / dt;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// storage of f0 at upper and lower plate
	if (y == 0) f0bc[gpu_field0_index(x, 0)]  = f0[gpu_field0_index(x, 0)];    // lower plate
	
	if (y==NY-1) f0bc[gpu_field0_index(x, 1)] = f0[gpu_field0_index(x, NY - 1)]; // upper plate

	// load populations from nodes (ft is the same as f1)
	double ft0    = f0[gpu_field0_index(x, y)];
	double ht0    = h0[gpu_field0_index(x, y)];
	double tempt0 = temp0[gpu_field0_index(x, y)];

	double ft1 = f1[gpu_fieldn_index(x, y, 1)];
	double ft2 = f1[gpu_fieldn_index(x, y, 2)];
	double ft3 = f1[gpu_fieldn_index(x, y, 3)];
	double ft4 = f1[gpu_fieldn_index(x, y, 4)];
	double ft5 = f1[gpu_fieldn_index(x, y, 5)];
	double ft6 = f1[gpu_fieldn_index(x, y, 6)];
	double ft7 = f1[gpu_fieldn_index(x, y, 7)];
	double ft8 = f1[gpu_fieldn_index(x, y, 8)];
	double ht1 = h1[gpu_fieldn_index(x, y, 1)];
	double ht2 = h1[gpu_fieldn_index(x, y, 2)];
	double ht3 = h1[gpu_fieldn_index(x, y, 3)];
	double ht4 = h1[gpu_fieldn_index(x, y, 4)];
	double ht5 = h1[gpu_fieldn_index(x, y, 5)];
	double ht6 = h1[gpu_fieldn_index(x, y, 6)];
	double ht7 = h1[gpu_fieldn_index(x, y, 7)];
	double ht8 = h1[gpu_fieldn_index(x, y, 8)];
	double tempt1 = temp1[gpu_fieldn_index(x, y, 1)];
	double tempt2 = temp1[gpu_fieldn_index(x, y, 2)];
	double tempt3 = temp1[gpu_fieldn_index(x, y, 3)];
	double tempt4 = temp1[gpu_fieldn_index(x, y, 4)];
	double tempt5 = temp1[gpu_fieldn_index(x, y, 5)];
	double tempt6 = temp1[gpu_fieldn_index(x, y, 6)];
	double tempt7 = temp1[gpu_fieldn_index(x, y, 7)];
	double tempt8 = temp1[gpu_fieldn_index(x, y, 8)];

	// compute macroscopic variables from microscopic variables
	double rho = ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8;
	double rhoinv = 1.0 / rho;
	double charge = ht0 + ht1 + ht2 + ht3 + ht4 + ht5 + ht6 + ht7 + ht8;
	double temp = tempt0 + tempt1 + tempt2 + tempt3 + tempt4 + tempt5 + tempt6 + tempt7 + tempt8;
	double Ex = ex[gpu_scalar_index(x, y)];
	double Ey = ey[gpu_scalar_index(x, y)];
	double forcex = 0;// convertCtoCharge*(charge - temp) * (Ex + Ext) + exf;
	double forcey = 0;// convertCtoCharge*(charge - temp) * Ey;

	double ux = rhoinv*((ft1 + ft5 + ft8 - (ft3 + ft6 + ft7)) / CFL + forcex*dt*0.5);
	double uy = rhoinv*((ft2 + ft5 + ft6 - (ft4 + ft7 + ft8)) / CFL + forcey*dt*0.5);

	if (perturb == 1) {
		double xx = x*dx;
		double yy = y*dy-0.5*dy;
		// Rolling patterns
		uy = (cos(2.0*M_PI*yy)-1)*cos(2*M_PI/LL*xx)*0.001;
		ux = LL*sin(2.0*M_PI*yy)*sin(2*M_PI/LL*xx)*0.001;
	}
	else {
		if (y == 0) {
			double ftm0 = f0[gpu_field0_index(x, 1)];
			double htm0 = h0[gpu_field0_index(x, 1)];
			double temptm0 = temp0[gpu_field0_index(x, 1)];
			double ftm1 = f1[gpu_fieldn_index(x, 1, 1)];
			double ftm2 = f1[gpu_fieldn_index(x, 1, 2)];
			double ftm3 = f1[gpu_fieldn_index(x, 1, 3)];
			double ftm4 = f1[gpu_fieldn_index(x, 1, 4)];
			double ftm5 = f1[gpu_fieldn_index(x, 1, 5)];
			double ftm6 = f1[gpu_fieldn_index(x, 1, 6)];
			double ftm7 = f1[gpu_fieldn_index(x, 1, 7)];
			double ftm8 = f1[gpu_fieldn_index(x, 1, 8)];
			double htm1 = h1[gpu_fieldn_index(x, 1, 1)];
			double htm2 = h1[gpu_fieldn_index(x, 1, 2)];
			double htm3 = h1[gpu_fieldn_index(x, 1, 3)];
			double htm4 = h1[gpu_fieldn_index(x, 1, 4)];
			double htm5 = h1[gpu_fieldn_index(x, 1, 5)];
			double htm6 = h1[gpu_fieldn_index(x, 1, 6)];
			double htm7 = h1[gpu_fieldn_index(x, 1, 7)];
			double htm8 = h1[gpu_fieldn_index(x, 1, 8)];
			double temptm1 = temp1[gpu_fieldn_index(x, 1, 1)];
			double temptm2 = temp1[gpu_fieldn_index(x, 1, 2)];
			double temptm3 = temp1[gpu_fieldn_index(x, 1, 3)];
			double temptm4 = temp1[gpu_fieldn_index(x, 1, 4)];
			double temptm5 = temp1[gpu_fieldn_index(x, 1, 5)];
			double temptm6 = temp1[gpu_fieldn_index(x, 1, 6)];
			double temptm7 = temp1[gpu_fieldn_index(x, 1, 7)];
			double temptm8 = temp1[gpu_fieldn_index(x, 1, 8)];

			// compute macroscopic variables from microscopic variables
			double rhom = ftm0 + ftm1 + ftm2 + ftm3 + ftm4 + ftm5 + ftm6 + ftm7 + ftm8;
			double rhoinvm = 1.0 / rhom;
			double chargem = htm0 + htm1 + htm2 + htm3 + htm4 + htm5 + htm6 + htm7 + htm8;
			double tempm = temptm0 + temptm1 + temptm2 + temptm3 + temptm4 + temptm5 + temptm6 + temptm7 + temptm8;
			double Exm = ex[gpu_scalar_index(x, 1)];
			double Eym = ey[gpu_scalar_index(x, 1)];
			double forcexm = 0;// convertCtoCharge*(charge - temp) * (Ex + Ext) + exf;
			double forceym = 0;// convertCtoCharge*(charge - temp) * Ey;

			ux = -rhoinvm*((ftm1 + ftm5 + ftm8 - (ftm3 + ftm6 + ftm7)) / CFL + forcexm*dt*0.5);
			uy = -rhoinvm*((ftm2 + ftm5 + ftm6 - (ftm4 + ftm7 + ftm8)) / CFL + forceym*dt*0.5);
		}
	}
	
	// write to memory (only when visualizing the data)
	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
	c[gpu_scalar_index(x, y)] = charge;
	Temperature[gpu_scalar_index(x, y)] = temp;

	// collision step
	// now compute and relax to equilibrium
	// note that
	// feq_i  = w_i rho [1 + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2 - (1/2) (u.u) / cs_square]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]
	// for charge transport equation, just change u into u + KE
	// heq_i  = w_i charge [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]

	// choices of c
	// cx = [0, 1, 0, -1, 0, 1, -1, -1, 1] / CFL
	// cy = [0, 0, 1, 0, -1, 1, 1, -1, -1] / CFL

	// calculate equilibrium
	// temporary variables
	double w0r = w0*rho;
	double wsr = ws*rho;
	double wdr = wd*rho;
	double w0c = w0*charge;
	double wsc = ws*charge;
	double wdc = wd*charge;
	double w0T = w0*temp;
	double wsT = ws*temp;
	double wdT = wd*temp;

	double omusq = 1.0 - 0.5*(ux*ux + uy*uy) / cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey)) / cs_square;
	double omusq_cn = 1.0 - 0.5*((ux + Kn*Ex)*(ux + Kn*Ex) + (uy + Kn*Ey)*(uy + Kn*Ey)) / cs_square;


	double tux = ux / cs_square / CFL;
	double tuy = uy / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;
	double tux_cn = (ux + Kn*Ex) / cs_square / CFL;
	double tuy_cn = (uy + Kn*Ey) / cs_square / CFL;

	// zero weight
	double fe0 = w0r*(omusq);
	double he0 = w0c*(omusq_c);
	double tempe0 = w0T*(omusq);


	// adjacent weight
	// flow
	double cidot3u = tux;
	double fe1 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy;
	double fe2 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux;
	double fe3 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy;
	double fe4 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	// charge
	cidot3u = tux_c;
	double he1 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	double he2 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	double he3 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	double he4 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	// Temperature
	cidot3u = tux_cn;
	double tempe1 = wsT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_cn;
	double tempe2 = wsT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_cn;
	double tempe3 = wsT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_cn;
	double tempe4 = wsT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));

	// diagonal weight
	// flow
	cidot3u = tux + tuy;
	double fe5 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux;
	double fe6 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux + tuy);
	double fe7 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy;
	double fe8 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	// charge
	cidot3u = tux_c + tuy_c;
	double he5 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	double he6 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_c + tuy_c);
	double he7 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	double he8 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	// Temperature
	cidot3u = tux_cn + tuy_cn;
	double tempe5 = wdT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_cn - tux_cn;
	double tempe6 = wdT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_cn + tuy_cn);
	double tempe7 = wdT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_cn - tuy_cn;
	double tempe8 = wdT*(omusq_cn + cidot3u*(1.0 + 0.5*cidot3u));

	// calculate force population
	// temperory variables
	double coe0 = w0 / cs_square;
	double coes = ws / cs_square;
	double coed = wd / cs_square;

	double cflinv = 1.0 / CFL;

	double fpop0 = coe0*(-ux*forcex - uy*forcey);
	double fpop1 = coes*(((cflinv - ux) + (cflinv*ux)*cflinv / cs_square)*forcex - uy*forcey);
	double fpop2 = coes*(-ux*forcex + ((cflinv - uy) + (cflinv*uy)*cflinv / cs_square)*forcey);
	double fpop3 = coes*(((-cflinv - ux) + (cflinv*ux)*cflinv / cs_square)*forcex - uy*forcey);
	double fpop4 = coes*(-ux*forcex + ((-cflinv - uy) + (cflinv*uy)*cflinv / cs_square)*forcey);

	double cflinv2 = cflinv*cflinv / cs_square;
	double fpop5 = coed*(((cflinv - ux) + (ux + uy)*cflinv2)*forcex + ((cflinv - uy) + (ux + uy)*cflinv2)*forcey);
	double fpop6 = coed*(((-cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((cflinv - uy) + (-ux + uy)*cflinv2)*forcey);
	double fpop7 = coed*(((-cflinv - ux) + (ux + uy)*cflinv2)*forcex + ((-cflinv - uy) + (ux + uy)*cflinv2)*forcey);
	double fpop8 = coed*(((cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((-cflinv - uy) + (-ux + uy)*cflinv2)*forcey);

	// calculate f1 plus and minus
	double fp0 = ft0;
	// adjacent direction
	double fp1 = 0.5 * (ft1 + ft3);
	double fp2 = 0.5 * (ft2 + ft4);
	double fp3 = fp1;
	double fp4 = fp2;
	// diagonal direction
	double fp5 = 0.5 * (ft5 + ft7);
	double fp6 = 0.5 * (ft6 + ft8);
	double fp7 = fp5;
	double fp8 = fp6;

	double fm0 = 0.0;
	// adjacent direction
	double fm1 = 0.5 * (ft1 - ft3);
	double fm2 = 0.5 * (ft2 - ft4);
	double fm3 = -fm1;
	double fm4 = -fm2;
	// diagonal direction
	double fm5 = 0.5 * (ft5 - ft7);
	double fm6 = 0.5 * (ft6 - ft8);
	double fm7 = -fm5;
	double fm8 = -fm6;

	// calculate feq plus and minus
	double fep0 = fe0;
	// adjacent direction
	double fep1 = 0.5 * (fe1 + fe3);
	double fep2 = 0.5 * (fe2 + fe4);
	double fep3 = fep1;
	double fep4 = fep2;
	// diagonal direction
	double fep5 = 0.5 * (fe5 + fe7);
	double fep6 = 0.5 * (fe6 + fe8);
	double fep7 = fep5;
	double fep8 = fep6;

	double fem0 = 0.0;
	// adjacent direction
	double fem1 = 0.5 * (fe1 - fe3);
	double fem2 = 0.5 * (fe2 - fe4);
	double fem3 = -fem1;
	double fem4 = -fem2;
	// diagonal direction
	double fem5 = 0.5 * (fe5 - fe7);
	double fem6 = 0.5 * (fe6 - fe8);
	double fem7 = -fem5;
	double fem8 = -fem6;

	// calculate h1 plus and minus
	double hp0 = ht0;
	// adjacent direction
	double hp1 = 0.5 * (ht1 + ht3);
	double hp2 = 0.5 * (ht2 + ht4);
	double hp3 = hp1;
	double hp4 = hp2;
	// diagonal direction
	double hp5 = 0.5 * (ht5 + ht7);
	double hp6 = 0.5 * (ht6 + ht8);
	double hp7 = hp5;
	double hp8 = hp6;

	double hm0 = 0.0;
	// adjacent direction
	double hm1 = 0.5 * (ht1 - ht3);
	double hm2 = 0.5 * (ht2 - ht4);
	double hm3 = -hm1;
	double hm4 = -hm2;
	// diagonal direction
	double hm5 = 0.5 * (ht5 - ht7);
	double hm6 = 0.5 * (ht6 - ht8);
	double hm7 = -hm5;
	double hm8 = -hm6;

	// calculate heq plus and minus
	double hep0 = he0;
	// adjacent direction
	double hep1 = 0.5 * (he1 + he3);
	double hep2 = 0.5 * (he2 + he4);
	double hep3 = hep1;
	double hep4 = hep2;
	// diagonal direction
	double hep5 = 0.5 * (he5 + he7);
	double hep6 = 0.5 * (he6 + he8);
	double hep7 = hep5;
	double hep8 = hep6;

	double hem0 = 0.0;
	// adjacent direction
	double hem1 = 0.5 * (he1 - he3);
	double hem2 = 0.5 * (he2 - he4);
	double hem3 = -hem1;
	double hem4 = -hem2;
	// diagonal direction
	double hem5 = 0.5 * (he5 - he7);
	double hem6 = 0.5 * (he6 - he8);
	double hem7 = -hem5;
	double hem8 = -hem6;

	// calculate temp1 plus and minus
	double tempp0 = tempt0;
	// adjacent direction
	double tempp1 = 0.5 * (tempt1 + tempt3);
	double tempp2 = 0.5 * (tempt2 + tempt4);
	double tempp3 = tempp1;
	double tempp4 = tempp2;
	// diagonal direction
	double tempp5 = 0.5 * (tempt5 + tempt7);
	double tempp6 = 0.5 * (tempt6 + tempt8);
	double tempp7 = tempp5;
	double tempp8 = tempp6;

	double tempm0 = 0.0;
	// adjacent direction
	double tempm1 = 0.5 * (tempt1 - tempt3);
	double tempm2 = 0.5 * (tempt2 - tempt4);
	double tempm3 = -tempm1;
	double tempm4 = -tempm2;
	// diagonal direction
	double tempm5 = 0.5 * (tempt5 - tempt7);
	double tempm6 = 0.5 * (tempt6 - tempt8);
	double tempm7 = -tempm5;
	double tempm8 = -tempm6;

	// calculate tempeq plus and minus
	double tempep0 = tempe0;
	// adjacent direction
	double tempep1 = 0.5 * (tempe1 + tempe3);
	double tempep2 = 0.5 * (tempe2 + tempe4);
	double tempep3 = tempep1;
	double tempep4 = tempep2;
	// diagonal direction
	double tempep5 = 0.5 * (tempe5 + tempe7);
	double tempep6 = 0.5 * (tempe6 + tempe8);
	double tempep7 = tempep5;
	double tempep8 = tempep6;

	double tempem0 = 0.0;
	// adjacent direction
	double tempem1 = 0.5 * (tempe1 - tempe3);
	double tempem2 = 0.5 * (tempe2 - tempe4);
	double tempem3 = -tempem1;
	double tempem4 = -tempem2;
	// diagonal direction
	double tempem5 = 0.5 * (tempe5 - tempe7);
	double tempem6 = 0.5 * (tempe6 - tempe8);
	double tempem7 = -tempem5;
	double tempem8 = -tempem6;

	// calculate force_plus and force_minus
	double forcep0 = fpop0;
	double forcep1 = 0.5 * (fpop1 + fpop3);
	double forcep2 = 0.5 * (fpop2 + fpop4);
	double forcep3 = forcep1;
	double forcep4 = forcep2;
	double forcep5 = 0.5 * (fpop5 + fpop7);
	double forcep6 = 0.5 * (fpop6 + fpop8);
	double forcep7 = forcep5;
	double forcep8 = forcep6;

	double forcem0 = 0.0;
	double forcem1 = 0.5 * (fpop1 - fpop3);
	double forcem2 = 0.5 * (fpop2 - fpop4);
	double forcem3 = -forcem1;
	double forcem4 = -forcem2;
	double forcem5 = 0.5 * (fpop5 - fpop7);
	double forcem6 = 0.5 * (fpop6 - fpop8);
	double forcem7 = -forcem5;
	double forcem8 = -forcem6;

	double sp = 1.0 - 0.5*dt*omega_plus;
	double sm = 1.0 - 0.5*dt*omega_minus;

	double source0 = sp*fpop0;
	double source1 = sp*forcep1 + sm*forcem1;
	double source2 = sp*forcep2 + sm*forcem2;
	double source3 = sp*forcep3 + sm*forcem3;
	double source4 = sp*forcep4 + sm*forcem4;
	double source5 = sp*forcep5 + sm*forcem5;
	double source6 = sp*forcep6 + sm*forcem6;
	double source7 = sp*forcep7 + sm*forcem7;
	double source8 = sp*forcep8 + sm*forcem8;
	// ===============================================================
	//if (x == 5 && y == 1) {
	//	printf("%2.16g\n", charge);

	//printf("%g\n", source1);

	//}
	// ===============================================================
	// temporary variables (relaxation times)
	double tw0rp = omega_plus*dt;  //   omega_plus*dt 
	double tw0rm = omega_minus*dt; //   omega_minus*dt 
	double tw0cp = omega_c_plus*dt;  //   omega_c_plus*dt 
	double tw0cm = omega_c_minus*dt; //   omega_c_minus*dt 
	double tw0Tp = omega_T_plus*dt;  //   omega_c_plus*dt 
	double tw0Tm = omega_T_minus*dt; //   omega_c_minus*dt 

	// TRT collision operations
	
	f0[gpu_field0_index(x, y)] = ft0 - (tw0rp * (fp0 - fep0) + tw0rm * (fm0 - fem0)) + dt*source0;
	h0[gpu_field0_index(x, y)] = ht0 - (tw0cp * (hp0 - hep0) + tw0cm * (hm0 - hem0));
	temp0[gpu_field0_index(x, y)] = tempt0 - (tw0Tp * (tempp0 - tempep0) + tw0Tm * (tempm0 - tempem0));


	f2[gpu_fieldn_index(x, y, 1)] = ft1 - (tw0rp * (fp1 - fep1) + tw0rm * (fm1 - fem1)) + dt*source1;
	h2[gpu_fieldn_index(x, y, 1)] = ht1 - (tw0cp * (hp1 - hep1) + tw0cm * (hm1 - hem1));
	temp2[gpu_fieldn_index(x, y, 1)] = tempt1 - (tw0Tp * (tempp1 - tempep1) + tw0Tm * (tempm1 - tempem1));
	f2[gpu_fieldn_index(x, y, 2)] = ft2 - (tw0rp * (fp2 - fep2) + tw0rm * (fm2 - fem2)) + dt*source2;
	h2[gpu_fieldn_index(x, y, 2)] = ht2 - (tw0cp * (hp2 - hep2) + tw0cm * (hm2 - hem2));
	temp2[gpu_fieldn_index(x, y, 2)] = tempt2 - (tw0Tp * (tempp2 - tempep2) + tw0Tm * (tempm2 - tempem2));
	f2[gpu_fieldn_index(x, y, 3)] = ft3 - (tw0rp * (fp3 - fep3) + tw0rm * (fm3 - fem3)) + dt*source3;
	h2[gpu_fieldn_index(x, y, 3)] = ht3 - (tw0cp * (hp3 - hep3) + tw0cm * (hm3 - hem3));
	temp2[gpu_fieldn_index(x, y, 3)] = tempt3 - (tw0Tp * (tempp3 - tempep3) + tw0Tm * (tempm3 - tempem3));
	f2[gpu_fieldn_index(x, y, 4)] = ft4 - (tw0rp * (fp4 - fep4) + tw0rm * (fm4 - fem4)) + dt*source4;
	h2[gpu_fieldn_index(x, y, 4)] = ht4 - (tw0cp * (hp4 - hep4) + tw0cm * (hm4 - hem4));
	temp2[gpu_fieldn_index(x, y, 4)] = tempt4 - (tw0Tp * (tempp4 - tempep4) + tw0Tm * (tempm4 - tempem4));
	f2[gpu_fieldn_index(x, y, 5)] = ft5 - (tw0rp * (fp5 - fep5) + tw0rm * (fm5 - fem5)) + dt*source5;
	h2[gpu_fieldn_index(x, y, 5)] = ht5 - (tw0cp * (hp5 - hep5) + tw0cm * (hm5 - hem5));
	temp2[gpu_fieldn_index(x, y, 5)] = tempt5 - (tw0Tp * (tempp5 - tempep5) + tw0Tm * (tempm5 - tempem5));
	f2[gpu_fieldn_index(x, y, 6)] = ft6 - (tw0rp * (fp6 - fep6) + tw0rm * (fm6 - fem6)) + dt*source6;
	h2[gpu_fieldn_index(x, y, 6)] = ht6 - (tw0cp * (hp6 - hep6) + tw0cm * (hm6 - hem6));
	temp2[gpu_fieldn_index(x, y, 6)] = tempt6 - (tw0Tp * (tempp6 - tempep6) + tw0Tm * (tempm6 - tempem6));
	f2[gpu_fieldn_index(x, y, 7)] = ft7 - (tw0rp * (fp7 - fep7) + tw0rm * (fm7 - fem7)) + dt*source7;
	h2[gpu_fieldn_index(x, y, 7)] = ht7 - (tw0cp * (hp7 - hep7) + tw0cm * (hm7 - hem7));
	temp2[gpu_fieldn_index(x, y, 7)] = tempt7 - (tw0Tp * (tempp7 - tempep7) + tw0Tm * (tempm7 - tempem7));
	f2[gpu_fieldn_index(x, y, 8)] = ft8 - (tw0rp * (fp8 - fep8) + tw0rm * (fm8 - fem8)) + dt*source8;
	h2[gpu_fieldn_index(x, y, 8)] = ht8 - (tw0cp * (hp8 - hep8) + tw0cm * (hm8 - hem8));	
	temp2[gpu_fieldn_index(x, y, 8)] = tempt8 - (tw0Tp * (tempp8 - tempep8) + tw0Tm * (tempm8 - tempem8));
}

__global__ void gpu_boundary(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2, double *f0bc)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;

	// Boundary conditions
	double multis = 2.0*rho0*uw / cs_square * ws / CFL;
	double multid = 2.0*rho0*uw / cs_square * wd / CFL;

	// Full way bounce back
	if (y == 0) {
		// lower plate
		f0[gpu_field0_index(x, 0)]    = f0bc[gpu_field0_index(x, 0)];
		f2[gpu_fieldn_index(x, 0, 3)] = f1[gpu_fieldn_index(x, 0, 1)];
		f2[gpu_fieldn_index(x, 0, 4)] = f1[gpu_fieldn_index(x, 0, 2)];
		f2[gpu_fieldn_index(x, 0, 1)] = f1[gpu_fieldn_index(x, 0, 3)];
		f2[gpu_fieldn_index(x, 0, 2)] = f1[gpu_fieldn_index(x, 0, 4)];
		f2[gpu_fieldn_index(x, 0, 7)] = f1[gpu_fieldn_index(x, 0, 5)];
		f2[gpu_fieldn_index(x, 0, 8)] = f1[gpu_fieldn_index(x, 0, 6)];
		f2[gpu_fieldn_index(x, 0, 5)] = f1[gpu_fieldn_index(x, 0, 7)];
		f2[gpu_fieldn_index(x, 0, 6)] = f1[gpu_fieldn_index(x, 0, 8)];
		//if (x == 1) printf("%1.16g\n", f2[gpu_fieldn_index(x, 0, 2)]);
		return;
	}

	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8
	
	if (y ==  NY - 1) {
		// upper plate
		f0[gpu_field0_index(x, NY - 1)]    = f0bc[gpu_field0_index(x, 1)];
		f2[gpu_fieldn_index(x, NY - 1, 3)] = f1[gpu_fieldn_index(x, NY - 1, 1)] -multis;
		f2[gpu_fieldn_index(x, NY - 1, 4)] = f1[gpu_fieldn_index(x, NY - 1, 2)];
		f2[gpu_fieldn_index(x, NY - 1, 1)] = f1[gpu_fieldn_index(x, NY - 1, 3)] + multis;
		f2[gpu_fieldn_index(x, NY - 1, 2)] = f1[gpu_fieldn_index(x, NY - 1, 4)];
		f2[gpu_fieldn_index(x, NY - 1, 7)] = f1[gpu_fieldn_index(x, NY - 1, 5)] - multid;
		f2[gpu_fieldn_index(x, NY - 1, 8)] = f1[gpu_fieldn_index(x, NY - 1, 6)] + multid;
		f2[gpu_fieldn_index(x, NY - 1, 5)] = f1[gpu_fieldn_index(x, NY - 1, 7)] + multid;
		f2[gpu_fieldn_index(x, NY - 1, 6)] = f1[gpu_fieldn_index(x, NY - 1, 8)] - multid;

		// Zero charge gradient on Ny
		//h0[gpu_field0_index(x, NY - 1)]    = h0[gpu_field0_index(x, NY - 2)];
		//h2[gpu_fieldn_index(x, NY - 1, 1)] = h2[gpu_fieldn_index(x, NY - 2, 1)];
		//h2[gpu_fieldn_index(x, NY - 1, 2)] = h2[gpu_fieldn_index(x, NY - 2, 2)];
		//h2[gpu_fieldn_index(x, NY - 1, 3)] = h2[gpu_fieldn_index(x, NY - 2, 3)];
		//h2[gpu_fieldn_index(x, NY - 1, 4)] = h2[gpu_fieldn_index(x, NY - 2, 4)];
		//h2[gpu_fieldn_index(x, NY - 1, 5)] = h2[gpu_fieldn_index(x, NY - 2, 5)];
		//h2[gpu_fieldn_index(x, NY - 1, 6)] = h2[gpu_fieldn_index(x, NY - 2, 6)];
		//h2[gpu_fieldn_index(x, NY - 1, 7)] = h2[gpu_fieldn_index(x, NY - 2, 7)];
		//h2[gpu_fieldn_index(x, NY - 1, 8)] = h2[gpu_fieldn_index(x, NY - 2, 8)];
		return;
	}
}

__global__ void gpu_stream(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2)
{
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// streaming step

	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;

	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8

	// load populations from adjacent nodes (ft is post-streaming population of f1)
	f1[gpu_fieldn_index(x, y, 1)] = f2[gpu_fieldn_index(xm1, y, 1)];
	f1[gpu_fieldn_index(x, y, 2)] = f2[gpu_fieldn_index(x, ym1, 2)];
	f1[gpu_fieldn_index(x, y, 3)] = f2[gpu_fieldn_index(xp1, y, 3)];
	f1[gpu_fieldn_index(x, y, 4)] = f2[gpu_fieldn_index(x, yp1, 4)];
	f1[gpu_fieldn_index(x, y, 5)] = f2[gpu_fieldn_index(xm1, ym1, 5)];
	f1[gpu_fieldn_index(x, y, 6)] = f2[gpu_fieldn_index(xp1, ym1, 6)];
	f1[gpu_fieldn_index(x, y, 7)] = f2[gpu_fieldn_index(xp1, yp1, 7)];
	f1[gpu_fieldn_index(x, y, 8)] = f2[gpu_fieldn_index(xm1, yp1, 8)];

	h1[gpu_fieldn_index(x, y, 1)] = h2[gpu_fieldn_index(xm1, y, 1)];
	h1[gpu_fieldn_index(x, y, 2)] = h2[gpu_fieldn_index(x, ym1, 2)];
	h1[gpu_fieldn_index(x, y, 3)] = h2[gpu_fieldn_index(xp1, y, 3)];
	h1[gpu_fieldn_index(x, y, 4)] = h2[gpu_fieldn_index(x, yp1, 4)];
	h1[gpu_fieldn_index(x, y, 5)] = h2[gpu_fieldn_index(xm1, ym1, 5)];
	h1[gpu_fieldn_index(x, y, 6)] = h2[gpu_fieldn_index(xp1, ym1, 6)];
	h1[gpu_fieldn_index(x, y, 7)] = h2[gpu_fieldn_index(xp1, yp1, 7)];
	h1[gpu_fieldn_index(x, y, 8)] = h2[gpu_fieldn_index(xm1, yp1, 8)];

	temp1[gpu_fieldn_index(x, y, 1)] = temp2[gpu_fieldn_index(xm1, y, 1)];
	temp1[gpu_fieldn_index(x, y, 2)] = temp2[gpu_fieldn_index(x, ym1, 2)];
	temp1[gpu_fieldn_index(x, y, 3)] = temp2[gpu_fieldn_index(xp1, y, 3)];
	temp1[gpu_fieldn_index(x, y, 4)] = temp2[gpu_fieldn_index(x, yp1, 4)];
	temp1[gpu_fieldn_index(x, y, 5)] = temp2[gpu_fieldn_index(xm1, ym1, 5)];
	temp1[gpu_fieldn_index(x, y, 6)] = temp2[gpu_fieldn_index(xp1, ym1, 6)];
	temp1[gpu_fieldn_index(x, y, 7)] = temp2[gpu_fieldn_index(xp1, yp1, 7)];
	temp1[gpu_fieldn_index(x, y, 8)] = temp2[gpu_fieldn_index(xm1, yp1, 8)];
}

__global__ void gpu_bc_charge(double *h0, double *h1, double *h2, double *temp0, double *temp1, double *temp2)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;

	perturb = 0;

	if (y == 0) {
		double multi0c = 2.0*charge0*w0;
		double multisc = 2.0*charge0*ws;
		double multidc = 2.0*charge0*wd;

		double multi0T = 2.0*charge0n*w0;
		double multisT = 2.0*charge0n*ws;
		double multidT = 2.0*charge0n*wd;
		// lower plate for charge density

		double ht1 = h2[gpu_fieldn_index(x, 0, 1)];
		double ht2 = h2[gpu_fieldn_index(x, 0, 2)];
		double ht3 = h2[gpu_fieldn_index(x, 0, 3)];
		double ht4 = h2[gpu_fieldn_index(x, 0, 4)];
		double ht5 = h2[gpu_fieldn_index(x, 0, 5)];
		double ht6 = h2[gpu_fieldn_index(x, 0, 6)];
		double ht7 = h2[gpu_fieldn_index(x, 0, 7)];
		double ht8 = h2[gpu_fieldn_index(x, 0, 8)];
		// lower plate for constant charge density

		h0[gpu_field0_index(x, 0)] = -h0[gpu_field0_index(x, 0)] + multi0c;
		h1[gpu_fieldn_index(x, 0, 3)] = -ht1 + multisc;
		h1[gpu_fieldn_index(x, 0, 4)] = -ht2 + multisc;
		h1[gpu_fieldn_index(x, 0, 1)] = -ht3 + multisc;
		h1[gpu_fieldn_index(x, 0, 2)] = -ht4 + multisc;
		h1[gpu_fieldn_index(x, 0, 7)] = -ht5 + multidc;
		h1[gpu_fieldn_index(x, 0, 8)] = -ht6 + multidc;
		h1[gpu_fieldn_index(x, 0, 5)] = -ht7 + multidc;
		h1[gpu_fieldn_index(x, 0, 6)] = -ht8 + multidc;

		// lower plate for temperature

		double tempt1 = temp2[gpu_fieldn_index(x, 0, 1)];
		double tempt2 = temp2[gpu_fieldn_index(x, 0, 2)];
		double tempt3 = temp2[gpu_fieldn_index(x, 0, 3)];
		double tempt4 = temp2[gpu_fieldn_index(x, 0, 4)];
		double tempt5 = temp2[gpu_fieldn_index(x, 0, 5)];
		double tempt6 = temp2[gpu_fieldn_index(x, 0, 6)];
		double tempt7 = temp2[gpu_fieldn_index(x, 0, 7)];
		double tempt8 = temp2[gpu_fieldn_index(x, 0, 8)];
		// lower plate for constant temperature

		temp0[gpu_field0_index(x, 0)] = -temp0[gpu_field0_index(x, 0)] + multi0T;
		temp1[gpu_fieldn_index(x, 0, 3)] = -tempt1 + multisT;
		temp1[gpu_fieldn_index(x, 0, 4)] = -tempt2 + multisT;
		temp1[gpu_fieldn_index(x, 0, 1)] = -tempt3 + multisT;
		temp1[gpu_fieldn_index(x, 0, 2)] = -tempt4 + multisT;
		temp1[gpu_fieldn_index(x, 0, 7)] = -tempt5 + multidT;
		temp1[gpu_fieldn_index(x, 0, 8)] = -tempt6 + multidT;
		temp1[gpu_fieldn_index(x, 0, 5)] = -tempt7 + multidT;
		temp1[gpu_fieldn_index(x, 0, 6)] = -tempt8 + multidT;

		return;
	}

	if (y == NY - 1) {
		double multi0c = 2.0*charge1*w0;
		double multisc = 2.0*charge1*ws;
		double multidc = 2.0*charge1*wd;

		double multi0T = 2.0*charge1n*w0;
		double multisT = 2.0*charge1n*ws;
		double multidT = 2.0*charge1n*wd;
		// lower plate for charge density

		double ht1 = h2[gpu_fieldn_index(x, y, 1)];
		double ht2 = h2[gpu_fieldn_index(x, y, 2)];
		double ht3 = h2[gpu_fieldn_index(x, y, 3)];
		double ht4 = h2[gpu_fieldn_index(x, y, 4)];
		double ht5 = h2[gpu_fieldn_index(x, y, 5)];
		double ht6 = h2[gpu_fieldn_index(x, y, 6)];
		double ht7 = h2[gpu_fieldn_index(x, y, 7)];
		double ht8 = h2[gpu_fieldn_index(x, y, 8)];
		// lower plate for constant charge density

		h0[gpu_field0_index(x, y)] = -h0[gpu_field0_index(x, y)] + multi0c;
		h1[gpu_fieldn_index(x, y, 3)] = -ht1 + multisc;
		h1[gpu_fieldn_index(x, y, 4)] = -ht2 + multisc;
		h1[gpu_fieldn_index(x, y, 1)] = -ht3 + multisc;
		h1[gpu_fieldn_index(x, y, 2)] = -ht4 + multisc;
		h1[gpu_fieldn_index(x, y, 7)] = -ht5 + multidc;
		h1[gpu_fieldn_index(x, y, 8)] = -ht6 + multidc;
		h1[gpu_fieldn_index(x, y, 5)] = -ht7 + multidc;
		h1[gpu_fieldn_index(x, y, 6)] = -ht8 + multidc;

		// lower plate for temperature

		double tempt1 = temp2[gpu_fieldn_index(x, y, 1)];
		double tempt2 = temp2[gpu_fieldn_index(x, y, 2)];
		double tempt3 = temp2[gpu_fieldn_index(x, y, 3)];
		double tempt4 = temp2[gpu_fieldn_index(x, y, 4)];
		double tempt5 = temp2[gpu_fieldn_index(x, y, 5)];
		double tempt6 = temp2[gpu_fieldn_index(x, y, 6)];
		double tempt7 = temp2[gpu_fieldn_index(x, y, 7)];
		double tempt8 = temp2[gpu_fieldn_index(x, y, 8)];
		// lower plate for constant temperature

		temp0[gpu_field0_index(x, y)] = -temp0[gpu_field0_index(x, y)] + multi0T;
		temp1[gpu_fieldn_index(x, y, 3)] = -tempt1 + multisT;
		temp1[gpu_fieldn_index(x, y, 4)] = -tempt2 + multisT;
		temp1[gpu_fieldn_index(x, y, 1)] = -tempt3 + multisT;
		temp1[gpu_fieldn_index(x, y, 2)] = -tempt4 + multisT;
		temp1[gpu_fieldn_index(x, y, 7)] = -tempt5 + multidT;
		temp1[gpu_fieldn_index(x, y, 8)] = -tempt6 + multidT;
		temp1[gpu_fieldn_index(x, y, 5)] = -tempt7 + multidT;
		temp1[gpu_fieldn_index(x, y, 6)] = -tempt8 + multidT;

		return;
	}
}


__host__ void compute_parameters(double *T, double *M, double *C, double *Fe) {
	double K_host;
	double eps_host;
	double voltage_host;
	double nu_host;
	double Ly_host;
	double diffu_host;
	double charge0_host;
	double rho0_host;


	cudaMemcpyFromSymbol(&K_host, K, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&eps_host, eps, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&voltage_host, voltage, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&nu_host, nu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&Ly_host, Ly, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&diffu_host, diffu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&charge0_host, charge0, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&rho0_host, rho0, sizeof(double), 0, cudaMemcpyDeviceToHost);




	*M = sqrt(eps_host / rho0_host) / K_host;
	*T = eps_host*voltage_host / K_host / nu_host / rho0_host;
	*C = charge0_host * Ly_host * Ly_host / (voltage_host * eps_host);
	*Fe = K_host * voltage_host / diffu_host;
}

__host__ void report_flow_properties(unsigned int n, double t, double *rho, 
	double *charge, double *phi, double *ux, double *uy, double *Ex, double *Ey)
{
    printf("Iteration: %u, physical time: %g.\n",n,t);
}

__host__ void save_scalar(const char* name, double *scalar_gpu, double *scalar_host, unsigned int n)
{
    // assume reasonably-sized file names
    char filename[128];
    char format[16];
    
    // compute maximum number of digits
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    
    // generate format string
    // file name format is name0000nnn.bin
    sprintf(format,"%%s%%0%dd.bin",ndigits);
    sprintf(filename,format,name,n);
    
    // transfer memory from GPU to host
    checkCudaErrors(cudaMemcpy(scalar_host,scalar_gpu,mem_size_scalar,cudaMemcpyDeviceToHost));
    
    // open file for writing
    FILE *fout = fopen(filename,"wb+");
    
    // write data
    fwrite(scalar_host,1,mem_size_scalar,fout);
    
    // close file
    fclose(fout);
    
    if(ferror(fout))
    {
        fprintf(stderr,"Error saving to %s\n",filename);
        perror("");
    }
    else
    {
        if(!quiet)
            printf("Saved to %s\n",filename);
    }
}

__host__
void save_data_tecplot(FILE *fout, double time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *Ex_gpu, double *Ey_gpu, double *Temp_gpu, int first) {
	
	double *rho    = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *phi    = (double*)malloc(mem_size_scalar);
	double *Temp   = (double*)malloc(mem_size_scalar);
	double *ux     = (double*)malloc(mem_size_scalar);
	double *uy     = (double*)malloc(mem_size_scalar);
	double *Ex     = (double*)malloc(mem_size_scalar);
	double *Ey     = (double*)malloc(mem_size_scalar);
	double dx_host;
	double dy_host;
	double *rhodx = (double*)malloc(mem_size_scalar);
	double *rhody = (double*)malloc(mem_size_scalar);
	double *curlForce = (double*)malloc(mem_size_scalar);
	double *vor = (double*)malloc(mem_size_scalar);
	double *vor_diff = (double*)malloc(mem_size_scalar);
	double X = 1000;
	double Re = 170.07/100;
	double extu=0.25;

	//Re = uw_host / nu_host;
	//extu = exf_host * 0.5 * 0.5 * 0.5 / (nu_host * 1600);

	//Re = extu / nu_host;

	//X = 100000 / 1600 / (extu * extu);


	//printf("X = %g and Re = %g.\n", X, Re);

	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho,    rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(phi,    phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Temp,    Temp_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ux,     ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy,     uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ex,     Ex_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ey,     Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	cudaMemcpyFromSymbol(&dx_host, dx, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dy_host, dy, sizeof(double), 0, cudaMemcpyDeviceToHost);

	
	
	for (unsigned int y = 1; y < NY - 1; ++y)
	{
		for (unsigned int x = 1; x < NX - 1; ++x)
		{
			rhodx[scalar_index(x, y)] = (charge[scalar_index(x + 1, y)] - charge[scalar_index(x - 1, y)]) / dx_host / 2;
			rhody[scalar_index(x, y)] = (charge[scalar_index(x, y + 1)] - charge[scalar_index(x, y - 1)]) / dy_host / 2;
		}
	}

	for (unsigned int y = 1; y < NY - 1; ++y)
	{
		rhodx[scalar_index(0, y)] = (charge[scalar_index(1, y)] - charge[scalar_index(NX-1, y)]) / dx_host / 2;
		rhodx[scalar_index(NX - 1, y)] = (charge[scalar_index(0, y)] - charge[scalar_index(NX - 2, y)]) / dx_host / 2;
		rhody[scalar_index(0, y)] = (charge[scalar_index(0, y + 1)] - charge[scalar_index(0, y - 1)]) / dy_host / 2;
		rhody[scalar_index(NX - 1, y)] = (charge[scalar_index(NX - 1, y + 1)] - charge[scalar_index(NX - 1, y - 1)]) / dy_host / 2;
	}



	for (unsigned int y = 1; y < NY - 1; ++y)
	{
		for (unsigned int x = 1; x < NX - 1; ++x)
		{
			vor[scalar_index(x, y)] = (uy[scalar_index(x + 1, y)] - uy[scalar_index(x - 1, y)]) / dx_host / 2 - (ux[scalar_index(x, y + 1)] - ux[scalar_index(x, y - 1)]) / dy_host / 2;
		}
	}

	for (unsigned int y = 1; y < NY - 1; ++y)
	{
		vor[scalar_index(0, y)] = (uy[scalar_index(1, y)] - uy[scalar_index(NX-1, y)]) / dx_host / 2 - (ux[scalar_index(0, y + 1)] - ux[scalar_index(0, y - 1)]) / dy_host / 2;
		vor[scalar_index(NX-1, y)] = (uy[scalar_index(0, y)] - uy[scalar_index(NX-2, y)]) / dx_host / 2 - (ux[scalar_index(NX-1, y + 1)] - ux[scalar_index(NX-1, y - 1)]) / dy_host / 2;
	}


	for (unsigned int y = 2; y < NY - 4; ++y)
	{
		for (unsigned int x = 1; x < NX - 1; ++x)
		{
			vor_diff[scalar_index(x, y)] = (vor[scalar_index(x + 1, y)] + vor[scalar_index(x, y + 1)] + vor[scalar_index(x - 1, y)] + vor[scalar_index(x, y - 1)] - 4 * vor[scalar_index(x, y)]) / (dy_host*dy_host);
		}
	}

	for (unsigned int y = 1; y < NY - 2; ++y)
	{
		vor_diff[scalar_index(0, y)] = (vor[scalar_index(1, y)] + vor[scalar_index(0, y + 1)] + vor[scalar_index(NX-1, y)] + vor[scalar_index(0, y - 1)] - 4 * vor[scalar_index(0, y)]) / (dy_host*dy_host);

		vor_diff[scalar_index(NX - 1, y)] = (vor[scalar_index(0, y)] + vor[scalar_index(NX-1, y + 1)] + vor[scalar_index(NX-2, y)] + vor[scalar_index(NX-1, y - 1)] - 4 * vor[scalar_index(NX-1, y)]) / (dy_host*dy_host);
	}





	// apply boundary conditions (upper and lower plate)
	for (unsigned int x = 0; x < NX; ++x) {
		rho[scalar_index(x, 0)] = 2.0*rho[scalar_index(x, 1)] - rho[scalar_index(x, 2)];
		charge[scalar_index(x, 0)] = 2.0*charge[scalar_index(x, 1)] - charge[scalar_index(x, 2)];
		Temp[scalar_index(x, 0)] = 2.0*Temp[scalar_index(x, 1)] - Temp[scalar_index(x, 2)];
		ux[scalar_index(x, 0)] = 2.0*ux[scalar_index(x, 1)] - ux[scalar_index(x, 2)];
		uy[scalar_index(x, 0)] = 2.0*uy[scalar_index(x, 1)] - uy[scalar_index(x, 2)];
		rhodx[scalar_index(x, 0)] = 2.0*rhodx[scalar_index(x, 1)] - rhodx[scalar_index(x, 2)];
		rhody[scalar_index(x, 0)] = 2.0*rhody[scalar_index(x, 1)] - rhody[scalar_index(x, 2)];
		vor[scalar_index(x, 0)] = 2.0*vor[scalar_index(x, 1)] - vor[scalar_index(x, 2)];
		vor_diff[scalar_index(x, 1)] = vor_diff[scalar_index(x, 2)];
		vor_diff[scalar_index(x, 0)] = vor_diff[scalar_index(x, 1)];


		rho[scalar_index(x, NY - 1)] = 2.0*rho[scalar_index(x, NY - 2)] - rho[scalar_index(x, NY - 3)];
		charge[scalar_index(x, NY - 1)] = 2.0*charge[scalar_index(x, NY - 2)] - charge[scalar_index(x, NY - 3)];
		Temp[scalar_index(x, NY - 1)] = 2.0*Temp[scalar_index(x, NY - 2)] - Temp[scalar_index(x, NY - 3)];
		ux[scalar_index(x, NY - 1)] = 2.0*ux[scalar_index(x, NY - 2)] - ux[scalar_index(x, NY - 3)];
		uy[scalar_index(x, NY - 1)] = 2.0*uy[scalar_index(x, NY - 2)] - uy[scalar_index(x, NY - 3)];
		rhodx[scalar_index(x, NY - 1)] = 2.0*rhodx[scalar_index(x, NY - 2)] - rhodx[scalar_index(x, NY - 3)];
		rhody[scalar_index(x, NY - 1)] = 2.0*rhody[scalar_index(x, NY - 2)] - rhody[scalar_index(x, NY - 3)];
		vor[scalar_index(x, NY - 1)] = 2.0*vor[scalar_index(x, NY - 2)] - vor[scalar_index(x, NY - 3)];
		
		vor_diff[scalar_index(x, NY - 4)] = vor_diff[scalar_index(x, NY - 5)];
		vor_diff[scalar_index(x, NY - 3)] = vor_diff[scalar_index(x, NY - 4)];

		vor_diff[scalar_index(x, NY - 2)] = vor_diff[scalar_index(x, NY - 3)];
		vor_diff[scalar_index(x, NY - 1)] = vor_diff[scalar_index(x, NY - 2)];

	}

	for (unsigned int y = 0; y < NY; ++y)
	{
		for (unsigned int x = 0; x < NX; ++x)
		{
			curlForce[scalar_index(x, y)] = rhodx[scalar_index(x, y)] * Ey[scalar_index(x, y)] - rhody[scalar_index(x, y)] * Ex[scalar_index(x, y)];
			curlForce[scalar_index(x, y)] = curlForce[scalar_index(x, y)] / 1e5;  // Non-dimensionlize
			vor[scalar_index(x, y)] = vor[scalar_index(x, y)] / extu;// *4;  // Non-dimensionlize
			vor_diff[scalar_index(x, y)] = vor_diff[scalar_index(x, y)] / extu;// *4;  // Non-dimensionlize
		}
	}

	if (first)
	{
		char str[] = "VARIABLES=\"x\",\"y\",\"u\",\"v\",\"p\",\"Positive charge\",\"phi\",\"Ex\",\"Ey\",\"vorticity\",\"curlForce\",\"vorticity_diff\",\"XcurlForce\",\"Negative charge\"";
		fprintf(fout, "%s\n", str);
	}
	fprintf(fout, "\n");
	fprintf(fout, "ZONE T=\"t=%g\", F=POINT, I = %d, J = %d\n", time, NX, NY);

	for (unsigned int y = 0; y < NY; ++y)
	{
		for (unsigned int x = 0; x < NX; ++x)
		{
			//double data[] = { dx*x, dy*y, u[scalar_index(x, y)], v[scalar_index(x, y)], r[scalar_index(x, y)], c[scalar_index(x, y)], fi[scalar_index(x, y)], ex[scalar_index(x, y)], ey[scalar_index(x, y)] };
			fprintf(fout, "%g %g %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f %10.12f\n", dx_host*x, dy_host*y,
				ux[scalar_index(x, y)], uy[scalar_index(x, y)], rho[scalar_index(x, y)], charge[scalar_index(x, y)], 
				phi[scalar_index(x, y)], Ex[scalar_index(x, y)], Ey[scalar_index(x, y)], vor[scalar_index(x, y)], curlForce[scalar_index(x, y)], 1/Re*vor_diff[scalar_index(x, y)], X*curlForce[scalar_index(x, y)], Temp[scalar_index(x, y)]);
			//printf("X is %g and Y is %g\n", dx_host*x, dy_host*y);
		}
	}

	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(Ex);
	free(Ey);
}


__host__
void save_data_end(FILE *fend, double time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *Ex_gpu, double *Ey_gpu, double *Temp_gpu) {

	double *rho = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *temp = (double*)malloc(mem_size_scalar);
	double *phi = (double*)malloc(mem_size_scalar);
	double *ux = (double*)malloc(mem_size_scalar);
	double *uy = (double*)malloc(mem_size_scalar);
	double *Ex = (double*)malloc(mem_size_scalar);
	double *Ey = (double*)malloc(mem_size_scalar);

	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho, rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temp, Temp_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(phi, phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ex, Ex_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ey, Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fprintf(fend, "%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n", time,
					ux[scalar_index(x, y)], uy[scalar_index(x, y)], rho[scalar_index(x, y)], charge[scalar_index(x, y)],
					phi[scalar_index(x, y)], Ex[scalar_index(x, y)], Ey[scalar_index(x, y)], temp[scalar_index(x, y)]);
			}
		}

	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(Ex);
	free(Ey);
	free(temp);
}

__host__
void save_data_dmd(FILE *fend, double time, double *ux_gpu, double *uy_gpu, double *charge_gpu, double *phi_gpu) {

	double *uy = (double*)malloc(mem_size_scalar);
	double *ux = (double*)malloc(mem_size_scalar);
	double *ch = (double*)malloc(mem_size_scalar);
	double *fi = (double*)malloc(mem_size_scalar);

	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ch, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fi, phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	// apply boundary conditions (upper and lower plate)
	for (unsigned int x = 0; x < NX; ++x) {
		//uy[scalar_index(x, 0)]          = 2.0*uy[scalar_index(x, 1)] - uy[scalar_index(x, 2)];
		uy[scalar_index(x, NY - 1)] = 2.0*uy[scalar_index(x, NY - 2)] - uy[scalar_index(x, NY - 3)];
		ux[scalar_index(x, NY - 1)] = 2.0*ux[scalar_index(x, NY - 2)] - ux[scalar_index(x, NY - 3)];
		ch[scalar_index(x, NY - 1)] = 2.0*ch[scalar_index(x, NY - 2)] - ch[scalar_index(x, NY - 3)];
	}

		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				//fprintf(fend, "%10.6f %10.6f %10.6f %10.6f %10.6f\n", time, ux[scalar_index(x, y)], uy[scalar_index(x, y)], ch[scalar_index(x, y)], fi[scalar_index(x, y)]);
				fprintf(fend, "%10.6f \n", ch[scalar_index(x, y)]);
			}
		}
	free(uy);
	free(ux);
	free(ch);
	free(fi);
}

__host__
void read_data(double *time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *Ex_gpu, double *Ey_gpu, double *T_gpu) {

	double *rho = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *temp = (double*)malloc(mem_size_scalar);
	double *phi = (double*)malloc(mem_size_scalar);
	double *ux = (double*)malloc(mem_size_scalar);
	double *uy = (double*)malloc(mem_size_scalar);
	double *Ex = (double*)malloc(mem_size_scalar);
	double *Ey = (double*)malloc(mem_size_scalar);

	FILE *fread = fopen("data_end.dat", "r");
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fscanf(fread, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", time,
					&ux[scalar_index(x, y)], &uy[scalar_index(x, y)],  &rho[scalar_index(x, y)], &charge[scalar_index(x, y)],
					&phi[scalar_index(x, y)], &Ex[scalar_index(x, y)], &Ey[scalar_index(x, y)], &temp[scalar_index(x, y)]);
			}
		}
	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho_gpu, rho, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(charge_gpu, charge, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(phi_gpu, phi, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(T_gpu, temp, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ux_gpu, ux, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(uy_gpu, uy, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ex_gpu, Ex, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ey_gpu, Ey, mem_size_scalar, cudaMemcpyHostToDevice));
	fclose(fread);
	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(Ex);
	free(Ey);
	free(temp);
}


__host__
double current(double* c, double* ey) {
	double I = 0;
	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			//rho[scalar_index(x, y, 0)] = 2.0*rho[scalar_index(x, y, 1)] - rho[scalar_index(x, y, 2)];
			c[scalar_index(x, 0)] = 2.0*c[scalar_index(x, 1)] - c[scalar_index(x, 2)];
			//ux[scalar_index(x, y, 0)] = 2.0*ux[scalar_index(x, y, 1)] - ux[scalar_index(x, y, 2)];
			//uy[scalar_index(x, y, 0)] = 2.0*uy[scalar_index(x, y, 1)] - uy[scalar_index(x, y, 2)];
			//uz[scalar_index(x, y, 0)] = 2.0*uz[scalar_index(x, y, 1)] - uz[scalar_index(x, y, 2)];
			//rho[scalar_index(x, y, NZ - 1)] = 2.0*rho[scalar_index(x, y, NZ - 2)] - rho[scalar_index(x, y, NZ - 3)];
			c[scalar_index(x, NY - 1)] = 2.0*c[scalar_index(x, NY - 2)] - c[scalar_index(x, NY - 3)];
			//ux[scalar_index(x, y, NZ - 1)] = 2.0*ux[scalar_index(x, y, NZ - 2)] - ux[scalar_index(x, y, NZ - 3)];
			//uy[scalar_index(x, y, NZ - 1)] = 2.0*uy[scalar_index(x, y, NZ - 2)] - uy[scalar_index(x, y, NZ - 3)];
			//uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}
		for (unsigned int x = 0; x<NX; x++) {
			I += c[scalar_index(x, NY - 1)] * ey[scalar_index(x, NY - 1)];
		}
	I = I * K_host * dy_host;
	return I;
}

__host__
void record_umax(FILE *fend, double time, double *ux_gpu, double *uy_gpu) {

	double *ux = (double*)malloc(mem_size_scalar);
	double *uy = (double*)malloc(mem_size_scalar);
	double umax = 0;


	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	// apply boundary conditions (upper and lower plate)
		for (unsigned int x = 0; x < NX; ++x) {
			ux[scalar_index(x, NY - 1)] = 2.0*ux[scalar_index(x, NY - 2)] - ux[scalar_index(x, NY - 3)];
			uy[scalar_index(x, NY - 1)] = 2.0*uy[scalar_index(x, NY - 2)] - uy[scalar_index(x, NY - 3)];
		}
	
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				//umax = MAX(umax, sqrt(ux[scalar_index(x, y, z)] * ux[scalar_index(x, y, z)] + uy[scalar_index(x, y, z)] * uy[scalar_index(x, y, z)]
				//	+ uz[scalar_index(x, y, z)] * uz[scalar_index(x, y, z)]));
				umax = MAX(umax, uy[scalar_index(x, y)]);
			}
		}

	fprintf(fend, "%10.6f %10.6f\n", time, umax);

	free(ux);
	free(uy);
}



