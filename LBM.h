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
#ifndef __LBM_H
#define __LBM_H
#include <math.h>
#include <cufft.h>

__device__ double test;

__device__ int perturb = 0;

int iteractionCount = 0;
double *T = (double*)malloc(sizeof(double));
double *M = (double*)malloc(sizeof(double));
double *C = (double*)malloc(sizeof(double));
double *Fe = (double*)malloc(sizeof(double));
double *Ra = (double*)malloc(sizeof(double));
double *Pr = (double*)malloc(sizeof(double));


const unsigned int flag = 0; // if flat == 1, read previous data, otherwise initialize
const int nThreads = 10; // can divide NX

// define grids
const unsigned int NX = 10; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int NY = 101; // number of grid points in y-direction, meaning NY-1 cells
const unsigned int NE = 2 * (NY - 1);
const unsigned int SIZE = NX*NE;
__constant__ double LL = 1.0e-7;
__constant__ double Lx = 1.0e-7;
__constant__ double Ly = 1.0e-6;
__constant__ double dx = 1.0e-6 / 100.0; //need to change according to NX and LX
__constant__ double dy = 1.0e-6 / 100.0; //need to change according to NY and LY

// define physics
double uw_host = 0.0; // velocity of the wall
double exf_host = 0.0;
__device__ double uw;
__device__ double exf;
__constant__ double CFL = 0.01; // CFL = dt/dx
__constant__ double dt = 0.01 * 1.0e-6 / 100.0; // dt = dx * CFL need to change according to dx, dy
__constant__ double cs_square = 1.0 / 3.0 / (0.01*0.01); // 1/3/(CFL^2)
__constant__ double rho0 = 1000.0;
__constant__ double chargeinf = 0.01;
__constant__ double charge0 = 1.2364549e-2;	// positive charge injection lower
__constant__ double charge1 = 1.2364549e-2;	// positive charge injection upper

__constant__ double voltage = -5.2574e-3; // lower plane
double voltage_host;
__constant__ double voltage2 = -5.2574e-3; // upper plane
double voltage2_host;
__constant__ double Ext = 1.0e3;	// External electric field
__constant__ double eps =  6.95e-10;	// positive permittivity
__constant__ double diffu = 1.0e-8;	//charge diffusivity
double nu_host = 0.889e-6;	// Kinematic Viscosity
__device__ double nu = 0.889e-6;
double K_host = 4.245e-7; // ion mobility
__device__ double K;

//Negative charge properties
double D_host = 1.0e-8; // Negative diffusivity
__device__ double D;

double Kn_host = -4.245e-7; // Negative mobility
__device__ double Kn;

double epsn_host = 6.95e-10; // Negative permittivity
__device__ double epsn;


__constant__ double charge0n = 8.087639e-3;// 864.0; // Negative charge injection lower

__constant__ double charge1n = 8.087639e-3;// 864.0; // Negative charge injection upper


__constant__ double NA = 6.022e23; //Avogadro number
__constant__ double kB = 1.38e-23; //Boltzmann constant
__constant__ double electron = 1.6e-19; //Elementary charge
__constant__ double roomT = 273; // Room temperature
__constant__ double convertCtoCharge = 9.64e4; // Conversion factor from ion concentration (mol/m^3) to charge density (C/m^3)
__constant__ double PB_omega = 0.05;





// define scheme
const unsigned int ndir = 9;
const size_t mem_size_0dir = sizeof(double)*NX*NY;
const size_t mem_size_n0dir = sizeof(double)*NX*NY*(ndir - 1);
const size_t mem_size_scalar = sizeof(double)*NX*NY;
const size_t mem_size_ext_scalar = sizeof(double)*NX*NE;

// weights of populations (total 9 for D2Q9 scheme)
__constant__ double w0 = 4.0 / 9.0;  // zero weight
__constant__ double ws = 1.0 / 9.0;  // adjacent weight
__constant__ double wd = 1.0 / 36.0; // diagonal weight

// parameters for (two-relaxation time) TRT scheme
__constant__ double V  = 1.0e-6;
__constant__ double VC = 1.0e-6;
__constant__ double VT = 1.0e-6;


const unsigned int NSTEPS = 100000;
const unsigned int NSAVE  = NSTEPS / 5;
const unsigned int NMSG   =  NSAVE;
const unsigned int NDMD = 5000000000000;
const unsigned int printCurrent = 2000;


// physical time
double t;
double *f0_gpu, *f1_gpu, *f2_gpu;
double *h0_gpu, *h1_gpu, *h2_gpu;
double *temp0_gpu, *temp1_gpu, *temp2_gpu;
double *rho_gpu, *ux_gpu, *uy_gpu;
double *charge_gpu, *phi_gpu, *T_gpu;
double *Ex_gpu, *Ey_gpu;
double *kx, *ky;
double *phi_old_gpu;

cufftHandle plan = 0;
cufftDoubleComplex *freq_gpu_ext, *charge_gpu_ext, *phi_gpu_ext, *T_gpu_ext;
double *f0bc; // store f0 of the lower plate for further use
double *kx_host = (double*)malloc(sizeof(double)*NX);
double *ky_host = (double*)malloc(sizeof(double)*NY);
double dt_host;
double Lx_host;
double Ly_host;
double dy_host;
double *charge_host = (double*)malloc(mem_size_scalar);
double *Ey_host = (double*)malloc(mem_size_scalar);


// suppress verbose output
const bool quiet = true;

void initialization(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, cufftHandle);
void read_data(double*, double*, double*, double*, double*, double*, double*, double*, double*);
void init_equilibrium(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);


void stream_collide_save(double*,double*,double*,double*,double*,double*, double*, double*, double*, double*,double*,double*, double*, double*, double*, double*, double,double*);
void report_flow_properties(unsigned int,double,double*,double*,double*,double*, double*, double*, double*);
void save_scalar(const char*,double*,double*,unsigned int);
void save_data_tecplot(FILE*, double, double*, double*, double*, double*, double*, double*, double*, double*, int);
void save_data_end(FILE*, double, double*, double*, double*, double*, double*, double*, double*, double*);
void compute_parameters(double*, double*, double*, double*);
void poisson_phi(double*, double*);
void extension(double*, cufftDoubleComplex*, double*);
void efield(double*, double*, double*);
void derivative(double*, double*, cufftDoubleComplex*);
void extract(double*, cufftDoubleComplex*);
void fast_Poisson(double*, double*, double*, double*, cufftHandle);

double current(double*, double*);
void record_umax(FILE*, double, double*, double*);
void save_data_dmd(FILE*, double, double*, double*, double*, double*);


inline size_t scalar_index(unsigned int x, unsigned int y)
{
	return NX*y + x;
}

inline size_t fieldn_index(unsigned int x, unsigned int y, unsigned int d)
{
	return (NX*(NY*(d - 1) + y) + x);
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
#endif /* __LBM_H */

