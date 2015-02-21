#ifndef CU_HELPERS_H   
#define CU_HELPERS_H
#include <cuComplex.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>


/********************************************************************************
*									MACROS										*
********************************************************************************/
/********************************************************************************
* Macro Name: 		CUFFT_SAFE_CALL												*
* Description:		Used to check CUFFT library calls. Just enclose the call in *
* 					this macro.													*
* Parameters:		--> call: the library call.									*
********************************************************************************/
#define CUDAR_SAFE_CALL( call) do {                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) { 	                                               \
        fprintf(stderr, "CUDA Runtime error in file '%s' in line %i: %s.\n",   \
                __FILE__, __LINE__, _cudaGetErrorEnum(err));            	   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

/********************************************************************************
* Macro Name: 		CUFFT_SAFE_CALL												*
* Description:		Used to check CUFFT library calls. Just enclose the call in *
* 					this macro.													*
* Parameters:		--> call: the library call.									*
********************************************************************************/
#define CUFFT_SAFE_CALL( call) do {                                            \
    cufftResult err = call;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
        fprintf(stderr, "CUFFT error in file '%s' in line %i: %s.\n", 		   \
                __FILE__, __LINE__, _cufftGetErrorEnum(err));            	   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

/********************************************************************************
*						FUNCTION 			PROTOTYPES							*
********************************************************************************/
/********************************************************************************
* Function Name: 	_cudaGetErrorEnum											*
* Description:		This functions decodes the error enumerations produced by   *
*					CUDA Runtime API Calls                                      *
* Parameters:		--> error: the cuda error. This is to be used with a macro  *
*						and never directly. 									*
********************************************************************************/
const char *_cudaGetErrorEnum(cudaError_t error);

/********************************************************************************
* Function Name: 	_cufftGetErrorEnum											*
* Description:		This functions decodes the error enumerations produced by   *
*					CUFFT library calls.                                        *
* Parameters:		--> error: the cufft error. This is to be used with a macro *
						and never directly. 									*
********************************************************************************/
const char *_cufftGetErrorEnum(cufftResult error);

/********************************************************************************
* Function Name: 	m_plot_1d													*
* Description:		This takes in a 1D complex function divided into real and	*
*					Imaginary arrays and plots both final and initial pulse		*
*					On the same graph.											*
* Parameters:		--> R_0: array for initial real part of function.			*
* 					--> I_0: array for final imaginary part of function.		*
* 					--> R: array for final real part of function.				*
* 					--> I: array for imaginary real part of function.			*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void m_plot_1d(double *Re_0, double *Im_0, double *Re, double *Im, double l, int xn, 
					char *filename);

/********************************************************************************
* Function Name: 	m_plot_1df													*
* Description:		This takes in a 1D complex function divided into real and	*
*					Imaginary arrays and plots both final and initial pulse		*
*					On the same graph.											*
* Parameters:		--> R_0: array for initial real part of function.			*
* 					--> I_0: array for final imaginary part of function.		*
* 					--> R: array for final real part of function.				*
* 					--> I: array for imaginary real part of function.			*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void m_plot_1df(float *Re_0, float *Im_0, float *Re, float *Im, float l, int xn, 
					char *filename);
/********************************************************************************
* Function Name: 	cm_plot_1df													*
* Description:		This takes in a complex 1D function and plots both initial	*
*					And final pulse on the same graph.							*
* Parameters:		--> psi_0: complex array for initial pulse					*
* 					--> psi: complex array for final pulse						*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_1df(cuComplex *psi_0, cuComplex *psi, float l, int xn, 
					char *filename);

/********************************************************************************
* Function Name: 	cm_plot_1d													*
* Description:		This takes in a complex 1D function and plots both initial	*
*					And final pulse on the same graph.							*
* Parameters:		--> psi_0: complex array for initial pulse					*
* 					--> psi: complex array for final pulse						*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_1d(cuDoubleComplex *psi_0, cuDoubleComplex *psi, double l, int xn, 
					char *filename);

/********************************************************************************
* Function Name: 	m_plot_2d													*
* Description:		This takes in a 2D function divided into real and imaginary	*
*					Arrays. It also takes in array containing max amplitude per	*
*					Time step. It plots the amplitude versus time step. It also	*
*					Creates 2 surfaces, one for initial pulse and one for		*
*					final pulse.												*
* Parameters:		--> R_0: array for initial real part of function.			*
* 					--> I_0: array for final imaginary part of function.		*
* 					--> R: array for final real part of function.				*
* 					--> I: array for imaginary real part of function.			*
* 					--> max: array containing max value per time step	 		*
*					--> lx: size of x-spatial domain							*
*					--> ly: size of y-spatial domain							*
*					--> xn: number of x nodes									*
*					--> yn: number of y nodes									*
*					--> tn: number of t nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void m_plot_2d(double *Re_0, double *Im_0, double *Re, double *Im, double *max, 
				   double lx, double ly, int xn, int yn, int tn, char *filename);

/********************************************************************************
* Function Name: 	m_plot_2df													*
* Description:		This takes in a 2D function divided into real and imaginary	*
*					Arrays. It also takes in array containing max amplitude per	*
*					Time step. It plots the amplitude versus time step. It also	*
*					Creates 2 surfaces, one for initial pulse and one for		*
*					final pulse.												*
* Parameters:		--> R_0: array for initial real part of function.			*
* 					--> I_0: array for final imaginary part of function.		*
* 					--> R: array for final real part of function.				*
* 					--> I: array for imaginary real part of function.			*
* 					--> max: array containing max value per time step	 		*
*					--> lx: size of x-spatial domain							*
*					--> ly: size of y-spatial domain							*
*					--> xn: number of x nodes									*
*					--> yn: number of y nodes									*
*					--> tn: number of t nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void m_plot_2df(float *Re_0, float *Im_0, float *Re, float *Im, float *max, 
				   float lx, float ly, int xn, int yn, int tn, char *filename);

/********************************************************************************
* Function Name: 	cm_plot_2d													*
* Description:		This takes in a 2D complex function. It also takes in array *
* 					containing max amplitude per time step. It plots the        *
* 					amplitude versus time step. It also	creates 2 surfaces,     *
* 					one for initial pulse and one for initial pulse.			*									*
* Parameters:		--> psi_0: complex initial conditions array					*
* 					--> psi: complex final conditions array						*
* 					--> max: array containing max value per time step	 		*
*					--> lx: size of x-spatial domain							*
*					--> ly: size of y-spatial domain							*
*					--> xn: number of x nodes									*
*					--> yn: number of y nodes									*
*					--> tn: number of t nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_2d(cuDoubleComplex *psi_0, cuDoubleComplex *psi, double *max, 
				   double lx, double ly, int xn, int yn, int tn, char *filename);

/********************************************************************************
* Function Name: 	cm_plot_2df													*
* Description:		This takes in a 2D complex function. It also takes in array *
* 					containing max amplitude per time step. It plots the        *
* 					amplitude versus time step. It also	creates 2 surfaces,     *
* 					one for initial pulse and one for initial pulse.			*									*
* Parameters:		--> psi_0: complex initial conditions array					*
* 					--> psi: complex final conditions array						*
* 					--> max: array containing max value per time step	 		*
*					--> lx: size of x-spatial domain							*
*					--> ly: size of y-spatial domain							*
*					--> xn: number of x nodes									*
*					--> yn: number of y nodes									*
*					--> tn: number of t nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_2df(cuComplex *psi_0, cuComplex *psi, float *max, 
				   float lx, float ly, int xn, int yn, int tn, char *filename);
/********************************************************************************
* Function Name: 	max_index													*
* Description:		This function takes in an array and returns the index of the*
*					Lagest element.												*
* Parameters:		--> arr: pointer to double (array)							*
* 					--> size: size of array										*
* Output:			int, index of largest element in input array				*
********************************************************************************/
int max_index(double *arr, int size);

/********************************************************************************
* Function Name: 	max_indexf													*
* Description:		This function takes in an array and returns the index of the*
*					Lagest element.												*
* Parameters:		--> arr: pointer to double (array)							*
* 					--> size: size of array										*
* Output:			int, index of largest element in input array				*
********************************************************************************/
int max_indexf(float *arr, int size);

/********************************************************************************
* Function Name: 	vtk_3d 		 												*
* Description:		This takes in a 3D function and x,y,z arrays and prints an	*
*					ASCII VTK file for the dataset.								*
* Parameters:		--> x: double array for 1st dimension						*
* 					--> y: double array for 2nd dimension						*
* 					--> z: double array for 3rd dimension						*
* 					--> f: double array for 3D function. This is a 3D array 	*
*							squished into one dimension of size nx*ny*nz.		*
*					--> nx: size of x dimension									*
*					--> ny: size of y dimension									*
*					--> nz: size of z dimension									*
*					--> filename: name of file generated (including .vtk)		*
********************************************************************************/
void vtk_3d(double *x, double *y, double *z, double *f, 
				int xn,	int yn, int zn, char *filename);

/********************************************************************************
* Function Name: 	cmax_psi													*
* Description:		This takes in a complex function and finds the magnitude at *
* 					at each point and saves the maximum in an array for this 	*
* 					time step.
* Parameters:		--> R: array for real part of function.						*
* 					--> I: array for Imaginary part of function.				*
*					--> max: used to save max value of |psi|.					*
*					--> step: the current time step.							*
*					--> size: the size of R and I.								*
********************************************************************************/
void cmax_psi(cuDoubleComplex *psi, double *max, int step, int size);

/********************************************************************************
* Function Name: 	max_psi														*
* Description:		This is the CUDA version of max_psi. This takes in a complex*
*					function's real and imaginary parts, find the magnitude 	*
*					at each point and save the maximum amplitude in an array	*
*					for this time step.											*
* Parameters:		--> d_R: device array, real part of function.				*
* 					--> d_I: device array, Imaginary part of function.			*
*					--> max: host array, used to save max value of |f|.			*
*					--> step: the current time step.							*
*					--> size: the size of d_R and d_I.							*
********************************************************************************/
void max_psi(double *d_Re, double *d_Im, double *max, int step, int size);
                                
/********************************************************************************
* Function Name: 	max_psif													*
* Description:		This is the CUDA version of max_psi. This takes in a complex*
*					function's real and imaginary parts, find the magnitude 	*
*					at each point and save the maximum amplitude in an array	*
*					for this time step.											*
* Parameters:		--> d_R: device array, real part of function.				*
* 					--> d_I: device array, Imaginary part of function.			*
*					--> max: host array, used to save max value of |f|.			*
*					--> step: the current time step.							*
*					--> size: the size of d_R and d_I.							*
********************************************************************************/
void max_psif(float *d_Re, float *d_Im, float *max, int step, int size);

#endif

