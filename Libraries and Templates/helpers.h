#ifndef HELPERS_H   
#define HELPERS_H

/********************************************************************************
* Function Name: 	matlab_gen_1d												*
* Description:		This takes in a 1D complex function divided into real and	*
					Imaginary arrays and plots both final and initial pulse		*
					On the same graph.											*
* Parameters:		--> R_0: array for initial real part of function.			*
* 					--> I_0: array for final imaginary part of function.		*
* 					--> R: array for final real part of function.				*
* 					--> I: array for imaginary real part of function.			*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void matlab_gen_1d(double *R_0, double *I_0, double *R, double *I, double l, int xn, *char filename);

/********************************************************************************
* Function Name: 	matlab_cgen_1d												*
* Description:		This takes in a complex 1D function and plots both initial	*
*					And final pulse on the same graph.							*
* Parameters:		--> psi_0: complex array for initial pulse					*
* 					--> psi: complex array for final pulse						*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void matlab_cgen_1d(fftw_complex *psi_0, fftw_complex *psi, double l, int xn, char *filename);

/********************************************************************************
* Function Name: 	matlab_gen_2d												*
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
void matlab_gen_2d(double *R_0, double *I_0, double *R, double *I, double *max, double lx, double ly, int xn, int yn, int tn, char *filename);

/********************************************************************************
* Function Name: 	vtk_gen_3d													*
* Description:		This takes in a 3D function and x,y,z arrays and prints an	*
*					ASCII VTK file for the dataset.								*
* Parameters:		--> x: double array for 1st dimension						*
* 					--> y: double array for 2nd dimension						*
* 					--> z: double array for 3rd dimension						*
* 					--> f: double array for 3D function. This is a 3D array 	*
							squished into one dimension of size nx*ny*nz.		*
*					--> nx: size of x dimension									*
*					--> ny: size of y dimension									*
*					--> nz: size of z dimension									*
*					--> filename: name of file generated (including .vtk)		*
********************************************************************************/
void vtk_gen_3d(double *x, double *y, double *z, double *f, int xn, int yn, int zn, char *filename);

/********************************************************************************
* Function Name: 	max_psi													*
* Description:		This takes in a complex function's real and imaginary parts,*
*					find the magnitude at each point and save the maximum 		*
*					amplitude in an array for this time step.					*
* Parameters:		--> R: array for real part of function.						*
* 					--> I: array for Imaginary part of function.				*
*					--> max: used to save max value of |psi|.					*
*					--> step: the current time step.							*
*					--> size: the size of R and I.								*
********************************************************************************/
void max_psi(double *R, double *I, double *max, int step, int size);

/********************************************************************************
* Function Name: 	cu_max_psi													*
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
void cu_max_psi(double *d_R, double *d_I, double *max, int step, int size);

#endif