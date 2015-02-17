#include "helpers.h"
/********************************************************************************
* Function Name: 	m_plot_1d													*
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
void m_plot_1d(double *Re_0, double *Im_0, double *Re, double *Im, 
				   double l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f, '--b', 'LineWidth', 1);\n"
						 "legend('Initial', 'Final', 0);\n"
						 "title('Initial and Final Pulse');\n"
						 "xlabel('x values'); ylabel('|f|');\n");
	fclose(matlab_file);
}

/********************************************************************************
* Function Name: 	cm_plot_1d 		 											*
* Description:		This takes in a complex 1D function and plots both initial	*
*					And final pulse on the same graph.							*
* Parameters:		--> psi_0: complex array for initial pulse					*
* 					--> psi: complex array for final pulse						*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_1d(complex double *f_0, complex double *f, 
				    double l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(f_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(f[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f, '--b', 'LineWidth', 1);\n"
						 "legend('Initial', 'Final', 0);\n"
						 "title('Initial and Final Pulse');\n"
						 "xlabel('x values'); ylabel('|f|');");
	fclose(matlab_file);
}

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
				   double lx, double ly, int xn, int yn, int tn, char *filename)
{
    FILE *matlab_file = fopen(filename, "w");

	// Initialize Arrays
	fprintf(matlab_file, "[x,y] = meshgrid(linspace(%f,%f,%d), linspace(%f, %f, %d));\n", 
																-lx, lx, xn, -ly, ly, yn);
	fprintf(matlab_file, "steps = [0:%d-1];\n\n", tn);

	// Generate the array for max |f|
    fprintf(matlab_file, "max = [");
	for(int i = 0; i < tn; i++)
		fprintf(matlab_file, "%0.10f ", max[i]);
	fprintf(matlab_file, "];\n\n");
	
	// generate initial pulse matrix
	fprintf(matlab_file, "psi_0 = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f = vec2mat(f,%d);\n\n", xn);
	
	// plot max |f| versus time step
	fprintf(matlab_file, "plot(steps, max, '-r', 'LineWidth', 1); grid on;\n"
						 "title('Maximum Value of |psi| per time step');\n"
						 "xlabel('Time Step'); ylabel('max |f|');\n\n");

	// generate initial pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,f_0);\n"
						 "title('Initial Pulse');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|psi|');\n\n");

	// generate final pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,f);\n"
						 "title('Final Pulse');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|psi|');\n\n");

	fclose(matlab_file);
	
}

/********************************************************************************
* Function Name: 	cm_plot_2d												*
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
void cm_plot_2d(complex double *psi_0, complex double *psi, double *max, 
				   double lx, double ly, int xn, int yn, int tn, char *filename)
{
    FILE *matlab_file = fopen(filename, "w");

	// Initialize Arrays
	fprintf(matlab_file, "[x,y] = meshgrid(linspace(%f,%f,%d), linspace(%f, %f, %d));\n", 
																-lx, lx, xn, -ly, ly, yn);
	fprintf(matlab_file, "steps = [0:%d-1];\n\n", tn);

	// Generate the array for max |f|
    fprintf(matlab_file, "max = [");
	for(int i = 0; i < tn; i++)
		fprintf(matlab_file, "%0.10f ", max[i]);
	fprintf(matlab_file, "];\n\n");
	
	// generate initial pulse matrix
	fprintf(matlab_file, "f_0 = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(psi_0[i]));
	fprintf(matlab_file, "];\n");                                                                 
    fprintf(matlab_file, "f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", cabs(psi[i]));
	fprintf(matlab_file, "];\n");                                                                 
    fprintf(matlab_file, "f_f = vec2mat(f_f,%d);\n\n", xn);
	
	// plot max |f| versus time step
	fprintf(matlab_file, "plot(steps, max, '-r', 'LineWidth', 1); grid on;\n"
						 "title('Maximum Value of |f| per time step');\n"
						 "xlabel('Time Step'); ylabel('max |f|');\n\n");

	// generate initial pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,f_0);\n"
						 "title('Initial Pulse');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|f|');\n\n");

	// generate final pulse figure
	fprintf(matlab_file, "figure;\n"
						 "surf(x,y,f_f);\n"
						 "title('Final Pulse');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|f|');\n\n");

	fclose(matlab_file);
}

/********************************************************************************
* Function Name: 	vtk_3d														*
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
			    int xn, int yn, int zn, char *filename)
{
	FILE *fp;
	fp = fopen(filename, "w");
	
	fprintf(fp, "# vtk DataFile Version 3.0\n"
				"vtk output\n"
				"ASCII\n"
				"DATASET STRUCTURED_GRID\n"
				"DIMENSIONS %d %d %d\n"
				"POINTS %d double\n", xn, yn, zn, xn*yn*zn);
	
	for(int counter = 0, i = 0, j = 0, k = 0; counter < xn * yn * zn; counter++)
    {
		fprintf(fp, "%f %f %f\n", x[i], y[j], z[k]);

		if(j == yn-1 && i == xn-1){i = 0; j = 0; k++;}
		else if(i == xn-1){i = 0; j++;}
		else i++;
		
	}
	fprintf(fp, "\nPOINT_DATA %d\n", xn * yn * zn);
	fprintf(fp, "SCALARS |f| double\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int i = 0; i < xn * yn * zn; i++)
		fprintf(fp, "%f\n", f[i]);

   fclose(fp); 
}

/********************************************************************************
* Function Name: 	max_index													*
* Description:		This function takes in an array and returns the index of the*
*					Lagest element.												*
* Parameters:		--> arr: pointer to double (array)							*
* 					--> size: size of array										*
* Output:			int, index of largest element in input array				*
********************************************************************************/
int max_index(double *arr, int size)
{
	int largest_index = 0;

	for (int index = largest_index; index < size; index++) 
		if (arr[largest_index] <= arr[index])
            largest_index = index;

    return largest_index;
}

/********************************************************************************
* Function Name: 	cmax_psi													*
* Description:		This takes in a complex function's real and imaginary parts,*
*					find the magnitude at each point and save the maximum 		*
*					amplitude in an array for this time step.					*
* Parameters:		--> R: array for real part of function.						*
* 					--> I: array for Imaginary part of function.				*
*					--> max: used to save max value of |psi|.					*
*					--> step: the current time step.							*
*					--> size: the size of R and I.								*
********************************************************************************/
void cmax_psi(complex double *psi, double *max, int step, int size)
{
	double *arr = (double*)malloc(sizeof(double) * size);
    
	for(int i = 0; i < size; i++)
		arr[i] = cabs(psi[i]);

    int index = max_index(arr, size);

	max[step] = arr[index];

	free(arr);
}

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
void max_psi(double *Re, double *Im, double *max, int step, int size)
{
	double *arr = (double*)malloc(sizeof(double) * size);
    
	for(int i = 0; i < size; i++)
		arr[i] = sqrt(Re[i] * Re[i] + Im[i] * Im[i]);

    int index = max_index(arr, size);

	max[step] = arr[index];

	free(arr);
}

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
/*
void cu_max_psi(double *d_Re, double *d_Im, double *max, int step, int size)
{
	double *h_R	= (double*)malloc(sizeof(double) * size);
    double *h_I	= (double*)malloc(sizeof(double) * size);   
	double *h_A	= (double*)malloc(sizeof(double) * size);
    
	cudaMemcpy(h_Re, d_Re, sizeof(double) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Im, d_Im, sizeof(double) * size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < size; i++)
		h_A[i] = sqrt(h_Re[i] * h_Re[i] + h_Im[i] * h_Im[i]);

    int index = max_index(h_A, size);

	max[step] = h_A[index];

    free(h_Re);
	free(h_Im);
	free(h_A);
}
*/
