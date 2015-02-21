#include "cu_helpers.h"

/********************************************************************************
* Function Name: 	_cufftGetErrorEnum											*
* Description:		This functions decodes the error enumerations produced by   *
*					CUFFT library calls.                                        *
* Parameters:		--> error: the cufft error. This is to be used with a macro *
						and never directly. 									*
********************************************************************************/
const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

/********************************************************************************
* Function Name: 	_cudaGetErrorEnum											*
* Description:		This functions decodes the error enumerations produced by   *
*					CUDA Runtime API Calls                                      *
* Parameters:		--> error: the cuda error. This is to be used with a macro  *
*						and never directly. 									*
********************************************************************************/
const char *_cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";

        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";

        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";

        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";

        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";

        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";

        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";

        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";

        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}

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
void m_plot_1d(double *Re_0, double *Im_0, double *Re, double *Im, 
				   double l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "f_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f_f, '--b', 'LineWidth', 1);\n"
						 "legend('Initial', 'Final', 0);\n"
						 "title('Initial and Final Pulse');\n"
						 "xlabel('x values'); ylabel('|f|');\n");
	fclose(matlab_file);
}

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
void m_plot_1df(float *Re_0, float *Im_0, float *Re, float *Im, 
				   float l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "f_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f_f, '--b', 'LineWidth', 1);\n"
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
void cm_plot_1d(cuDoubleComplex *f_0, cuDoubleComplex *f, 
				    double l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "f_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabs(f_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabs(f[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f_f, '--b', 'LineWidth', 1);\n"
						 "legend('Initial', 'Final', 0);\n"
						 "title('Initial and Final Pulse');\n"
						 "xlabel('x values'); ylabel('|f|');");
	fclose(matlab_file);
}

/********************************************************************************
* Function Name: 	cm_plot_1df 		 											*
* Description:		This takes in a complex 1D function and plots both initial	*
*					And final pulse on the same graph.							*
* Parameters:		--> psi_0: complex array for initial pulse					*
* 					--> psi: complex array for final pulse						*
*					--> l: size of x-spatial domain								*
*					--> xn: number of x nodes									*
*					--> filename: name of file generated (including .m)			*
********************************************************************************/
void cm_plot_1df(cuComplex *f_0, cuComplex *f, 
				    float l, int xn, char *filename)
{
	FILE *matlab_file;
	matlab_file = fopen(filename, "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n\n", -l, l, xn);                                                                 

	fprintf(matlab_file, "f_0 = [");

	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabsf(f_0[i]));
	fprintf(matlab_file,"];\n\n");                                                                 

	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabsf(f[i]));
	fprintf(matlab_file,"];\n\n");                                                                 
	
	fprintf(matlab_file, "plot(x, f_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, f_f, '--b', 'LineWidth', 1);\n"
						 "legend('Initial', 'Final', 0);\n"
						 "title('Initial and Final Pulse');\n"
						 "xlabel('x values'); ylabel('|f|');");
	fclose(matlab_file);
}

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
				   float lx, float ly, int xn, int yn, int tn, char *filename)
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
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f_f = vec2mat(f_f,%d);\n\n", xn);
	
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
						 "surf(x,y,f_f);\n"
						 "title('Final Pulse');\n"
						 "xlabel('x'); ylabel('y'); zlabel('|psi|');\n\n");

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
	fprintf(matlab_file, "f_0 = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re_0[i] * Re_0[i] + Im_0[i] * Im_0[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(Re[i] * Re[i] + Im[i] * Im[i]));
	fprintf(matlab_file,"];\n");                                                                 
    fprintf(matlab_file,"f_f = vec2mat(f_f,%d);\n\n", xn);
	
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
						 "surf(x,y,f_f);\n"
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
void cm_plot_2d(cuDoubleComplex *psi_0, cuDoubleComplex *psi, double *max, 
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
		fprintf(matlab_file, "%0.10f ", cuCabs(psi_0[i]));
	fprintf(matlab_file, "];\n");                                                                 
    fprintf(matlab_file, "f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabs(psi[i]));
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
				   float lx, float ly, int xn, int yn, int tn, char *filename)
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
		fprintf(matlab_file, "%0.10f ", cuCabsf(psi_0[i]));
	fprintf(matlab_file, "];\n");                                                                 
    fprintf(matlab_file, "f_0 = vec2mat(f_0,%d);\n\n", xn);

	// Generate final pulse matrix
	fprintf(matlab_file, "f_f = [");
	for(int i = 0; i < xn*yn; i++)	
		fprintf(matlab_file, "%0.10f ", cuCabsf(psi[i]));
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
int max_indexf(float *arr, int size)
{
	int largest_index = 0;

	for (int index = largest_index; index < size; index++) 
		if (arr[largest_index] <= arr[index])
            largest_index = index;

    return largest_index;
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
* Function Name: 	max_psi													*
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
void max_psi(double *d_Re, double *d_Im, double *max, int step, int size)
{
	double *h_Re	= (double*)malloc(sizeof(double) * size);
    double *h_Im	= (double*)malloc(sizeof(double) * size);   
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
void max_psif(float *d_Re, float *d_Im, float *max, int step, int size)
{
	float *h_Re	= (float*)malloc(sizeof(float) * size);
    float *h_Im	= (float*)malloc(sizeof(float) * size);   
	float *h_A	= (float*)malloc(sizeof(float) * size);
    
	cudaMemcpy(h_Re, d_Re, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Im, d_Im, sizeof(float) * size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < size; i++)
		h_A[i] = sqrt(h_Re[i] * h_Re[i] + h_Im[i] * h_Im[i]);

    int index = max_indexf(h_A, size);

	max[step] = h_A[index];

    free(h_Re);
	free(h_Im);
	free(h_A);
}
