 // Cubic Quintic Nonlinear Schrodinger Equation
#include "../lib/helpers.h"

// given parameters for FDTD                                              XN
#define XN	64			// number of X nodes                       _____________
#define YN	64			// number of Y nodes                    Y |_|_|_|_|_|_|_|H
#define TN	100			// number of temporal nodes             N |_|_|_|_|_|_|_|E
#define LX	50.0		// maximum X                            O |_|_|_|_|_|_|_|Im
#define LY	50.0		// maximum Y                            D |_|_|_|_|_|_|_|G
#define TT	10.0  		// maximum t                            E |_|_|_|_|_|_|_|H
//                                                              S |_|_|_|_|_|_|_|T
// Gaussian Parameters                                                 WIDTH
#define  A_S 	(3.0/sqrt(8.0))
#define  R_s 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R 	(1.0/(A*sqrt(1.0-A*A)))

// calculated from given
#define DX	(2*LX / XN)
#define DY	(2*LY / YN)
#define DT	(TT / TN)

// Imndex linearization for kernels [x,y] = [x * width + y] 
#define ind(i,j)  ((i)*XN+(j))		//[i  ,j  ] 

// Function prototypes 
void Re_lin(double *Re, double *Im, double dt);
void Im_lin(double *Re, double *Im, double dt);
void nonlin(double *Re, double *Im, double dt);

int main(void)
{
    printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));
	
    // Allocate x, y, Re and Im on host. Max will be use to store max |psi|
	// Re_0 and Im_0 will keep a copy of initial pulse for printing
	double *x = (double*)malloc(sizeof(double) * XN);
	double *y = (double*)malloc(sizeof(double) * YN);
	double *max = (double*)malloc(sizeof(double) * TN);
	double *Re = (double*)malloc(sizeof(double) * XN * YN);
    double *Im = (double*)malloc(sizeof(double) * XN * YN);   
	double *Re_0 = (double*)malloc(sizeof(double) * XN * YN);
    double *Im_0 = (double*)malloc(sizeof(double) * XN * YN);   
	
	// initialize x and y.
	for(int i = 0; i < XN ; i++)
		x[i] = (i-XN/2)*DX;
		
    for(int i = 0; i < YN ; i++)
		y[i] = (i-YN/2)*DY;

    // Initial Conditions
    for(int j = 0; j < YN; j++)
		for(int i = 0; i < XN; i++)
			{
				Re[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])/(2*R*R*R_s*R_s)); 
				Im[ind(i,j)] = 0;
				Re_0[ind(i,j)] = Re[ind(i,j)];
				Im_0[ind(i,j)] = Im[ind(i,j)];
			}
	
	// print max |psi| for initial conditions
	max_psi(Re, Im, max, 0, XN*YN);
	// Begin timing
	for (int i = 1; i < TN; i++)
	{
		// linear
		Re_lin(Re, Im, DT*0.5);
        Im_lin(Re, Im, DT*0.5);
		// nonlinear
		nonlin(Re, Im, DT);
		// linear
		Re_lin(Re, Im, DT*0.5);
        Im_lin(Re, Im, DT*0.5);
		// find max psi
		max_psi(Re, Im, max, i, XN*YN);
	}

	// Generate MATLAB file to plot max |psi| and the initial and final pulses
	m_plot_2d(Re_0, Im_0, Re, Im, max, LX, LY, XN, YN, TN, "cpu_new.m");
	// wrap up                                                  
	free(Re); 
	free(Im); 
	free(Re_0); 
	free(Im_0); 
	free(x); 
	free(y);
	free(max);
	
	return 0;
}

void Re_lin(double *Re, double *Im, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int j = 1; j < YN - 1; j++)
		for(int i = 1; i < XN - 1; i++)
			Re[ind(i,j)] = Re[ind(i,j)] 
						   - dt/(DX*DX)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
						   - dt/(DY*DY)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

void Im_lin(double *Re, double *Im, double dt)
{                  
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
    for(int j = 1; j < YN - 1; j++)
		for(int i = 1; i < XN - 1; i++)
			Im[ind(i,j)] = Im[ind(i,j)] 
						   + dt/(DX*DX)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
						   + dt/(DY*DY)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

void nonlin(double *Re, double *Im, double dt)
{                  
	double Rp, Ip, A2;
	// We're avoiding Boundary Elements (kept at initial value approx = 0)
	for(int j = 1; j < YN-1; j++)
		for(int i = 1; i < XN-1; i++)
		{
			Rp = Re[ind(i,j)];  Ip = Im[ind(i,j)];
			A2 = Rp*Rp+Ip*Ip; // |psi|^2
			
			Re[ind(i,j)] = Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
			Im[ind(i,j)] = Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
		}
}

