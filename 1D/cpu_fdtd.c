// nlse (1+1)D
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<sys/time.h>
#include<stddef.h>

// given stuff
#define XNODES	1000
#define TNODES	1000000
#define L		10.0
#define TMAX	1.0

// calculated from given
#define DELTAX	(L / (XNODES - 1.0))
#define DELTAT	(TMAX / (TNODES - 1.0))

double *linspace(double start, double end, int number);
void R_lin(double *R, double *I, double dt);
void I_lin(double *R, double *I, double dt);
void nonlin(double *R, double *I, double dt);
void matlab_plot(double *R_0, double *I_0, double *R, double *I);

int main(void)
{
    printf("DELTAX: %f, DELTAT: %f, dt/dx^2: %f\n", DELTAX, DELTAT, DELTAT/(DELTAX*DELTAX));

    double t1,t2,elapsed;
	struct timeval tp;
	int rtn;

	// allocate and initialize the arrays
    double *x = (double*)malloc(sizeof(double) * XNODES);
	double *R = (double*)malloc(sizeof(double) * XNODES);
    double *I = (double*)malloc(sizeof(double) * XNODES);
	double *R_0 = (double*)malloc(sizeof(double) * XNODES);
    double *I_0 = (double*)malloc(sizeof(double) * XNODES);
	
	for (int i = 0; i < XNODES; i++)
	{
		x[i] = (i-XNODES/2)*DELTAX;
		R[i] = sqrt(2.0)/(cosh(x[i]));  
		I[i] = 0;
		R_0[i] = R[i];
		I_0[i] = I[i];
	}
    
	rtn=gettimeofday(&tp, NULL);
	t1=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	for (int i = 1; i < TNODES; i++)
	{
		// linear
		R_lin(R, I, DELTAT*0.5);
        I_lin(R, I, DELTAT*0.5);
		// nonlinear
		nonlin(R, I, DELTAT);
		// linear
		R_lin(R, I, DELTAT*0.5);
        I_lin(R, I, DELTAT*0.5);
		// function to print some data to file
		//if(i == 10000-1 || i == 50000-1 || i == 100000-1 || i == 200000-1 || i == 500000-1 || i == 1000000-1)
			//checker(x, R, I, i, fp2, t, fp3);
	}
	rtn=gettimeofday(&tp, NULL);
	t2=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	elapsed=t2-t1;
	
	FILE *time_file;
	time_file = fopen("cpu_time.txt", "a"); 
	fprintf(time_file, "%f, ", elapsed);
	fclose(time_file);


	matlab_plot(R_0, I_0, R, I);

	free(R); 
	free(I); 
	free(R_0); 
	free(I_0); 
	free(x); 

	return 0;
}

void R_lin(double *R, double *I, double dt)
{                  
	for(int i = 1; i < XNODES-1; i++)
		R[i] = R[i] - dt/(DELTAX*DELTAX)*(I[i+1] - 2*I[i] + I[i-1]);
}

void I_lin(double *R, double *I, double dt)
{                  
	for(int i = 1; i < XNODES-1; i++)
		I[i] = I[i] + dt/(DELTAX*DELTAX)*(R[i+1] - 2*R[i] + R[i-1]);
}

void nonlin(double *R, double *I, double dt)
{                  
	for(int i = 0; i < XNODES; i++)
	{
		double Rp = R[i]; 
		double Ip = I[i];
		double A2 = Rp*Rp+Ip*Ip;
	
		R[i] =	Rp*cos(A2*dt) - Ip*sin(A2*dt);
   		I[i] =	Rp*sin(A2*dt) + Ip*cos(A2*dt);
	}
}

void matlab_plot(double *R_0, double *I_0, double *R, double *I)
{
	FILE *matlab_file;
	matlab_file = fopen("plot_CPU.m", "w");

	fprintf(matlab_file, "x = linspace(%f, %f, %d); \n", -L, L, XNODES);                                                                 

	fprintf(matlab_file, "psi_0 = [");

	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(R_0[i] * R_0[i] + I_0[i] * I_0[i]));
	fprintf(matlab_file,"];\n");                                                                 

	fprintf(matlab_file, "psi_f = [");
	for(int i = 0; i < XNODES; i++)	
		fprintf(matlab_file, "%0.10f ", sqrt(R[i] * R[i] + I[i] * I[i]));
	fprintf(matlab_file,"];\n");                                                                 
	
	fprintf(matlab_file, "plot(x, psi_0, '-r', 'LineWidth', 1); grid on;\n"
						 "hold on\n"
						 "plot(x, psi_f, '--b', 'LineWidth', 1);\n"
						 "legend('t = 0', 't = %f', 0);\n"
						 "title('Soliton Solution for CPU');\n"
						 "xlabel('x values'); ylabel('|psi|');", TMAX);
	fclose(matlab_file);
}

