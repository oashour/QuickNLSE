// nlse (1+1)D     
#include "../lib/helpers.h"
#include<mpi.h>

#define ROOT 0
#define TAG1 1
#define TAG2 2

// given stuff
#define XN	256
#define YN 	256
#define TN	10000
#define LX  50.0
#define LY	50.0
#define TT	10.0

// calculated from given
#define DX	(2*LX / XN)
#define DY	(2*LY / YN)
#define DT	(TT / TN)

// Gaussian Parameters                      
#define  A_S 	(3.0/sqrt(8.0))
#define  R_S 	(sqrt(32.0/9.0))
#define  A 		0.6
#define  R	 	(1.0/(A*sqrt(1.0-A*A)))

// Index linearization for kernels [x,y] = [x * width + y] 
#define ind(i,j)  ((i)*XN+(j))		//[i  ,j  ] 

void Re_lin(double *Re, double *Im, double dt, int xn_loc);
void Im_lin(double *Re, double *Im, double dt, int xn_loc);
void nonlin(double *Re, double *Im, double dt, int xn_loc);
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, 
									double *Re_up, double *Re_dn, MPI_Status *status);
void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, 
									double *Re_up, double *Re_dn, MPI_Status *status);
void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, 
									double *Re_up, double *Re_dn, MPI_Status *status);

int main(int argc, char** argv)
{

	int rank, p;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	// allocate and initialize the arrays
    if(rank == ROOT)
		printf("DX: %f, DT: %f, dt/dx^2: %f\n", DX, DT, DT/(DX*DX));

	int xn_loc = XN/p;
	double *Re_0, *Im_0, *Re_new, *Im_new;
	// allocate space for local arrays and row above and row below
   	double *Re = (double*)malloc(sizeof(double) * (xn_loc+2)*YN);
   	double *Im = (double*)malloc(sizeof(double) * (xn_loc+2)*YN);
    
	// initialize the extra rows of local arrays to 0.
	// The up row is basically row 0 of each array
	for(int j = 0; j < YN; j++)
	{
		Re[ind(0,j)] = 0; Im[ind(0,j)] = 0;
	}
	// the down row is basically row (xn_loc+2) - 1 of each array
	for(int j = 0; j < YN; j++)
	{
		Re[ind(xn_loc+1,j)] = 0; Im[ind(xn_loc+1,j)] = 0;
	}
	
	// allocate arrays to copy the up and down rows for local arrays
   	double *Re_up = (double*)malloc(sizeof(double) * YN);
   	//double *Im_up = (double*)malloc(sizeof(double) * YN);
   	double *Re_dn = (double*)malloc(sizeof(double) * YN);
   	//double *Im_dn = (double*)malloc(sizeof(double) * YN);

	if (rank == ROOT)
	{
		double *x = (double*)malloc(sizeof(double) * XN);
    	double *y = (double*)malloc(sizeof(double) * YN);
		Re_0 = (double*)malloc(sizeof(double) * XN*YN);
    	Im_0 = (double*)malloc(sizeof(double) * XN*YN);

		// initialize x and y.
		for(int i = 0; i < XN; i++)
			x[i] = (i-XN/2)*DX;
			
		for(int i = 0; i < YN ; i++)
			y[i] = (i-YN/2)*DY;

		// Initial Conditions
		for(int i = 0; i < XN; i++)
			for(int j = 0; j < YN; j++)
				{
					Re_0[ind(i,j)] = A_S*A*exp(-(x[i]*x[i]+y[j]*y[j])
											/(2*R*R*R_S*R_S)); 
					Im_0[ind(i,j)] = 0;
				}
		
		// save first row into up array.
		for(int j = 0; j < YN; j++)
		{
			// save first row into up array
			Re_up[j] = Re_0[ind(0,j)];
			//Im_up[j] = Im_0[ind(0,j)];
			// save last row into down array.
			Re_dn[j] = Re_0[ind(XN-1,j)];
			//Im_dn[j] = Im_0[ind(0,j)];
		}

		free(x);
	}
	// send the up and down arrays to all processors
	// possibly improve to send up to root and down to p-1, only increases overhead
	MPI_Bcast(&Re_up[0], YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&Re_dn[0], YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	// Scatters real and imaginary parts starting from first element (0,0).
	// The first element of this  portion is saved into the second row 
	// 1st column of each local array. This makes its index (1,0).
	
	MPI_Scatter(&Re_0[ind(0,0)], xn_loc*YN, MPI_DOUBLE, &Re[ind(1,0)], xn_loc*YN,
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(&Im_0[ind(0,0)], xn_loc*YN, MPI_DOUBLE, &Im[ind(1,0)], xn_loc*YN, 
													MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	syncImRe(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
	MPI_Barrier(MPI_COMM_WORLD);
	double timei = MPI_Wtime();
	for (int i = 1; i < TN; i++)
	{
		// linear
		Re_lin(Re, Im, DT*0.5, xn_loc);
		syncRe(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc);
		syncIm(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
		// nonlinear
		nonlin(Re, Im, DT, xn_loc);
		syncImRe(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
		// linear
		Re_lin(Re, Im, DT*0.5, xn_loc);
		syncRe(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
		Im_lin(Re, Im, DT*0.5, xn_loc);
		syncIm(rank, p, xn_loc, Re, Im, Re_up, Re_dn, &status);
	}
	double timef = MPI_Wtime();
	double wtime = timef-timei;
	
    if (rank == ROOT)
	{
	   printf("%f\n", wtime);
	   Re_new = (double*)malloc(sizeof(double) * XN*YN);
	   Im_new = (double*)malloc(sizeof(double) * XN*YN);
	}

	// now we want to gatehr on root from all nodes. We want to gather from first row
	// to the row before the last one (neglecting the boundary rows and ghost rows).
	// This means we need to start the transfer from second row first column (1,0) 
	// And send it to (0,0) of the new perfectly square array. Note that all this code 
	// needs to be revised for the rectangular case, although it might look as if it works
	// to the outsider. It almost surely won't
    MPI_Gather(&Re[ind(1,0)], xn_loc*YN, MPI_DOUBLE, &Re_new[ind(0,0)], 
										xn_loc*YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Gather(&Im[ind(1,0)], xn_loc*YN, MPI_DOUBLE, &Im_new[ind(0,0)], 
										xn_loc*YN, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	
	if(rank == ROOT)
	{
		double *max = calloc(TN, sizeof(double));
		m_plot_2d(Re_0, Im_0, Re_new, Im_new, max, LX, LY, XN, YN, TN, "mpi_fdtd.m");

		free(Re_0); free(Im_0); free(Re_new); free(Im_new); free(max);
	}

	free(Re); 
	free(Im); 
	
    MPI_Finalize();

	return 0;
}

void Re_lin(double *Re, double *Im, double dt, int xn_loc)
{                  
	for(int i = 1; i < xn_loc+1; i++) // first and last rows avoided (ghost) 
		for(int j = 1; j < YN-1; j++)
 			   Re[ind(i,j)] = Re[ind(i,j)] 
							- dt/(DX*DX)*(Im[ind(i+1,j)] - 2*Im[ind(i,j)] + Im[ind(i-1,j)])
							- dt/(DY*DY)*(Im[ind(i,j+1)] - 2*Im[ind(i,j)] + Im[ind(i,j-1)]);
}

void Im_lin(double *Re, double *Im, double dt, int xn_loc)
{                  
	for(int i = 1; i < xn_loc+1; i++) // first and last rows avoided (ghost) 
		for(int j = 1; j < YN-1; j++) // first and last columns avoided (BC)
			Im[ind(i,j)] = Im[ind(i,j)] 
							+ dt/(DX*DX)*(Re[ind(i+1,j)] - 2*Re[ind(i,j)] + Re[ind(i-1,j)])
							+ dt/(DY*DY)*(Re[ind(i,j+1)] - 2*Re[ind(i,j)] + Re[ind(i,j-1)]);
}

void nonlin(double *Re, double *Im, double dt, int xn_loc)
{                  
	for(int i = 1; i < xn_loc+1; i++) // first and last rows avoided (ghost)
		for(int j = 1; j < YN-1; j++) // first and last columns avoided (BC)
		{
			double Rp = Re[ind(i,j)]; double Ip = Im[ind(i,j)];
			double A2 = Rp*Rp+Ip*Ip; // |psi|^2
			
			Re[ind(i,j)] =	Rp*cos((A2-A2*A2)*dt) - Ip*sin((A2-A2*A2)*dt);
			Im[ind(i,j)] =	Rp*sin((A2-A2*A2)*dt) + Ip*cos((A2-A2*A2)*dt);
		}
}

void syncIm(int rank, int p, int xn_loc, double *Re, double *Im, 
								double *Re_up, double *Re_dn, MPI_Status *status)
{
	if(rank == 0)
	{
		// first we set the upper ghost row to 0
		// We also set the first row after ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			Im[ind(0,j)]=0;// set ghost to 0
			Im[ind(1,j)] = 0; // set first non-ghost to IC
		}
		
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0)], YN, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0)], YN, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0)], YN, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0)], YN, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		
		// then we set the lower ghost row to 0
		// We also set the last row before ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			// set last non-ghost to IC
			Im[ind(xn_loc,j)] = 0; 
			Im[ind(xn_loc+1,j)] = 0; // set ghost to 0
		}
	}
	else
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Im[ind(0,0)], YN, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Im[ind(1,0)], YN, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Im[ind(xn_loc,0)], YN, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Im[ind(xn_loc+1,0)], YN, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncRe(int rank, int p, int xn_loc, double *Re, double *Im, 
								double *Re_up, double *Re_dn, MPI_Status *status)
{
	if(rank == 0)
	{
		// first we set the upper ghost row to 0
		// We also set the first row after ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			Re[ind(0,j)] = 0; ;// set ghost to 0
			Re[ind(1,j)] = Re_up[j]; // set first non-ghost to IC
		}
		
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], YN, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], YN, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], YN, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], YN, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		
		// then we set the lower ghost row to 0
		// We also set the last row before ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			// set last non-ghost to IC
			Re[ind(xn_loc,j)] = Re_dn[j];  
			Re[ind(xn_loc+1,j)] = 0; // set ghost to 0
		}
	}
	else
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], YN, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], YN, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], YN, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], YN, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}
void syncImRe(int rank, int p, int xn_loc, double *Re, double *Im, 
									double *Re_up, double *Re_dn, MPI_Status *status)
{
	if(rank == 0)
	{
		// first we set the upper ghost row to 0
		// We also set the first row after ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			Re[ind(0,j)] = 0; Im[ind(0,j)]=0;// set ghost to 0
			Re[ind(1,j)] = Re_up[j]; Im[ind(1,j)] = 0; // set first non-ghost to IC
		}
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], YN, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0)], YN, MPI_DOUBLE, 1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], YN, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0)], YN, MPI_DOUBLE, 1, TAG2, MPI_COMM_WORLD, status);
	}
	else if(rank == p-1)
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], YN, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0)], YN, MPI_DOUBLE, p-2, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], YN, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0)], YN, MPI_DOUBLE, p-2, TAG2, MPI_COMM_WORLD);
		
		// then we set the lower ghost row to 0
		// We also set the last row before ghost back to initial conditions
		for(int j = 0; j < YN; j++)
		{
			// set last non-ghost to IC
			Re[ind(xn_loc,j)] = Re_dn[j]; Im[ind(xn_loc,j)] = 0; 
			Re[ind(xn_loc+1,j)] = 0; Im[ind(xn_loc+1,j)] = 0; // set ghost to 0
		}
	}
	else
	{
		// Reeceive from left (TAG1)
		// Receive upper ghost row (i.e. first row)
		MPI_Recv(&Re[ind(0,0)], YN, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(0,0)], YN, MPI_DOUBLE, rank-1, TAG1, MPI_COMM_WORLD, status);
		// Send left (TAG2)
		// Send first non-ghost to left
		MPI_Send(&Re[ind(1,0)], YN, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(1,0)], YN, MPI_DOUBLE, rank-1, TAG2, MPI_COMM_WORLD);
		// Send right (TAG1)
		// Send last non-ghost to right
		MPI_Send(&Re[ind(xn_loc,0)], YN, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		MPI_Send(&Im[ind(xn_loc,0)], YN, MPI_DOUBLE, rank+1, TAG1, MPI_COMM_WORLD);
		// Reeceive from right (TAG2)
		// Receive bottom ghost row (i.e. last row)
		MPI_Recv(&Re[ind(xn_loc+1,0)], YN, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
		MPI_Recv(&Im[ind(xn_loc+1,0)], YN, MPI_DOUBLE, rank+1, TAG2, MPI_COMM_WORLD, status);
	}
}

