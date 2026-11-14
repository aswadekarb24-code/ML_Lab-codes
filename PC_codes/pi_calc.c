#include<stdio.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    int my_rank, num_procs, n_interval, i;
    double mypi=0.0, pi, h, sum=0.0, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank==0) {
        printf("Enter number of intervals: ");
        scanf("%d", &n_interval);
    }
    MPI_Bcast(&n_interval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    h=1.0/n_interval;
    for (i=my_rank+1; i<=n_interval; i+=num_procs) {
        x=h*(i-0.5);
        sum+=4.0/(1.0+x*x);
    }
    mypi=h*sum;
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (my_rank==0) printf("pi is approximately %.16f\n", pi);
    MPI_Finalize();
    return 0;
}