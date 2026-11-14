#include<stdio.h>
#include<omp.h>
#include<stdlib.h>

int main(){
    int m,n,p,i,j,k;

    printf("Enter sizes (m n p) for mxn and nxp matrices: ");
    scanf("%d %d %d", &m, &n, &p);
    
    int **A=malloc(m*sizeof(int*));
    for(i=0; i<m; i++) A[i]=malloc(n*sizeof(int));

    int **B=malloc(n*sizeof(int*));
    for(i=0; i<n; i++) B[i]=malloc(p*sizeof(int));

    int **C=malloc(m*sizeof(int*));
    for(i=0; i<m; i++) C[i]=malloc(p*sizeof(int));

    #pragma omp parallel private(i,j,k) shared(A,B,C)
    {
        #pragma omp for schedule(static)
        for(i=0; i<m; i++){
            for(j=0; j<p; j++){
                C[i][j]=0;
                for(k=0; k<n; k++){
                    C[i][j]+=A[i][k]*B[k][j];
                }
            }
        }
        
    }
    printf("Matrix Multiplication complete.\n");
    return 0;
}