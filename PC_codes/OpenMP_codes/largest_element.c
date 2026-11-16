#include<stdio.h>
#include<omp.h>

int main(){
    int n, current_max=-1, i, arr[10000];

    printf("Enter number of Elements: ");
    scanf("%d", &n);

    printf("Enter the array input: ");
    for(i=0; i<n; i++) scanf("%d",&arr[i]);

    #pragma omp parallel for shared(current_max)
    for(i=0; i<n; i++){
        #pragma omp critical
        {
            if(arr[i]>current_max){
                current_max=arr[i];
            }
        }
    }
    printf("Maximum Element is: %d\n", current_max);
    return 0;
}