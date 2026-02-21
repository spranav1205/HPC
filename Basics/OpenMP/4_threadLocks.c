#include <stdio.h>
#include <omp.h>

typedef struct {
    int balance;
    omp_lock_t lock; // Each account has its own "lock object"
} Account;

int main() {
    Account accounts[2];

    // 1. Initialize locks
    for(int i = 0; i < 2; i++) {
        accounts[i].balance = 1000;
        omp_init_lock(&accounts[i].lock);
    }

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        
        // Everyone tries to withdraw 10 dollars from Account 0
        // 2. Set the lock
        omp_set_lock(&accounts[0].lock);
        
        // printf("Thread %d is updating Account 0...\n", tid);
        accounts[0].balance -= tid*10;

        printf("Thread %d updated Account 0 balance to %d\n", tid, accounts[0].balance);
        
        // 3. Unset the lock
        omp_unset_lock(&accounts[0].lock);
    }

    // 4. Clean up
    for(int i = 0; i < 2; i++) {
        omp_destroy_lock(&accounts[i].lock);
    }

    printf("Final Balance: %d\n", accounts[0].balance);
    return 0;
}