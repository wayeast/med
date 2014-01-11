#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include "alert_sql.h"
#include "cuda_f.h"
#include "dd.h"
#include "od.h"
#include "ddlex.h"
#include "odlex.h"
#include "med_config.h"
#include "medconflex.h"


void prep_dd();
void feed_od(int *, int *, int *, int *);

int main() {
    medconf_configure();
    dd_init();

    // timing stuff
    //clock_t start, preptime, end;
    //double time_elapsed;
    //printf("Starting whole shebang...\n");
    //start = clock();

    // populate dev_dat *dd
    prep_dd();
    fprintf(stderr, "In main, following prep_dd(), reporting %d targets.\n", dd_getntargs());
    //preptime = clock();
    //time_elapsed = ( (double) (preptime - start) ) / CLOCKS_PER_SEC ;
    //printf("Finished dd_prep().  Time elapsed = %.4f\n", time_elapsed);


    /****************** Load Persistent Data to Device ***************/
    // device targets array
    int targ_bytes;
    targ_bytes = dd_getstargs() * sizeof(int);
    int *d_targs;
    call_cudaMalloc(&d_targs, targ_bytes);
    call_cudaMemcpy(d_targs, dd_gettargs(), targ_bytes, CUDA_H2D);
    // host and device edits space
    int edit_bytes;
    edit_bytes = dd_getntargs() * sizeof(int);
    int *h_edits;
    h_edits = (int *) malloc(edit_bytes);
    int *d_edits;
    call_cudaMalloc(&d_edits, edit_bytes);
    // device tbi
    int tbi_bytes;
    tbi_bytes = (dd_getntargs() + 1) * sizeof(int);
    int *d_tbi;
    call_cudaMalloc(&d_tbi, tbi_bytes);
    call_cudaMemcpy(d_tbi, dd_gettbi(), tbi_bytes, CUDA_H2D);
    /******* finished copying persistent data to device ***************/

    feed_od(d_edits, d_targs, d_tbi, h_edits);

    //end = clock();
    //time_elapsed = ( (double) (end - start) ) / CLOCKS_PER_SEC ;
    //printf(" finished whole shebang.  Time elapsed = %.4f\n", time_elapsed);


    medconf_cleanup();
    call_cudaFree(d_targs);
    call_cudaFree(d_edits);
    call_cudaFree(d_tbi);
    dd_freetable();
    dd_break();
    free(h_edits);

    return 1;
}

void prep_dd() {
    /*
     * Label: pop_tab
     * I would have liked to split the following segment off into a separate
     * file to make the main program nice and modular.  Alas, it is bound
     * (seemingly to me) inextricably to the lexer.  The lexer must declare
     * an extern variable *tab to send its output; this program must reset
     * yyin for the lexer.  I can't think of a way to pass *tab as an argument
     * to the lexer, or have whatever other file reset yyin, without bungling
     * modularity.
     */
    int sqlpipe[2];       // pipe object for feeding sql query results to lexer
    pipe(sqlpipe);    // create pipe for sql results -> lexer

    pid_t childpid;   // fork processes for pipe
    if ( (childpid = fork()) == -1 ) {
        perror("fork");
        exit(1);
    }

    if (childpid == 0) {
        /* Child process responsible for writing sql result strings to pipe */
        close(sqlpipe[0]);    // close input side of pipe
        db_setConx();
        db_feed(sqlpipe[1]);
        db_closeConx();
        close(sqlpipe[1]);
        exit(0);
    }

    else {
        /* Parent process responsible for lexing strings fed from sql result */
        close(sqlpipe[1]);     // close output side of pipe
        ddprepin = fdopen(sqlpipe[0], "r");
        ddpreplex();
        close(sqlpipe[0]);
    }
}

void feed_od(int *d_edits, int *d_targs, int *d_tbi, int *h_edits) {
    int sqlpipe[2];       // pipe object for feeding sql query results to lexer
    pipe(sqlpipe);    // create pipe for sql results -> lexer

    pid_t childpid;   // fork processes for pipe
    if ( (childpid = fork()) == -1 ) {
        perror("fork");
        exit(1);
    }

    if (childpid == 0) {
        /* Child process responsible for writing sql result strings to pipe */
        close(sqlpipe[0]);    // close input side of pipe
        db_setConx();
        db_feed(sqlpipe[1]);
        db_closeConx();
        close(sqlpipe[1]);
        exit(1);
    }

    else {
        /* Parent process responsible for lexing strings fed from sql result */

        // some timing stuff
        //clock_t start, end;
        //double time_elapsed;

        close(sqlpipe[1]);     // close output side of pipe
        odfeedin = fdopen(sqlpipe[0], "r");
        db_setWritex();
        od_init();
        int status;
        status = ODSTAT_TOK;
        while (status != EOF) {
            status = odfeedlex();
            if (status == ODSTAT_ID) {
                //printf("Calling med_kernel on %s...", od_getid());
                //start = clock();

                //int a;
                //for (a=0; a<od_getlen(); a++) printf("%d ", (od_getarray())[a]);
                //printf("\n");
                int origlen;
                origlen = od_getlen() * sizeof(int);
                int *d_orig;
                call_cudaMalloc(&d_orig, origlen);
                call_cudaMemcpy(d_orig, od_getarray(), origlen, CUDA_H2D);
                int *d_dist;
                call_cudaMalloc(&d_dist, (od_getlen() + 1) * dd_getntargs() * sizeof(int) * 2);
                call_med_kernel(
                        d_orig,
                        od_getlen(),
                        d_edits,
                        d_targs,
                        d_dist,
                        d_tbi,
                        dd_getstargs(),
                        dd_getntargs() );
                call_cudaMemcpy(h_edits, d_edits, dd_getntargs() * sizeof(int), CUDA_D2H);
                /*
                printf("Edits(40):");
                for (a=0; a<40; a++) printf(" %d", h_edits[a]);
                printf("\n");
                */
                
                //end = clock();
                //time_elapsed = ( (double) (end - start) ) / CLOCKS_PER_SEC;
                //printf(" finished. Time elapsed = %.4f\n", time_elapsed);

                // write results to out db
                int e, ee, t, tt;
                tt = dd_getntargs();
                ee = medconf_getnoevals();
                for (t=0; t<tt; t++) {
                    for (e=0; e<ee; e++) {
                        if (h_edits[t] == e) db_write(e, od_getid(), dd_gettids()[t]);
                    }
                }
                
                call_cudaFree(d_orig);
                call_cudaFree(d_dist);
                od_free();
                od_init();
            }  // end status == ODSTAT_ID
        }  // end while
        od_free();
        db_closeWritex();
        close(sqlpipe[0]);
    }  // end parent process routine
}
