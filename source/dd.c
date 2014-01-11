#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dd.h"

#define VTAB_INIT 10000
#define ITARGS_INIT 10000
#define TIDS_INIT 5000
#define ITBI_INIT ( TIDS_INIT + 1 )


struct {
    char **v_tab;
    int    v_entries;
    int    v_cap;

    int   *i_targs;
    int   *i_tbi;
    char **tids;
    int    n_targs;
    int    s_targs;
    int    i_cap;
    int    t_cap;
} dd;
void tid_append(char *);
void itargs_append(int);
int  vtab_append(char *);

char **dd_getvtab()      { return dd.v_tab; }
int    dd_getventries()  { return dd.v_entries; }
int   *dd_gettargs()     { return dd.i_targs; }
int   *dd_gettbi()       { return dd.i_tbi; }
char **dd_gettids()      { return dd.tids; }
int    dd_getntargs()    { return dd.n_targs; }
int    dd_getstargs()    { return dd.s_targs; }


void dd_init() {
    /*
     * Initialize elements of struct dd.
     */
    if (! (dd.v_tab = (char **) malloc( VTAB_INIT * sizeof(char *) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for v_tab during initialization.\n");
        exit(0);
    }
    if (! (dd.i_targs = (int *) malloc( ITARGS_INIT * sizeof(int) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for i_targs during initialization.\n");
        free(dd.v_tab);
        exit(0);
    }
    if (! (dd.i_tbi = (int *) malloc( ITBI_INIT * sizeof(int) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for i_tbi during initialization.\n");
        free(dd.i_targs);
        free(dd.v_tab);
        exit(0);
    }
    if (! (dd.tids = (char **) malloc( TIDS_INIT * sizeof(char *) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for tids during initialization.\n");
        free(dd.i_tbi);
        free(dd.i_targs);
        free(dd.v_tab);
        exit(0);
    }

    dd.v_entries    = 0;
    dd.v_cap        = VTAB_INIT;
    dd.i_tbi[0]     = 0;
    dd.n_targs      = 0;
    dd.s_targs      = 0;
    dd.i_cap        = ITARGS_INIT;
    dd.t_cap        = TIDS_INIT;
}

void dd_freetable() {
    /*
     * Just free v_table.  I want to separate this function in order to free up
     * memory from this table before continuing to process the edit distances.
     */
    int i;
    for (i=0; i<dd.v_entries; i++) free(dd.v_tab[i]);
    free(dd.v_tab);
}

void dd_break() {
    /*
     * As in "break camp."  This will clean up the rest of allocated memory
     * after dd_freetable().
     */
    int i;
    for (i=0; i<dd.n_targs; i++) free(dd.tids[i]);
    free(dd.tids);
    free(dd.i_targs);
    free(dd.i_tbi);
}

int dd_lookup(char *cand) {
    /*
     * Return index of cand in v_table.
     * Return -1 if cand not in v_table.
     */
    int i;
    i = dd.v_entries - 1;
    while ( i >= 0 && strcmp(dd.v_tab[i], cand) ) --i;
    return i;
}

void dd_addentry(char *entry) {
    /*
     * Add entry to v_tab if not already present;
     * append entry's index in v_tab to i_targs.
     */
    int index;
    if ( (index = dd_lookup(entry)) < 0 )
        index = vtab_append(entry);
    itargs_append(index);
}

void dd_addtarg(char *tid) {
    /*
     * Add tid to tids and tbi to i_tbi.
     */
    tid_append(tid);
}

void tid_append(char *tid) {
    /*
     * Handles both adding new tid to i_tids and adding tbi to i_tbi
     */
    // check for room in dd.tids
    if (dd.n_targs == dd.t_cap) {
        //printf("Expanding tid/tbi\n");
        // expand tids
        char **new_tids;
        if (! (new_tids = (char **) malloc( (dd.t_cap + TIDS_INIT) * sizeof(char *) ))) {
            fprintf(stderr, "ERROR: unable to allocate memory for expanded tids.\n");
            dd_freetable();
            dd_break();
            exit(0);
        }
        int i;
        for (i=0; i<dd.n_targs; i++) {
            //printf("int tids_append, expanding tids, copying tids[%d] = '%s'\n", i, dd.tids[i]);
            if (! (new_tids[i] = (char *) malloc( (strlen(dd.tids[i]) + 1) * sizeof(char) ))) {
                fprintf(stderr, "ERROR: unable to allocate memory for string in new_tids.\n");
                dd_freetable();
                dd_break();
                exit(0);
            }
            strcpy(new_tids[i], dd.tids[i]);
            free(dd.tids[i]);
        }
        free(dd.tids);
        dd.tids = new_tids;
        // expand tbi
        int *new_tbi;
        if (! (new_tbi = (int *) malloc( (dd.t_cap + ITBI_INIT) * sizeof(int) ))) {
            fprintf(stderr, "ERROR: unable to allocate memory for expanded tbi.\n");
            dd_freetable();
            dd_break();
            exit(0);
        }
        for (i=0; i<=dd.n_targs; i++) new_tbi[i] = dd.i_tbi[i];
        free(dd.i_tbi);
        dd.i_tbi = new_tbi;

        dd.t_cap += TIDS_INIT;
    }
    // add new tid and increment n_targs
    //printf("in tids_append, inserting '%s'\n", tid);
    if (! (dd.tids[dd.n_targs] = (char *) malloc( (strlen(tid) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for new tid entry.\n");
        dd_freetable();
        dd_break();
        exit(0);
    }
    strcpy(dd.tids[dd.n_targs], tid);
    //printf("in tids_append, inserted  '%s'\n", dd.tids[dd.n_targs]);
    dd.n_targs++;
    dd.i_tbi[dd.n_targs] = dd.s_targs;
}

void itargs_append(int index){
    if (dd.s_targs == dd.i_cap) {
        //printf("expanding itargs\n");
        int *new_targs;
        if (! (new_targs = (int *) malloc( (dd.i_cap + ITARGS_INIT) * sizeof(int) ))) {
            fprintf(stderr, "ERROR: unable to allocate memory for expanded i_targs.\n");
            dd_freetable();
            dd_break();
            exit(0);
        }
        int i;
        for (i=0; i<dd.s_targs; i++) new_targs[i] = dd.i_targs[i];
        free(dd.i_targs);
        dd.i_targs = new_targs;
        dd.i_cap += ITARGS_INIT;
    }
    dd.i_targs[dd.s_targs] = index;
    dd.s_targs++;
}

int vtab_append(char *entry) {
    /*
     * Append entry to v_tab; return index of entry in v_tab.
     * This function should ONLY be called if dd_lookup(entry)
     * has already determined that entry is not in v_tab!
     */
    int index;
    index = dd.v_entries;
    // check for room in v_tab
    if (dd.v_entries == dd.v_cap) {
        //printf("expanding v_tab\n");
        char **new_tab;
        if (! (new_tab = (char **) malloc( (dd.v_cap + VTAB_INIT) * sizeof(char *) ))) {
            fprintf(stderr, "ERROR: unable to allocate memory for expanded v_tab.\n");
            dd_freetable();
            dd_break();
            exit(0);
        }
        int i;
        for (i=0; i<dd.v_entries; i++) {
            if (! (new_tab[i] = (char *) malloc( (strlen(dd.v_tab[i]) + 1) * sizeof(char) ))) {
                fprintf(stderr, "ERROR: unable to allocate memory for string in new table.\n");
                dd_freetable();
                dd_break();
                exit(0);
            }
            strcpy(new_tab[i], dd.v_tab[i]);
            free(dd.v_tab[i]);
        }
        free(dd.v_tab);
        dd.v_tab = new_tab;
        dd.v_cap += VTAB_INIT;
    }  // end expand to bigger v_tab
    // append entry
    if (! (dd.v_tab[dd.v_entries] = (char *) malloc( (strlen(entry) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for new v_tab entry %s\n", entry);
        dd_freetable();
        dd_break();
        exit(0);
    }
    strcpy(dd.v_tab[dd.v_entries], entry);
    dd.v_entries++;
    return index;
}
