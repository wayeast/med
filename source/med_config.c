#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "med_config.h"
#include "medconflex.h"

/********************************************************
 *   Some local definitions
 *******************************************************/
#define CONFIG_FILE "config.rc"
#define ET_INIT_SZ 10
int *_growTab(int *old_array, int old_sz) {
    /* 
     * Free old_array and return new array with double capacity and first
     * old_sz elements same as old_array.
     */
    int *new_array;
    if (! (new_array = (int *) malloc( (2 * old_sz) * sizeof(int) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for new edit table.\n");
        exit(0);
    }
    int i;
    for (i=0; i<old_sz; i++) new_array[i] = old_array[i];
    free(old_array);
    return new_array;
}

struct {
    char *in_dbname;
    char *in_hostaddr;
    char *in_port;
    char *in_user;
    char *in_connect_timeout;
    char *in_password;
    char *in_table;
    char *input_col;
    char *input_separator;

    char *out_dbname;
    char *out_hostaddr;
    char *out_port;
    char *out_user;
    char *out_connect_timeout;
    char *out_password;
    char *out_table;
    char *out_origcol;
    char *out_destcol;

    int   clobber;
    int  *edit_tab;
    int   no_edits;
 
} medconfig;

/*********************************************************
 *   Mutator functions to set string members of medconfig
 *******************************************************/

void  medconf_setindbname(const char *dbname) {
    if (! (medconfig.in_dbname = (char *) malloc( (strlen(dbname) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_dbname.\n");
        exit(0);
    }
    strcpy(medconfig.in_dbname, dbname);
}
void  medconf_setinhostaddr(const char *addr) {
    if (! (medconfig.in_hostaddr = (char *) malloc( (strlen(addr) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_hostaddr.\n");
        exit(0);
    }
    strcpy(medconfig.in_hostaddr, addr);
}
void  medconf_setinport(const char *port) {
    if (! (medconfig.in_port = (char *) malloc( (strlen(port) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_port.\n");
        exit(0);
    }
    strcpy(medconfig.in_port, port);
}
void  medconf_setinuser(const char *user) {
    if (! (medconfig.in_user = (char *) malloc( (strlen(user) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_user.\n");
        exit(0);
    }
    strcpy(medconfig.in_user, user);
}
void  medconf_setinconnecttimeout(const char *cto) {
    if (! (medconfig.in_connect_timeout = (char *) malloc( (strlen(cto) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_connect_timeout.\n");
        exit(0);
    }
    strcpy(medconfig.in_connect_timeout, cto);
}
void  medconf_setinpassword(const char *pwd) {
    if (! (medconfig.in_password = (char *) malloc( (strlen(pwd) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_password.\n");
        exit(0);
    }
    strcpy(medconfig.in_password, pwd);
}
void  medconf_setintable(const char *table) {
    if (! (medconfig.in_table = (char *) malloc( (strlen(table) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for in_table.\n");
        exit(0);
    }
    strcpy(medconfig.in_table, table);
}
void  medconf_setinputcol(const char *col) {
    if (! (medconfig.input_col = (char *) malloc( (strlen(col) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for input_col.\n");
        exit(0);
    }
    strcpy(medconfig.input_col, col);
}
void  medconf_setinputseparator(const char *sep) {
    if (! (medconfig.input_separator = (char *) malloc( (strlen(sep) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for input_separator.\n");
        exit(0);
    }
    strcpy(medconfig.input_separator, sep);
}
/************************************************************************/
void  medconf_setoutdbname(const char *dbname) {
    if (! (medconfig.out_dbname = (char *) malloc( (strlen(dbname) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_dbname.\n");
        exit(0);
    }
    strcpy(medconfig.out_dbname, dbname);
}
void  medconf_setouthostaddr(const char *addr) {
    if (! (medconfig.out_hostaddr = (char *) malloc( (strlen(addr) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_hostaddr.\n");
        exit(0);
    }
    strcpy(medconfig.out_hostaddr, addr);
}
void  medconf_setoutport(const char *port) {
    if (! (medconfig.out_port = (char *) malloc( (strlen(port) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_port.\n");
        exit(0);
    }
    strcpy(medconfig.out_port, port);
}
void  medconf_setoutuser(const char *user) {
    if (! (medconfig.out_user = (char *) malloc( (strlen(user) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_user.\n");
        exit(0);
    }
    strcpy(medconfig.out_user, user);
}
void  medconf_setoutconnecttimeout(const char *cto) {
    if (! (medconfig.out_connect_timeout = (char *) malloc( (strlen(cto) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_connect_timeout.\n");
        exit(0);
    }
    strcpy(medconfig.out_connect_timeout, cto);
}
void  medconf_setoutpassword(const char *pwd) {
    if (! (medconfig.out_password = (char *) malloc( (strlen(pwd) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_password.\n");
        exit(0);
    }
    strcpy(medconfig.out_password, pwd);
}
void  medconf_setouttable(const char *table) {
    if (! (medconfig.out_table = (char *) malloc( (strlen(table) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_table.\n");
        exit(0);
    }
    strcpy(medconfig.out_table, table);
}
void  medconf_setoutorigcol(const char *col) {
    if (! (medconfig.out_origcol = (char *) malloc( (strlen(col) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_origcol.\n");
        exit(0);
    }
    strcpy(medconfig.out_origcol, col);
}
void  medconf_setoutdestcol(const char *col) {
    if (! (medconfig.out_destcol = (char *) malloc( (strlen(col) + 1) * sizeof(char) ))) {
        fprintf(stderr, "ERROR: unable to allocate memory for out_destcol.\n");
        exit(0);
    }
    strcpy(medconfig.out_destcol, col);
}
void medconf_setoutclobber(const char *c) {
    medconfig.clobber = ( strcmp(c, "True") ? 0 : 1 );
}

/****************************************************************
 *   Mutator functions for edit table and no_edits member
 ***************************************************************/

void  medconf_setzerorangeevals(const char *val) {
    /* Fill edit table given a single, maximal edit distance value */
    int hi_bound = atoi(val);
    if (! (medconfig.edit_tab = (int *) malloc( (hi_bound + 1) * sizeof(int) )) ) {
        fprintf(stderr, "ERROR: unable to allocate memory for medconfig.edit_tab.\n");
        exit(0);
    }
    int i;
    for (i=0; i<=hi_bound; i++) medconfig.edit_tab[i] = i;
    medconfig.no_edits = hi_bound + 1;
}

void  medconf_setlinearrangeevals(const char *vals) {
    /* Fill edit table given a range of edit values of form 'x-y' */
    int lo_bound, hi_bound;

    lo_bound = 0;
    // set lo and hi bounds
    while (*vals != '-') {
        if (*vals >= '0' && *vals <= '9') {
            lo_bound *= 10;
            lo_bound += (int ) (*vals - '0');
            vals++;
        }
        else {
            fprintf(stderr, "ERROR: unrecognized input.  Check edit_range configuration parameter.\n");
            exit(0);
        }
    }
    hi_bound = atoi(++vals);

    if (! (hi_bound > lo_bound)) {
        fprintf(stderr, "ERROR: illegal range.  Check edit_range configuration parameter.\n");
        exit(0);
    }

    int range = (hi_bound - lo_bound) + 1;
    if (! (medconfig.edit_tab = (int *) malloc( range * sizeof(int) )) ) {
        fprintf(stderr, "ERROR: unable to allocate memory for medconfig.edit_tab.\n");
        exit(0);
    }
    int i;
    for (i=0; i<range; i++) medconfig.edit_tab[i] = lo_bound + i;
    medconfig.no_edits = range;
}

void  medconf_setdiscreteevals(const char *vals) {
    /* 
     * Fill edit table given series of discrete edit values separated by commas
     * and no spaces: eg. 'x,y,z'
     */
    int i, count, nextInt, localmax;
    i = count = nextInt = 0;
    if (! (medconfig.edit_tab = (int *) malloc( ET_INIT_SZ * sizeof(int) )) ) {
        fprintf(stderr, "ERROR: unable to allocate memory for medconfig.edit_tab.\n");
        exit(0);
    }
    localmax = ET_INIT_SZ;
    while ( i<strlen(vals) ) {
        if (vals[i] == ',') {
            if (count == localmax) {
                medconfig.edit_tab = _growTab(medconfig.edit_tab, localmax);
                localmax *= 2;
            }
            medconfig.edit_tab[count++] = nextInt;
            nextInt = 0;
            i++;
        }
        if (vals[i] >= '0' && vals[i] <= '9') {
            nextInt *= 10;
            nextInt += (int) (vals[i] - '0');
        }
        else {
            fprintf(stderr, "ERROR: unrecognized input.  Check edit_range configuration parameter.\n");
            exit(0);
        }
        i++;
    }
    medconfig.edit_tab[count++] = nextInt;
    medconfig.no_edits = count;
}

/********************************************************************
 *   Accessor functions for all medconfig members
 *******************************************************************/

char *medconf_getindbname(void) { return medconfig.in_dbname; }
char *medconf_getinhostaddr(void) { return medconfig.in_hostaddr; }
char *medconf_getinport(void) { return medconfig.in_port; }
char *medconf_getinuser(void) { return medconfig.in_user; }
char *medconf_getinconnecttimeout(void) { return medconfig.in_connect_timeout; }
char *medconf_getinpassword(void) { return medconfig.in_password; }
char *medconf_getintable(void) { return medconfig.in_table; }
char *medconf_getinputcol(void) { return medconfig.input_col; }
char *medconf_getinputseparator(void) { return medconfig.input_separator; }

char *medconf_getoutdbname(void) { return medconfig.out_dbname; }
char *medconf_getouthostaddr(void) { return medconfig.out_hostaddr; }
char *medconf_getoutport(void) { return medconfig.out_port; }
char *medconf_getoutuser(void) { return medconfig.out_user; }
char *medconf_getoutconnecttimeout(void) { return medconfig.out_connect_timeout; }
char *medconf_getoutpassword(void) { return medconfig.out_password; }
char *medconf_getouttable(void) { return medconfig.out_table; }
char *medconf_getoutorigcol(void) { return medconfig.out_origcol; }
char *medconf_getoutdestcol(void) { return medconfig.out_destcol; }
int   medconf_isclobber(void) { return medconfig.clobber; }

int  *medconf_getetab(void) { return medconfig.edit_tab; }
int   medconf_getnoevals(void) { return medconfig.no_edits; }

/*****************************************************************
 *   Cleanup function
 ****************************************************************/

void  medconf_cleanup(void) {
    free(medconfig.in_dbname);
    free(medconfig.in_hostaddr);
    free(medconfig.in_port);
    free(medconfig.in_user);
    free(medconfig.in_connect_timeout);
    free(medconfig.in_password);
    free(medconfig.in_table);
    free(medconfig.input_col);
    free(medconfig.input_separator);

    free(medconfig.out_dbname);
    free(medconfig.out_hostaddr);
    free(medconfig.out_port);
    free(medconfig.out_user);
    free(medconfig.out_connect_timeout);
    free(medconfig.out_password);
    free(medconfig.out_table);
    free(medconfig.out_origcol);
    free(medconfig.out_destcol);

    free(medconfig.edit_tab);
}

/*****************************************************************
 *   Wrapper function for medconf.lex
 ****************************************************************/
void medconf_configure(void) {
    medconfin = fopen(CONFIG_FILE, "r");
    medconflex();
}
