/*
 * Wrapper routines for libpq functions that hopefully will 
 * allow libpq to coordinate with lex.
 *
 * These functions are set up to be called from a child
 * pipeline process in main:
 *   alert_setConx   = establish a connection to the db
 *   alert_feed      = feed the results of query to yyin
 *                     and clean up the memory consumed by result
 *   alert_closeConx = close the connection
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpq-fe.h>
#include "alert_sql.h"
#include "med_config.h"



// The following can't be declared in alert_sql.h because
// 1) I don't know how to compile a header file with the special
//    library switches required for libpq-fe.h, so can't #include
//    it in alert_sql.h
// 2) without #including libpq-fe.h in the header, the types will
//    not be recognized
PGconn *conx;
PGconn *writex;
PGresult *result;

void db_setConx(void) {
    /* Establish connection through PGconn *conx */
    char *conx_string;
    asprintf(&conx_string,
            "hostaddr = '%s' \
             port = '%s' \
             dbname = '%s' \
             user = '%s' \
             connect_timeout = '%s' \
             password = '%s'",
             medconf_getinhostaddr(),
             medconf_getinport(),
             medconf_getindbname(),
             medconf_getinuser(),
             medconf_getinconnecttimeout(),
             medconf_getinpassword() );
    //fprintf(stderr, "conx_string = %s\n", conx_string);

    conx = PQconnectdb(conx_string);
    free(conx_string);
    // verify connected before returning
    if (PQstatus(conx) != CONNECTION_OK) {
        fprintf(stderr, "ERROR: failed to establish connection to db.\n");
        PQfinish(conx);
        exit(0);
    }
}

void db_setWritex(void) {
    /* Establish connection through PGconn *conx */
    char *writex_string;
    asprintf(&writex_string,
            "hostaddr = '%s' \
             port = '%s' \
             dbname = '%s' \
             user = '%s' \
             connect_timeout = '%s' \
             password = '%s'",
             medconf_getouthostaddr(),
             medconf_getoutport(),
             medconf_getoutdbname(),
             medconf_getoutuser(),
             medconf_getoutconnecttimeout(),
             medconf_getoutpassword() );
    //fprintf(stderr, "writex_string = %s", writex_string);

    writex = PQconnectdb(writex_string);
    free(writex_string);
    // verify connected before returning
    if (PQstatus(writex) != CONNECTION_OK) {
        fprintf(stderr, "ERROR: failed to establish write connection to db.\n");
        PQfinish(writex);
        exit(0);
    }
    else db_setResultTab();
}

void db_setResultTab(void) {
    if ( medconf_isclobber() ) {
        char *drop_tab;
        asprintf(&drop_tab,
                 "drop table if exists %s;",
                 medconf_getouttable() );
        PQexec(writex, drop_tab);
        free(drop_tab);
    }
    char *create_string;
    asprintf(&create_string,
             "create table %s ( \
              edits        int, \
              %s           varchar(20), \
              %s           varchar(20) \
            );",
            medconf_getouttable(),
            medconf_getoutorigcol(),
            medconf_getoutdestcol() );

    if ( PQresultStatus(PQexec(writex, create_string)) != PGRES_COMMAND_OK ) {
        printf("Unable to set result table in db\n");
    }
    free(create_string);
}

void db_write(int edits, char *o_id, char *d_id) {
    char *write_string;
    asprintf(&write_string, "insert into %s values (%d, '%s', '%s');",
                    medconf_getouttable(), edits, o_id, d_id) ;
    if ( PQresultStatus(PQexec(writex, write_string)) != PGRES_COMMAND_OK ) {
        printf("Trouble executing '%s' in db\n", write_string);
    }
    free(write_string);
}

void db_feed(int filedes) {
    char *query;
    asprintf(&query,
              "select %s, %s from %s;",
              medconf_getinputseparator(),
              medconf_getinputcol(),
              medconf_getintable() );
    result = PQexec(conx, query);
    free(query);

    // check result of query and clean up if not ok
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        fprintf(stderr, "%s%s\n",
                PQresStatus(PQresultStatus(result)),
                PQresultErrorMessage(result)
                );
                PQclear(result);
                PQfinish(conx);
                exit(1);
    }

    // otherwise start feeding strings
    int nrows, ncols, c;
    nrows = PQntuples(result);
    ncols = PQnfields(result);
    fprintf(stderr, "In alert_sql lib, found %d alert entries.\n", nrows);

    char *next_string;
    char *next_id;
    for (c=0; c<nrows; c++) {
        next_id     = PQgetvalue(result, c, 0);
        next_string = PQgetvalue(result, c, 1);
        write(filedes, next_string, strlen(next_string) + 1);
        //printf("alert_feed: writing '%s'\n", next_id);
        write(filedes, next_id, strlen(next_id) + 1);
    }
    // write end of file char to lexer
    char end[2];
    end[0] = (char) EOF;
    end[1] = '\n';
    write(filedes, end, 2);
    PQclear(result);

}
/*
void db_testfeed(int filedes) {
    char *query;
    result = PQexec(conx, TEST_QUERY);

    // check result of query and clean up if not ok
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        fprintf(stderr, "%s%s\n",
                PQresStatus(PQresultStatus(result)),
                PQresultErrorMessage(result)
                );
                PQclear(result);
                PQfinish(conx);
                exit(1);
    }

    // otherwise start feeding strings
    int nrows, ncols, c;
    nrows = PQntuples(result);
    ncols = PQnfields(result);

    char *next_string;
    char *next_id;
    for (c=0; c<nrows; c++) {
        next_id     = PQgetvalue(result, c, 0);
        next_string = PQgetvalue(result, c, 1);
        write(filedes, next_string, strlen(next_string) + 1);
        //printf("alert_feed: writing '%s'\n", next_id);
        write(filedes, next_id, strlen(next_id) + 1);
    }
    // write end of file char to lexer
    char end[2];
    end[0] = (char) EOF;
    end[1] = '\n';
    write(filedes, end, 2);
    PQclear(result);
}
*/
void db_closeConx(void) {  
    PQfinish(conx);
}

void db_closeWritex(void) {  
    PQfinish(writex);
}
