#ifndef ALERT_SQL_H
#define ALERT_SQL_H



void db_setConx(void);
void db_setWritex(void);
void db_setResultTab(void);
void db_write(int, char *, char *);
void db_feed(int filedes);
void db_testfeed(int filedes);
void db_closeConx(void);
void db_closeWritex(void);

#endif
