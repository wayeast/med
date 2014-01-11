/* 
   A lexer tailored to the alert libraries.
   Need to find a way around the longest match
   first rule to ignore contractions.
   contractions currently treated as whole tokens
*/

%{

#include <stdio.h>
#include "od.h"

%}

eid     J-(TEXT|ALERT)---([0-9]){10}
token   (-?)([a-zA-Z0-9]+)([,\.'][a-zA-Z0-9]+)?
punc    [-,\.;:/?!#$%&*<>+="[\]\(\)_~\\']
parens  \([^\)]*\)
white   [ \t\n]+
huh     .

%option noyywrap
%option prefix="odfeed"
%option header-file="include/odlex.h"

%%

{eid}       { od_setid(odfeedtext); return ODSTAT_ID; }
{token}     { od_addtoken(odfeedtext); return ODSTAT_TOK; }       /* send tokens */
{punc}      return ODSTAT_TOK;                                   /* ignore punctuation */
{parens}    return ODSTAT_TOK;                                   /* ignore parenthesized content */
{white}     return ODSTAT_TOK;                                   /* ignore whitespace */
{huh}       return ODSTAT_TOK;                /* there are things I don't know what they are... */
<<EOF>>     return EOF;

%%


