/* 
   A lexer tailored to the alert libraries.
   Need to find a way around the longest match
   first rule to ignore contractions.
   contractions currently treated as whole tokens
*/

%{

#include <stdio.h>
#include "dd.h"

%}

eid     J-(TEXT|ALERT)---([0-9]){10}
token   (-?)([a-zA-Z0-9]+)([,\.'][a-zA-Z0-9]+)?
punc    [-,\.;:/?!#$%&*<>+="[\]\(\)_~\\']
parens  \([^\)]*\)
white   [ \t\n]+
huh     .

%option noyywrap
%option prefix="ddprep"
%option header-file="include/ddlex.h"

%%

{eid}       { dd_addtarg(ddpreptext); }
{token}     { dd_addentry(ddpreptext); }       /* send tokens */
{punc}      ;                                   /* ignore punctuation */
{parens}    ;                                   /* ignore parenthesized content */
{white}     ;                                   /* ignore whitespace */
{huh}       ;                /* there are things I don't know what they are... */
<<EOF>>     return EOF;

%%


