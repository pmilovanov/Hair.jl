#!/bin/sh


julia --project=~/hair -J ~/.julia/sysimgs/custom1.6.so -e "using LanguageServer, LanguageServer.SymbolServer; runserver()" ~/hair
