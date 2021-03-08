#!/bin/bash

#-J /sysimgs/custom.so
julia -O3 -t auto --project=/app "$@"
