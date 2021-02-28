#!/bin/bash

#-J /sysimgs/custom.so
julia -O3 --project=/app "$@"
