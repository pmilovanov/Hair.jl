#!/bin/bash

#-J /sysimgs/custom.so
julia -O2 --project=/app "$@"
