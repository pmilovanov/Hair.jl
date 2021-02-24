#!/bin/bash

julia -O2 --project=/app -J /sysimgs/custom.so "$@"
