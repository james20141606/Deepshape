#! /bin/bash
xargs -t -l $@ -I '{}' bash -c '{}'
