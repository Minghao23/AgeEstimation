#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
export PROJECTNAME=AgeEstimationServer
export PROJECTDIR=${basepath}
export PROCESSNAME=${PROJECTDIR}/manage.py
pid=`ps -ef |grep "${PROCESSNAME}" |grep -v "grep" |awk '{print $2}'`
if [[ $pid ]]; then
    kill -9 $pid
    if [[ $? -eq 0 ]];then
       echo "[\033[32mOK\033[0m] The $PROJECTNAME is shut down "
    else
       echo "[\033[31mERROR\033[0m] Fail to stop $PROJECTNAME "
     fi
else
    echo "[\033[31mERROR\033[0m] The $PROJECTNAME process is NOT running"
fi

