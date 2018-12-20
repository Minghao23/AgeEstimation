#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
export PROJECTNAME=AgeEstimationServer
export PROJECTDIR=${basepath}
export PROCESSNAME=${PROJECTDIR}/src/manage.py
export VENVDIR=${PROJECTDIR}/venv

pid=`ps -ef |grep "${PROCESSNAME}" |grep -v "grep" |awk '{print $2}'`
if [[ $pid ]]; then
    echo "[\033[31mERROR\033[0m] The ${PROJECTNAME} process is still running and pid=${pid}"
else
    exec python3 ${PROJECTDIR}/src/manage.py runserver 127.0.0.1:8023
    echo "[\033[32mOK\033[0m] Succeed to start ${PROJECTNAME}"
fi

