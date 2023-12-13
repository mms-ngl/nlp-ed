#!/bin/bash

# initial check

if [ "$#" != 1 ]; then
    echo "$# parameters given. Only 1 expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: $(basename "$0") [file to evaluate upon]"
  exit 1
fi

test_path=$1

# delete old docker if exists
docker ps -q --filter "name=nlp-ed" | grep -q . && docker stop nlp-ed
docker ps -aq --filter "name=nlp-ed" | grep -q . && docker rm nlp-ed

# build docker file
docker build . -f Dockerfile -t nlp-ed

# bring model up
docker run -d -p 12345:12345 --name nlp-ed nlp-ed

# perform evaluation
/usr/bin/env python ed/evaluate.py "$test_path"

# stop container
docker stop nlp-ed

# dump container logs
docker logs -t nlp-ed > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm nlp-ed