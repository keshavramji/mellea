#!/bin/bash

dir=$(readlink -ef $(dirname $0))
rm $dir/vllm.log $dir/vllm.err

bash $dir/serve.sh &
vllm_pid=$!

trap "kill $vllm_pid ; wait" EXIT

while sleep 1 ; do
    if grep -q "Application startup complete." $dir/vllm.err
    then
        break
    fi
done

VLLM_TESTS_ENABLED="1" python $dir/test_openai_vllm.py


