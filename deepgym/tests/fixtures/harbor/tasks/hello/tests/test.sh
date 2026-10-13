#!/bin/bash
# Reward 1 only if /app/hello.txt says hello.
if [ "$(cat /app/hello.txt 2>/dev/null)" = "hello" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
