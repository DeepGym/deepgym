#!/bin/bash
# Full reward needs the agent's hello.txt next to the image's seed.txt.
hello=0
seed=0
[ "$(cat hello.txt 2>/dev/null)" = "hello" ] && hello=1
[ "$(cat seed.txt 2>/dev/null)" = "seed" ] && seed=1
echo "{\"reward\": $((hello * seed)), \"hello\": $hello, \"seed\": $seed}" > /logs/verifier/reward.json
