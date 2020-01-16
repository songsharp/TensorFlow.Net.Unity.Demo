#!/usr/bin/env bash
node index.js
bash tsjs_to_keras.sh
python keras_to_pb.py
bash pb_to_frozen_pb.sh

