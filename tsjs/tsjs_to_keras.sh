#!/usr/bin/env bash
tensorflowjs_converter     --input_format tfjs_layers_model  --output_format keras ./model_as_tsjs/model.json ./model_as_keras/model.h5
