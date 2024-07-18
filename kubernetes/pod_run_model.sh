#!/bin/bash

huggingface-cli login --token $HF_TOKEN
python /finetune_llama2_70b.py
