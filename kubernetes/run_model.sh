#!/bin/bash

# Prompt user for pod name
read -p "Enter the name of the pod: " pod_name

kubectl cp finetune_llama2_70b.py "${pod_name}:/"
kubectl cp pod_run_model.sh "${pod_name}:/"

kubectl exec -it "${pod_name}" -- /bin/bash
