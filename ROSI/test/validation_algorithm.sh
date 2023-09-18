#\bin\bash

conda activate chloe_env
cd Chloe/code

taskset --cpu-list 0-12:1 bash gaussian_test.sh 
taskset --cpu-list 0-12:1 bash outliers_detection_test.sh 