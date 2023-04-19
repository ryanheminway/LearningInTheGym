import os
from slurm_util import submit_job

for runIdx in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
    run_name = 'gaLunar'
    command = f'/bin/bash launch.sh {runIdx}'
    job_name = f'{run_name}.run={runIdx}'
    submit_job(
        command=command,
        partition='short',
        duration_hours='24',
        job_name=job_name,
        mem_gb=24,
        n_cpu=1,
        logfile=f'/home/heminway.r/logs/{job_name}.LOG')

            



 

