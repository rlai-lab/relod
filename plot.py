import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from statistics import mean

if __name__ == "__main__":
    res_dir = Path(__file__).parent/'backups'
    envs = next(os.walk(res_dir))[1]
    plot_interval = 1200

    for env in envs:
        tasks = next(os.walk(res_dir/env))[1]
        for task in tasks:
            df = pd.DataFrame(columns=["step", "avg_ret", "seed", "timeout"])
            timeouts = next(os.walk(res_dir/env/task))[1]
            for timeout in timeouts:
                seeds = next(os.walk(res_dir/env/task/timeout))[1]
                for seed in seeds:
                    return_folder = res_dir/env/task/timeout/seed/'returns'
                    filename = next(return_folder.glob("*.txt"))

                    with open(filename, 'r') as return_file:
                        epi_steps = [int(float(step)) for step in return_file.readline().split()]
                        returns = [int(float(ret)) for ret in return_file.readline().split()]
            
                    steps = 0
                    end_step = plot_interval
                    rets = []
                    for (i, epi_s) in enumerate(epi_steps):
                        steps += epi_s
                        ret = returns[i]
                        if steps > end_step:
                            if len(rets) > 0:
                                df = df.append({'step':end_step, 'avg_ret':mean(rets), 'seed':seed, 'timeout': timeout}, ignore_index=True) 

                                rets = []
                            while end_step < steps:
                                end_step += plot_interval
                        
                        rets.append(ret)
                
            plt.ylim(-1000, 0)
            
            sns.lineplot(x="step", y='avg_ret', data=df, hue='timeout', palette='bright')
            title = f'{task} {env} learning curves, penalty 20'
            plt.title(title)
            plt.savefig(title+'.png')
            plt.close()