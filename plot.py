import seaborn as sn
import matplotlib.pyplot as plt
import statistics as stat
import pandas
from os import walk

root = 'curiosity/'
_, exps, _ = next(walk(root))
plot_intervals = {
                  "roomba local remote visual reacher": 1600,
                  "ur5 local only async visual reacher": 1000,
                  "ur5 local only sync visual reacher": 1000,
                  "ur5 local remote min time reacher": 1200,
                  "ur5 local remote visual reacher": 1000
                 }

for exp in exps:
    df = pandas.DataFrame(columns=["step", "avg_ret", "seed"])
    _, runs, _ = next(walk(root+exp))
    plot_interval = 5000
    for run in runs:
        if run == 'plots':
            continue

        seed = run[-1]
        with open(root+exp+'/'+run+"/train.log") as f:
            episodes = f.readlines()

        returns = []
        end_step = plot_interval
        for epi in episodes: 
            epi_dict = eval(epi)
            step = epi_dict['step']+1
            if step > end_step:
                if len(returns) > 0:
                    df = df.append({'step':end_step, 'avg_ret':stat.mean(returns), 'seed':seed}, ignore_index=True)
                    returns = []

                while end_step < step:
                    end_step += plot_interval

            returns.append(epi_dict['episode_reward'])
        
        plt.figure()
        sn.lineplot(x="step", y='avg_ret', data=df[df['seed']==seed])
        plt.xlabel('step')
        plt.ylabel('return')
        plt.title(exp)
        plt.savefig(root+exp+'/plots/'+run+'.png')
        plt.close()

    plt.figure()
    sn.lineplot(x="step", y='avg_ret', data=df)
    plt.xlabel('step')
    plt.ylabel('return')
    plt.title(exp)
    plt.savefig(root+exp+'/plots/combined.png')
    plt.close()