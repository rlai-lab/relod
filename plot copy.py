import seaborn as sn
import matplotlib.pyplot as plt
import statistics as stat
import pandas


df = pandas.DataFrame(columns=["step", "avg_ret"])

int = 2000
for seed in range(0, 1):
    exp = "SAC_async_create2_visual_reacher_dt=0.045_seed=2_target_size=0.2"

    with open("results/"+exp+"/train.log") as f:
        episodes = f.readlines()

        returns = []
        next_plot_step = int
        for epi in episodes: 
            epi_dict = eval(epi)
            step = epi_dict['step'] + 1
            if step > next_plot_step:
                if len(returns) > 0:
                    df = df.append({'step':next_plot_step, 'avg_ret':stat.mean(returns)}, ignore_index=True)
                    returns = []

                while next_plot_step < step:
                    next_plot_step += int

            returns.append(epi_dict['episode_reward'])

    plt.figure()
    sn.lineplot(x="step", y='avg_ret', data=df)
    plt.xlabel('step')
    plt.ylabel('return')
    # plt.title(str(tol)+' buffer size='+str(15000)) 
    plt.savefig('test_'+str(seed)+'.png')
    plt.close()
