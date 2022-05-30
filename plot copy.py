import seaborn as sn
import matplotlib.pyplot as plt
import statistics as stat
import pandas


df = pandas.DataFrame(columns=["step", "avg_ret"])

int = 1000
for seed in range(5, 10):
    exp = "Visual-UR5_reaching_dt=0.04_bs=128_dim=160*90_"+str(seed)

    with open("results/paper/"+exp+"/train.log") as f:
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
