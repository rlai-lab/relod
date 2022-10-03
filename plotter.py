# Plotting code to generate all the pretty learning curves in our IROS submission.

import pandas

import seaborn as sn
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl


def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 14.0
    plt.rcParams['ytick.labelsize'] = 14.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 3.0
    plt.rcParams['ytick.minor.pad'] = 50.0

    # https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
    plt.rcParams['pdf.fonttype'] = 42   

def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=12, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=12, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    for tick in ax.xaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)


def get_all_plot_rets(basepath, fname_part, seeds=np.arange(0, 5, dtype=int)):
    all_x = []
    all_rets = []
    for seed in seeds:
        fname = basepath + fname_part.format(seed)

        with open(fname) as f:
            episodes = f.readlines()

            bin_rets = []
            rets = []
            x = []
            next_plot_step = x_tick = 5000
            for epi in episodes:
                epi_dict = eval(epi)
                step = epi_dict['step'] + 1
                if step > next_plot_step:
                    if len(bin_rets) > 0:
                        x.append(next_plot_step)
                        rets.append(np.mean(bin_rets))
                        bin_rets = []

                    while next_plot_step < step:
                        next_plot_step += x_tick

                # returns.append(epi_dict['return'])
                bin_rets.append(epi_dict['episode_reward'])

            # df = df.append({'step':step, 'avg_ret':stat.mean(returns), "tol":tol}, ignore_index=True)

        # Hack to ignore last plot point
        # all_x.append(x[:31])
        all_x.append(x[:26]) # For ppo only
        # all_rets.append(rets[:31])
        all_rets.append(rets[:26]) # For ppo only
        #plt.ylim(-1000, 0)

    return all_x, all_rets

def tick_function(X, dt=0.04):
    return ["{:.1f}".format(z*dt/60.) for z in X]

def plot_create_reacher(title):
    setsizes()
    setaxes()

    bp1 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard remote/create 2 visual reacher/"
    bp2 = "/Users/gautham/src/data_rl/remote_onboard/paper results/remote only/create 2 visual reacher/"
    fnp1 = "/create2_visual_reacher_dt=0.045_target_size=0.12_seed={}/train.log"
    fnp2 = "/SAC_async_create2_visual_reacher_dt=0.045_target_size=0.12_seed={}/train.log"

    basepaths = [bp1, bp2]
    fname_parts = [fnp1, fnp2]
    labels = ["Remote-WStation-Local-JNano", "Remote-WStation-Only"]

    seeds = np.arange(0, 5, dtype=int)
    colors = ['tab:blue', 'tab:orange']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for i, (bp, fnp) in enumerate(zip(basepaths, fname_parts)):
        all_x, all_rets = get_all_plot_rets(basepath=bp, fname_part=fnp, seeds=seeds)
        for x, rets in zip(all_x, all_rets):
            ax1.plot(x[:30], rets[:30], color=colors[i], linewidth=0.6, alpha=0.75)
        avg_rets = np.mean(np.array(all_rets), axis=0)
        ax1.plot(all_x[0], avg_rets, label=labels[i], color=colors[i])
    
    # ax1.set_xlim([0, 150000])    
    ax1.legend(loc="best")
    # Set number of tick labels
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6)) 
    # ax.tick_params(axis='x', labelrotation=90)

    all_ax2_ticks = tick_function(all_x[0])
    ax2_ticks = []
    for i in range(0, len(all_ax2_ticks), 4):
        ax2_ticks.append(all_ax2_ticks[i])
    
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
   
    ax2.set_xticklabels([round(x * 0.045/60., 1) for x in ax1.get_xticks()])
    labels = ax1.get_xticks()
    new_labels = [human_format_numbers(k) for k in labels[1:]] 
    ax1.set_xticklabels(new_labels)
    ax2.set_xlabel("Real Experience Time (mins)", fontsize=14, labelpad=10)

    save_path = "/Users/gautham/Pictures/remote-onboard-agent/iros_create_reacher.pdf"
    ax1.grid()
    ax1.set_xlabel('Timesteps', fontsize=14)    
    h = ax1.set_ylabel("Average\nEpisodic\nReturn", labelpad=40, fontsize=14)
    h.set_rotation(0)    
    ax1.set_title(title, fontsize=15, fontweight='bold', pad=10)
    
    # ax1.set_title(title, fontsize=15, pad=10)
    fig.tight_layout()
    plt.savefig(save_path)

    plt.show()

def plot_ur5_reacher(basepaths, fname_parts, colors, labels, save_path='', title='UR5-VisualReacher', order=''):
    setsizes()
    setaxes()

    seeds = np.arange(0, 5, dtype=int)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for i, (bp, fnp) in enumerate(zip(basepaths, fname_parts)):
        all_x, all_rets = get_all_plot_rets(basepath=bp, fname_part=fnp, seeds=seeds)
        for x, rets in zip(all_x, all_rets):
            ax1.plot(x, rets, color=colors[i], linewidth=0.6, alpha=0.75)
        avg_rets = np.mean(np.array(all_rets), axis=0)
        ax1.plot(all_x[0], avg_rets, label=labels[i], color=colors[i])

    plt.locator_params(axis='x', nbins=5)
        
    #get handles and labels
    handles, labels = ax1.get_legend_handles_labels()
    
    #add legend to plot
    if not order:
        order = np.arange(len(handles))

    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
    # ax1.legend(loc="best")
    # ax1.legend(loc="lower right")
    # ax.tick_params(axis='x', labelrotation=90)

    ax1.set_xlim([0, 100000])
    all_ax2_ticks = tick_function(all_x[0])
    ax2_ticks = []
    for i in range(0, len(all_ax2_ticks), 4):
        ax2_ticks.append(all_ax2_ticks[i])
    
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([round(x * 0.04/60., 1) for x in ax1.get_xticks()])
    labels = ax1.get_xticks()
    new_labels = [human_format_numbers(k) for k in labels] 
    # new_labels = [human_format_numbers(k) for k in labels[1:]] 
    ax1.set_xticklabels(new_labels)
    ax2.set_xlabel("Real Experience Time (mins)", fontsize=14, labelpad=10)
    
    ax1.grid()
    ax1.set_xlabel('Timesteps', fontsize=14)
    h = ax1.set_ylabel("Average\nEpisodic\nReturn", labelpad=40, fontsize=14)
    h.set_rotation(0)
    ax1.set_ylim([0, 200])
    # ax1.set_xlim([0, 100000])
    
    ax1.set_title(title, fontsize=15, fontweight='bold', pad=10)
    # ax1.set_title(title, fontsize=15, pad=10)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    plt.show()

def iros_ur5_main_plot():
    bp1 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard remote/UR5/"
    bp2 = "/Users/gautham/src/data_rl/remote_onboard/paper results/remote only/ur5/"    
    bp3 = "/Users/gautham/src/data_rl/remote_onboard/paper results/max=0.08, buffer=16000, batch=64"
    bp4 = "/Users/gautham/src/data_rl/remote_onboard/paper results/paper_local_only/buffer=15000/"
    bp5 = "/Users/gautham/src/data_rl/remote_onboard/paper results/max=0.15, buffer=16000, batch=32"
    bp6 = "/Users/gautham/src/data_rl/remote_onboard/paper results/to compare with yufeng/ours"    
    fnp1 = "/Visual-UR5_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"
    fnp2 = "/Visual-UR5_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"
    fnp3 = "/Visual-UR5_reaching_dt=0.04_bs=64_dim=160_90_{}/train.log"
    fnp4 = "/Visual-UR5_reaching_dt=0.04_bs=64_dim=160_90_{}/train.log"
    fnp5 = "/Visual-UR5_reaching_dt=0.04_bs=32_dim=160_90_{}/train.log"
    fnp6 = "/Visual-UR5_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"

    basepaths = [bp6, bp1, bp2, bp3, bp4]
    fname_parts = [fnp6, fnp1, fnp2, fnp3, fnp4]
    labels = ["Local-WStation-Only", "Remote-WStation-Local-Laptop", "Remote-WStation-Only", "Local-JNanoEm-Only", "Local-Laptop-Only"]
    colors = ["gray", 'tab:blue', 'tab:orange', "tab:green", "tab:purple"]

    save_path = "/Users/gautham/Pictures/remote-onboard-agent/iros_ur5_reacher.pdf"

    plot_ur5_reacher(basepaths=basepaths, fname_parts=fname_parts, colors=colors, labels=labels, save_path=save_path, title="SAC in different modes\non UR5-VisualReacher", order=[1, 2, 3, 4, 0])

def yuan_mahmood_comparison():
    bp1 = "/Users/gautham/src/data_rl/remote_onboard/paper results/to compare with yufeng/yufeng"
    bp2 = "/Users/gautham/src/data_rl/remote_onboard/paper results/to compare with yufeng/ours" 
    bp3 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard remote/UR5/"   
    fnp1 = "/SACv2_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"
    fnp2 = "/Visual-UR5_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"
    fnp3 = "/Visual-UR5_reaching_dt=0.04_bs=128_dim=160_90_{}/train.log"

    basepaths = [bp1, bp2]
    fname_parts = [fnp1, fnp2]
    labels = ["Yuan and Mahmood (2022)", "ReLoD"]
    colors = ["tab:red", "midnightblue"]

    save_path = "/Users/gautham/Pictures/remote-onboard-agent/yufeng_ours_comparison.pdf"

    plot_ur5_reacher(basepaths=basepaths, fname_parts=fname_parts, colors=colors, labels=labels, 
        save_path=save_path, title="SAC using only a local workstation\non UR5-VisualReacher")

def ur5_reacher_all_runs_ppo():
    bp1 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard remote/UR5/ppo"
    bp2 = "/Users/gautham/src/data_rl/remote_onboard/paper results/remote only/ur5/ppo"
    bp3 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard-only/ur5"
    bp4 = "/Users/gautham/src/data_rl/remote_onboard/paper results/onboard remote/UR5/ppo/SAC_buffer=16000, steps=150k"
    fnp1 = "/Visual-UR5_reaching_dt=0.04_bs=4096_dim=160_90_{}/train.log"
    fnp2 = "/Visual-UR5_reaching_dt=0.04_bs=4096_dim=160_90_{}/train.log"
    fnp3 = "/Visual-UR5_reaching_dt=0.04_bs=4096_dim=160_90_{}/train.log"
    fnp4 = "/Visual-UR5_reaching_dt=0.04_bs=64_dim=160_90_{}/train.log"

    basepaths = [bp2, bp1, bp3]
    fname_parts = [fnp2, fnp1, fnp3]
    labels = ["Remote-WStation-Local-Laptop", "Remote-WStation-Only", "Local-Laptop-Only", "SAC-Local-JNanoEm-Only"]
    colors = ['tab:blue', 'tab:orange', "tab:green", "tab:purple"]

    save_path = "/Users/gautham/Pictures/remote-onboard-agent/ur5_reacher_all_runs_ppo.pdf"

    plot_ur5_reacher(basepaths=basepaths, fname_parts=fname_parts, colors=colors, labels=labels, save_path=save_path, title="PPO in different modes\non UR5-VisualReacher")

if __name__ == '__main__':
    # mpl.style.use('seaborn')
    plot_create_reacher(title="SAC in different modes\non Create-Reacher")
    # iros_ur5_main_plot()
    # ur5_reacher_all_runs_ppo()
    # yuan_mahmood_comparison()
