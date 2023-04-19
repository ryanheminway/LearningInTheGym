# -*- coding: utf-8 -*-
"""
Plotting tools to process CSV files representing Pandas DataFrames collected
during training of RL agents using QDL and GA. Uses Seaborn library to create
single and multi-line plots of results. Results are plotted against training
episodes or number of weight updates. 
"""

def plot_results_qdl(df):
    """
    Helper function to plot single-line plots of results collected from QDL 
    experiments. Expects a dataframe with columns ['Episode', 'TotalReward',
                                                   'Success', 'NumSteps'].
    Parameters
    ----------
    df : Pandas Dataframe with results data to plot.
    """    
    # Reset the index to convert 'Episode' from index to column
    df = df.reset_index()
    df_qdl = df

    # Updates happen every 4 steps in an episode. 
    # Episodes are almost always 200 steps for mountaincar 
    df_qdl['Update'] = df_qdl['Episode'] * 50
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='TotalReward', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Total Reward')
    plt.title('[LunarLander-v2] Reward vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_episode_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='TotalReward', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Total Reward')
    plt.title('[LunarLander-v2] Reward vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_updates_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='Success', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Success Rate')
    plt.title('[LunarLander-v2] Success Rate vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_episode_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='Success', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Success Rate')
    plt.title('[LunarLander-v2] Success Rate vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_updates_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='NumSteps', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Number of Steps')
    plt.title('[LunarLander-v2] Steps vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_episode_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='NumSteps', err_style="band", label="QDL")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Number of Steps')
    plt.title('[LunarLander-v2] Steps vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_updates_lunar_qdl.png')
    # Show the plot
    plt.show()
    
    
def plot_results_ga(df):
    """
    Helper function to plot single-line plots of results collected from GA 
    experiments. Expects a dataframe with columns ['Generation', 'TotalReward',
                                                   'Success', 'NumSteps'].
    Parameters
    ----------
    df : Pandas Dataframe with results data to plot.
    """
    df_ga = df
    
    # Create a new column to track episodes. 1000 episodes per generation 
    # because of (10 episodes per individual) * 100 individuals
    df_ga['Episode'] = df_ga['Generation'] * 1000
    
    # One update per individual
    df_ga['Update'] = df_ga['Episode'] / 10
    
    # Reset the index to convert 'Generation' from index to column
    df_ga = df_ga.reset_index()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Episode', y='TotalReward', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Total Reward')
    plt.title('[MountainCar-v0 Sparse] Reward vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_episode_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Update', y='TotalReward', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Total Reward')
    plt.title('[MountainCar-v0 Sparse] Reward vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_updates_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Episode', y='Success', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Success Rate')
    plt.title('[MountainCar-v0 Sparse] Success Rate vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_episode_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Update', y='Success', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Success Rate')
    plt.title('[MountainCar-v0 Sparse] Success Rate vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_updates_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Episode', y='NumSteps', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Number of Steps')
    plt.title('[MountainCar-v0 Sparse] Steps vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_episode_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_ga, x='Update', y='NumSteps', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Number of Steps')
    plt.title('[MountainCar-v0 Sparse] Steps vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_updates_sparse_mountaincar_ga.png')
    # Show the plot
    plt.show()
    
    
def plot_both(df_qdl, df_ga):
    """
    Helper function to plot comparison plots of results collected from QDL and
    GA experiments on the same task.
    
    Parameters
    ----------
    df_qdl : Pandas Dataframe with QDL data to plot. Expected columns are:
            ['Episode', 'TotalReward', 'Success', 'NumSteps']
    df_ga : Pandas Dataframe with GA data to plot. Expected columns are:
            ['Generation', 'TotalReward', 'Success', 'NumSteps']         
    """    
    # Create a new column to track episodes. 1000 episodes per generation 
    # because of (10 episodes per individual) * 100 individuals
    df_ga['Episode'] = df_ga['Generation'] * 1000
    
    # One update per individual
    df_ga['Update'] = df_ga['Episode'] / 10
    # Updates happen every 4 steps in an episode. 
    # Episodes are almost always 200 steps for mountaincar 
    df_qdl['Update'] = df_qdl['Episode'] * 50
    
    # Reset the index to convert 'Generation' from index to column
    df_qdl = df_qdl.reset_index()
    df_ga = df_ga.reset_index()
    
    df_qdl = df_qdl[df_qdl['Update'] < 20000]
    #df_ga = df_ga[df_ga['Episode'] < 20000]
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='TotalReward', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Episode', y='TotalReward', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Total Reward')
    plt.title('[LunarLander-v2 Sparse] Reward vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_episode_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='TotalReward', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Update', y='TotalReward', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Total Reward')
    plt.title('[LunarLander-v2 Sparse] Reward vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_reward_vs_updates_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='Success', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Episode', y='Success', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Success Rate')
    plt.title('[LunarLander-v2 Sparse] Success Rate vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_episode_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='Success', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Update', y='Success', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Success Rate')
    plt.title('[LunarLander-v2 Sparse] Success Rate vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_success_vs_updates_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Episode', y='NumSteps', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Episode', y='NumSteps', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Training Episode')
    plt.ylabel('Average Number of Steps')
    plt.title('[LunarLander-v2 Sparse] Steps vs Training Episode')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_episode_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()
    
    # Create the plot using seaborn
    sns.lineplot(data=df_qdl, x='Update', y='NumSteps', err_style="band", label="QDL")
    sns.lineplot(data=df_ga, x='Update', y='NumSteps', err_style="band", label="GA")
    # Set labels and title for the plot
    plt.xlabel('Updates to Network Weights')
    plt.ylabel('Average Number of Steps')
    plt.title('[LunarLander-v2 Sparse] Steps vs Number of Weight Updates')
    # # Save the plot as an image file
    plt.savefig('avg_steps_vs_updates_sparse_lunar_compare_20k_updates.png')
    # Show the plot
    plt.show()

if __name__ == '__main__':
    from pathlib import Path
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    my_df = None
    my_df_ga = None
    # Specify the run index
    for run_index in [*range(10)]:
        
        # Specify the path to the folder containing the CSV files
        folder_path = Path.cwd()
        
        # Specify the file name pattern
        file_pattern = "results/FinalResults/qdlForLunar/" + 'LunarLander-v2QDL_RUN={}.csv'
        #file_pattern = "results/FinalResults/qdlForCar/" + 'MountainCar-v0QDL_RUN={}.csv'
        #file_pattern_ga = "results/FinalResults/gaForLunar/" + 'LunarLander-v2GARUN={}.csv'
        file_pattern_ga = "results/FinalResults/gaForCar/" + 'MountainCar-v0GARUN={}.csv'
        
        # Construct the file name with the run index
        file_name = file_pattern.format(run_index)
        file_name_ga = file_pattern_ga.format(run_index)
        file_path = os.path.join(folder_path, file_name)
        file_path_ga = os.path.join(folder_path, file_name_ga)
        
        # Read in the CSV file as a DataFrame
        df = pd.read_csv(file_path)
        df_ga = pd.read_csv(file_path_ga)
        
        my_df = pd.concat([df, my_df], ignore_index=True)
        my_df_ga = pd.concat([df_ga, my_df_ga], ignore_index=True)
    
    # Can create single or multi-line plots
    plot_results_qdl(my_df)
    #plot_results_ga(my_df_ga)
    #plot_both(my_df, my_df_ga)

