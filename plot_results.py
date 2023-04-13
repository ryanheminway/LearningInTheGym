# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:02:12 2023

@author: hemin
"""

def plot_results_qdl(df):
    # Reset the index to convert 'Generation' from index to column
    df = df.reset_index()
    
    # Create the plot using seaborn
    sns.lineplot(data=df, x='Episode', y='TotalReward', err_style="band")
    
    # Set labels and title for the plot
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward vs Episode')
    
    # Save the plot as an image file
    plt.savefig('average_total_reward_vs_episode.png')
    
    # Show the plot
    plt.show()
    
    
    # Create the plot using seaborn
    sns.lineplot(data=df, x='Episode', y='Success', err_style="band")
    
    # Set labels and title for the plot
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title('Average Success Rate vs Episode')
    
    # Save the plot as an image file
    plt.savefig('average_success_rate_vs_episode.png')
    
    # Show the plot
    plt.show()
    
    
    
    # Create the plot using seaborn
    sns.lineplot(data=df, x='Episode', y='NumSteps', err_style="band")
    
    # Set labels and title for the plot
    plt.xlabel('Episode')
    plt.ylabel('Average Num Steps')
    plt.title('Average Num Steps vs Episode')
    
    # Save the plot as an image file
    plt.savefig('average_num_steps_vs_episode.png')
    
    # Show the plot
    plt.show()
    


if __name__ == '__main__':
    from pathlib import Path
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    my_df = None
    # Specify the run index
    for run_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        
        # Specify the path to the folder containing the CSV files
        folder_path = Path.cwd()
        
        # Specify the file name pattern
        file_pattern = 'LunarLander-V2SPARSE_QDL_RUN={}.csv'
        
        # Construct the file name with the run index
        file_name = file_pattern.format(run_index)
        file_path = os.path.join(folder_path, file_name)
        
        # Read in the CSV file as a DataFrame
        df = pd.read_csv(file_path)
        
        my_df = pd.concat([df, my_df], ignore_index=True)
        
    plot_results_qdl(my_df)

