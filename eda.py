import pandas as pd
import matplotlib.pyplot as plt
import os
import torch 
from yaml_config_override import add_arguments
from addict import Dict

import pandas as pd
import matplotlib.pyplot as plt
import os

def eda_on_folds(config):
    # Define the base directory for saving plots
    save_dir = os.path.join( 'results', 'eda')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize accumulators for overall statistics
    overall_gender_counts = pd.Series(dtype=int)
    overall_task_counts = pd.Series(dtype=int)
    overall_task_sex_distribution = pd.DataFrame()
    overall_age_status_distribution = pd.DataFrame()
    
    # Loop over each fold
    for test_fold in range(1, 11):
        # Define paths to test and train files for the current fold
        test_path = f"{config.data.fold_root_path}/TRAIN_TEST_{test_fold}/test.csv"
        train_path = f"{config.data.fold_root_path}/TRAIN_TEST_{test_fold}/train.csv"
        
        # Load test and train data
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)
        
        # Test set analysis
        print(f"Fold {test_fold} - Test Set Analysis:")
        
        # 1. Gender distribution in test set
        gender_counts = test_df['SEX'].value_counts()
        overall_gender_counts = overall_gender_counts.add(gender_counts, fill_value=0)
        
        print("Gender distribution in test set:")
        print(gender_counts)
        
        # Plot and save gender distribution
        gender_counts.plot(kind='bar', color=['skyblue', 'salmon'], title=f"Gender Distribution in Test Set (Fold {test_fold})")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"fold_{test_fold}_gender_distribution.png"))
        plt.close()
        
        # 2. Distribution of speech tasks in test set
        # Map audio_path to task type
        def extract_task(audio_path):
            if 'ddk_analysis' in audio_path.lower():
                return 'ddk_analysis'
            elif 'sentences' in audio_path.lower():
                return 'sentences'
            elif 'read_text' in audio_path.lower():
                return 'read_text'
            return 'other'
        
        test_df['task'] = test_df['audio_path'].apply(extract_task)
        task_counts = test_df['task'].value_counts()
        overall_task_counts = overall_task_counts.add(task_counts, fill_value=0)
        
        print("Distribution of speech tasks in test set:")
        print(task_counts)
        
        # Plot and save task distribution
        task_counts.plot(kind='bar', color='lightgreen', title=f"Speech Task Distribution in Test Set (Fold {test_fold})")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"fold_{test_fold}_task_distribution.png"))
        plt.close()
        
        # 3. Distribution of speech tasks per gender
        task_sex_distribution = test_df.groupby(['SEX', 'task']).size().unstack(fill_value=0)
        overall_task_sex_distribution = overall_task_sex_distribution.add(task_sex_distribution, fill_value=0)
        
        print("Distribution of speech tasks per gender in test set:")
        print(task_sex_distribution)
        
        # Plot and save task distribution by gender
        task_sex_distribution.plot(kind='bar', stacked=True, title=f"Speech Task Distribution by Gender in Test Set (Fold {test_fold})")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"fold_{test_fold}_task_distribution_by_gender.png"))
        plt.close()
        
        print("\n" + "-"*50 + "\n")
        
        # Train set analysis
        print(f"Fold {test_fold} - Train Set Analysis:")
        
        # 4. Age distribution in train set by PD/HC status
        age_status_distribution = train_df.groupby(['status', 'AGE']).size().unstack(fill_value=0)
        overall_age_status_distribution = overall_age_status_distribution.add(age_status_distribution, fill_value=0)
        
        print("Age distribution in train set by PD/HC status:")
        print(age_status_distribution)
        
        # Plot and save age distribution by status
        age_status_distribution.T.plot(kind='bar', stacked=True, title=f"Age Distribution by PD/HC Status in Train Set (Fold {test_fold})")
        plt.ylabel("Count")
        plt.xlabel("Age")
        plt.savefig(os.path.join(save_dir, f"fold_{test_fold}_age_distribution_by_status.png"))
        plt.close()
        
        print("\n" + "="*50 + "\n")
    
    # Overall analysis across all folds
    print("Overall Test Set Analysis Across All Folds:")
    
    # 1. Overall gender distribution
    print("Overall gender distribution in test sets:")
    print(overall_gender_counts)
    overall_gender_counts.plot(kind='bar', color=['skyblue', 'salmon'], title="Overall Gender Distribution in Test Sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "overall_gender_distribution.png"))
    plt.close()
    
    # 2. Overall task distribution
    print("Overall distribution of speech tasks in test sets:")
    print(overall_task_counts)
    overall_task_counts.plot(kind='bar', color='lightgreen', title="Overall Speech Task Distribution in Test Sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "overall_task_distribution.png"))
    plt.close()
    
    # 3. Overall distribution of speech tasks per gender
    print("Overall distribution of speech tasks per gender in test sets:")
    print(overall_task_sex_distribution)
    overall_task_sex_distribution.plot(kind='bar', stacked=True, title="Overall Speech Task Distribution by Gender in Test Sets")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "overall_task_distribution_by_gender.png"))
    plt.close()
    
    # 4. Overall age distribution by status in train sets
    print("Overall age distribution by PD/HC status in train sets:")
    print(overall_age_status_distribution)
    overall_age_status_distribution.T.plot(kind='bar', stacked=True, title="Overall Age Distribution by PD/HC Status in Train Sets")
    plt.ylabel("Count")
    plt.xlabel("Age")
    plt.savefig(os.path.join(save_dir, "overall_age_distribution_by_status.png"))
    plt.close()


# Example usage
# config should contain the data root path as config.data.fold_root_path
# eda_on_folds(config)


def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    eda_on_folds(config)


if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    main(config)
