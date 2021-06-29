"""
Script to create multiple sbash scripts for submitting jobs on SAIL.

Optional headers to add back in:
#SBATCH -t 0-01:00                     # Runtime

Example output:
python train.py --dataset csv --csv_train annotation_train.csv --csv_val annotation_val.csv --csv_classes ../hand-detection-dataset/class_list.csv --learning_rate 0.0001 --batch_size 16 --depth 50 --model_name surgery
"""

train_dataset = 'annotation_train_val.csv'
val_dataset   = 'old_surgery-annotation_test.csv'
csv_classes = '../hand-detection-dataset/class_list.csv'

# for lr in [1e-4, 1e-5, 1e-6]:
for lr in [1e-4, 1e-5]:
    for bs in [2, 4, 8, 16]:
        for depth in [50]:
            for model_name in ['surgery_new_train_val_old_test']:
                name = 'surgery-lr={}-bs={}-d={}-mn={}'.format(lr, bs, depth, model_name)
                fname = './run_scripts/{}.sh'.format(name)
                job_id = '{}'.format(name)
                mem = '51200M' if bs == 16 else '25600M'
                with open(fname, 'w') as rsh:
                    rsh.write('''\
#!/bin/bash
#SBATCH -J {}                          # Job name
#SBATCH -p pasteur                     # Partition to submit to
#SBATCH --nodelist=pasteur[2]
#SBATCH --gres=gpu:1                   # Number of GPUs to use
#SBATCH --mem={}                       # Memory  
#SBATCH -o ./myoutputs/output_{}_%j.o  # File that STDOUT writes to
#SBATCH -e ./myerrors/error_{}_%j.e    # File that STDERR writes to

## Setup environment ##
source /sailhome/mzhang/.bashrc
source activate cv

python train.py \
--dataset csv \
--csv_train {} \
--csv_val {} \
--csv_classes {} \
--learning_rate {} \
--batch_size {} \
--depth {} \
--model_name {}
'''.format(job_id, mem, job_id, job_id, train_dataset, val_dataset, csv_classes, lr, bs, depth, model_name))
