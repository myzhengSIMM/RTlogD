# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
from process import run
import sys
sys.path.append("..") 
from torch.utils.data import DataLoader
from utils import  load_model
from utils import get_configure, \
    collate_molgraphs, load_model, predict, load_dataset

# Function to run evaluation for one epoch
def run_an_eval_epoch(smiles_list,args, model, data_loader):
    # Set the model to evaluation mode
    model.eval()
    predictions = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            prediction = predict(args, model, bg)

            # Transform predictions back to original scale
            prediction = prediction.detach().cpu() * args['train_std'].cpu()+args['train_mean'].cpu()
            predictions.append(prediction)
            
        # Concatenate predictions along the specified dimension
        predictions = torch.cat(predictions, dim=0)

        # Prepare output data in a DataFrame
        output_data = {'canonical_smiles': smiles_list}
        if args['task'] is None:
            args['task'] = ['task_{:d}'.format(t) for t in range(1, args['n_tasks'] + 1)]
        else:
            pass

        # Add predictions for each task to the output DataFrame
        for task_id, task_name in enumerate(args['task']):
            output_data[task_name] = predictions[:, task_id]
        df = pd.DataFrame(output_data)

        # Load CSV file and add predictions column
        out=pd.read_csv(args['test_csv_path'])
        out['predict']=round(df['standard_value'],3)

        # Save the updated DataFrame to a new CSV file
        out.to_csv(args['predict_results_csv_path'], index=False)


# Main function for running evaluation
def main(smiles_list,args, exp_config, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'mode':args['mode'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })

    # Update input node and edge features based on featurizer types
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()+2
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    # Load test data and create a DataLoader
    t0 = time.time()
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    
    # Load the pre-trained model
    model = load_model(exp_config).to(args['device'])
    checkpoint = torch.load("final_model/RTlogD/model_pretrain_76.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    model.load_state_dict(checkpoint)

    # Run evaluation for one epoch
    run_an_eval_epoch(smiles_list,args, model, test_loader)

    print('It took {:.4f}s to complete the task'.format(time.time() - t0))

# Entry point of the script   
if __name__ == '__main__':
    import torch
    from utils import setup
    # Load experiment arguments from a pickle file
    with open('final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
    # Load experiment configuration from a JSON file
    with open('final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)

    # Set the device to CPU
    args['device'] = torch.device('cpu')

    # csv path for test data 
    args['test_csv_path'] = "T-data_predictions(chembl32_logD).csv"

    # csv path to put the prediction 
    args['predict_results_csv_path'] = "T-data_predictions(chembl32_logD)_copy.csv"

    # Initialize and set up experiment arguments
    args = setup(args)

    # Load the test set from a CSV file and add placeholder columns
    test_set = pd.read_csv(args['test_csv_path'])
    test_set['logp']=np.nan
    test_set['exp']=np.nan
    test_set['standard_value']=np.nan

    # Extract SMILES list from the test set
    smiles_list=test_set['smiles'].to_list()

    # Load the dataset for testing
    test_set = load_dataset(args,test_set,"test")

    # Get experiment configuration for testing
    exp_config = get_configure(args['model'],"test")

    # Run the main evaluation function
    main(smiles_list,args, exp_config,test_set)
