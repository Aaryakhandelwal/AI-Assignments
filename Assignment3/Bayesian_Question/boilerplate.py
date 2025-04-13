#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    train_data = pd.read_csv("train_data.csv")
    validation_data = pd.read_csv("validation_data.csv")
    return train_data, validation_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    features = df.columns
    features = sorted(features)
    edges = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            edges.append((features[i], features[j]))
    dag = bn.make_DAG(DAG=edges, verbose=0)
    model = bn.parameter_learning.fit(dag, df)
    bn.plot(model)
    return model


def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    with open ("base_model.pkl", 'rb') as file:
        base_model = pickle.load(file)
    pruned_model = bn.independence_test(base_model, df, prune=True)
    bn.plot(pruned_model)
    return pruned_model


def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    with open("base_model.pkl", 'rb') as file:
        base_model = pickle.load(file)
    optimized_model = bn.structure_learning.fit(df, scoretype='bic', bw_list_method=base_model['model_edges'])
    optimized_model = bn.parameter_learning.fit(optimized_model, df)
    bn.plot(optimized_model)
    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as file:
        pickle.dump(model, file)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

