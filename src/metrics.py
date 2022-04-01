import numpy as np
import pandas as pd

class ScoreDataframe():
    # Class to extract scores and relevant metrics from a dataframe
    def __init__(self):
        self.df = pd.DataFrame({'conformation_name': [], 'fold': [], 'empty_cell_1': [], 'empty_cell_2': [], 'score': [], 'time_of_prediction': [], 'target': [], 'epoch': []})
        self.df = self.df.astype({'conformation_name': 'str', 'fold': 'int', 'score': 'float', 'time_of_prediction': 'float', 'target': 'int', 'epoch': 'int'})

    def add_row(self, conformation_name, fold, score, time_of_prediction, target, epoch):
        # Add one result to the dataframe
        if score < 0.0 or score > 1.0:
            raise ValueError('Score must be between 0 and 1')
        
        self.df = pd.concat([self.df, {'conformation_name': conformation_name, 'fold': fold, 'score': score, 'time_of_prediction': time_of_prediction, 'target': target, 'epoch': epoch}], ignore_index=True)
    
    def add_rows(self, conformation_names, folds, scores, time_of_predictions, targets, epoch):
        # Add multiple results to the dataframe
        if len(conformation_names) != len(folds) or len(conformation_names) != len(scores) or len(conformation_names) != len(time_of_predictions) or len(conformation_names) != len(targets):
            raise ValueError('All lists must be of the same length')

        news = {'conformation_name': conformation_names, 'fold': folds, 'score': scores, 'time_of_prediction': time_of_predictions, 'target': targets, 'epoch': [epoch for _ in range(len(conformation_names))]}
        new_df = pd.DataFrame(news)
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def export(self, savepath):
        # Export the dataframe to a csv file
        self.df.to_csv(savepath, header=True)
