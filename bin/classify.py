#! /usr/bin/env python
from cmdtool import CommandLineTool

class TrainClassifier(CommandLineTool):
    arguments = [Argument('infile', short_opt='-i', type=str, required=True),
        Argument('model_name', type=str, required=True, choices=('SVM', 'RandomForest'), help='name of the classifier'),
        Argument('model_file', type=str, required=True, help='file path for saving the model'),
        Argument('hyperparam', type=str, help='model hyper-parameter in JSON format')
        ]

    def get_model(self, model_name, hyperparam={}):
        if model == 'SVM':
            from sklearn.svm import SVC
            model = SVC(**hyperparam)
        elif model == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**hyperparam)
        return model

    def __call__(self):
        import json
        import h5py

        hyperparam = json.loads(self.hyperparam)
        model = self.get_model(self.model_name, hyperparam)

        fin = h5py.File(self.infile, 'r')
        X_train = fin['X_train'][:]
        y_train = fin['y_train'][:]
        fin.close()
