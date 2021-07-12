import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.svm import SVR
from LDA import add_lda_topics
from FK_score import add_fk_scores
plt.style.use("ggplot")


class SVM_model():
    def __init__(self):
        self.svr = SVR()
        self.publisher_score = {}

    def setup(self):
        from sklearn.model_selection import train_test_split, GridSearchCV

        # load data from files and store in a DataFrame
        citations = {}
        with open('Citations-Project/Pub Cumulative Cites.txt', 'r') as citations_file:
            citations_rows = citations_file.readlines()
            for row in citations_rows[1:]:
                comma = row.rfind(',')
                publisher = row[:comma].strip().replace("\n", "").replace("\t", "")
                citations[publisher] = int(row[(comma + 1):-1])

        frequency = {}
        with open('Citations-Project/Pub Frequency.txt', 'r') as frequency_file:
            freq_rows = frequency_file.readlines()
            for row in freq_rows[1:]:
                comma = row.rfind(',')
                publisher = row[:comma].strip().replace("\n", "").replace("\t", "")
                frequency[row[:comma]] = int(row[(comma + 1):-1])

        publisher_score = {}
        for publisher in citations.keys():
            publisher_score[publisher] = citations[publisher] / frequency[publisher]  # citation/frequency
        self.publisher_score = publisher_score
        with open('Citations-Project/data_file/data.csv', 'r') as data_file:
            data_df = pd.read_csv(data_file)
            df_scores = []
            for index, row in data_df.iterrows():
                publisher = row['Publishers'].strip().replace("\n", "").replace("\t", "")
                try:
                    df_scores.append(publisher_score[publisher])
                except:
                    print(publisher)
            data_df['Publisher_Scores'] = df_scores

        # data_df.head()
        df = data_df

        df.drop(["bib_code", 'ArXiv Identifier', 'Title', 'Abstract Text',
                 'Authors', 'Publishers'], axis=1, inplace=True)

        def calculate_age(year):
            """Calculates age based on published year
            Args:
                int: year published

            Returns:
                int: The age based on year published
            """
            todays_year = 2021
            return todays_year - year

        ages = []
        for paper in df["Year Published"]:
            age = calculate_age(paper)
            ages.append(age)

        df["Age of Paper"] = ages

        df.drop(["Year Published", "Published Date"], axis=1, inplace=True)

        X = df[["Length of Title", "Length of Abstract", "Age of Paper", "No. Authors",
                "Abstract FK grade", "Title FK grade", "Topic 1", "Topic 2", "Topic 3",
                "Topic 4", "Topic 5", "Topic 6", "Topic 7", "Topic 8", "Topic 9", "Topic 10", "Topic 11",
                "Topic 12", "Topic 13", "Topic 14", "Topic 15", "Topic 16", "Topic 17", "Topic 18",
                "Topic 19", "Topic 20", "Publisher_Scores"]]
        y = df[["Citation Count"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.01, test_size=0.0025, shuffle=True)

        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
        x_test_vals = X_test.values
        print((y_test.values.transpose()[0]))
        self.svr.fit(X_train.values, y_train.values.transpose()[0])

        y_pred = self.svr.predict(X_test.values)
        print(y_pred)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

        scoring_metrics = ['mean_absolute_error', 'mean_squared_error', 'r2_score', 'explained_variance_score']
        my_scores = []

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ev = explained_variance_score(y_test, y_pred)

        params = [
            {"C": [1e-5, 1e-3, 1e-1, 1, 1e3], "kernel": ['linear']},
            {"C": [1e-5, 1e-3, 1e-2, 1, 1e2], "kernel": ['rbf', 'sigmoid'],
             "gamma": [0.1, 0.3, 0.5, 0.7, 0.9]}
        ]

        svr = sk.svm.SVR()
        grid_search = sk.model_selection.GridSearchCV(estimator=svr, scoring='neg_mean_absolute_error',
                                                      param_grid=params, cv=2)
        grid_search.fit(X_train, y_train.values.ravel())

        new_params = grid_search.best_params_
        new_C = new_params['C']
        # new_gamma = new_params['gamma']
        new_kernel = new_params['kernel']

        gridsearch_pred = grid_search.predict(X_test)

        my_gridsearch_scores = []

        mae = mean_absolute_error(y_test, gridsearch_pred)
        mse = mean_squared_error(y_test, gridsearch_pred)
        r2 = r2_score(y_test, gridsearch_pred)
        ev = explained_variance_score(y_test, gridsearch_pred)

        self.best_svm = sk.svm.SVR(C=new_C, kernel=new_kernel)  # TODO: Set the params with new_params
        self.best_svm.fit(X_train, y_train.values.ravel())

        self.X_test = X_test

    def predict(self, input):
        y_pred = self.best_svm.predict(input)
        return y_pred

if __name__ == "__main__":
    s = SVM_model()
    s.setup()
    print(s.predict(s.X_test))

    df = pd.DataFrame()
    df["Title"] = ["Classification, inference and segmentation of anomalous diffusion with recurrent neural networks"]
    df["Abstract Text"] = ["Countless systems in biology, physics, and finance undergo diffusive dynamics. Many of these systems, including biomolecules inside cells, active matter systems and foraging animals, exhibit anomalous dynamics where the growth of the mean squared displacement with time follows a power law with an exponent that deviates from 1. When studying time series recording the evolution of these systems, it is crucial to precisely measure the anomalous exponent and confidently identify the mechanisms responsible for anomalous diffusion. These tasks can be overwhelmingly difficult when only few short trajectories are available, a situation that is common in the study of non-equilibrium and living systems. Here, we present a data-driven method to analyze single anomalous diffusion trajectories employing recurrent neural networks, which we name RANDI. We show that our method can successfully infer the anomalous exponent, identify the type of anomalous diffusion process, and segment the trajectories of systems switching between different behaviors. We benchmark our performance against the state-of-the art techniques for the study of single short trajectories that participated in the Anomalous Diffusion (AnDi) Challenge. Our method proved to be the most versatile method, being the only one to consistently rank in the top 3 for all tasks proposed in the AnDi Challenge."]
    df["Age of Paper"] = [1]
    df["No. Authors"] = [3]
    df["Length of Title"] = [len("Classification, inference and segmentation of anomalous diffusion with recurrent neural networks")]
    df["Length of Abstract"] = [len("Countless systems in biology, physics, and finance undergo diffusive dynamics. Many of these systems, including biomolecules inside cells, active matter systems and foraging animals, exhibit anomalous dynamics where the growth of the mean squared displacement with time follows a power law with an exponent that deviates from 1. When studying time series recording the evolution of these systems, it is crucial to precisely measure the anomalous exponent and confidently identify the mechanisms responsible for anomalous diffusion. These tasks can be overwhelmingly difficult when only few short trajectories are available, a situation that is common in the study of non-equilibrium and living systems. Here, we present a data-driven method to analyze single anomalous diffusion trajectories employing recurrent neural networks, which we name RANDI. We show that our method can successfully infer the anomalous exponent, identify the type of anomalous diffusion process, and segment the trajectories of systems switching between different behaviors. We benchmark our performance against the state-of-the art techniques for the study of single short trajectories that participated in the Anomalous Diffusion (AnDi) Challenge. Our method proved to be the most versatile method, being the only one to consistently rank in the top 3 for all tasks proposed in the AnDi Challenge.")]
    df["Publisher_Scores"] = [s.publisher_score["Nature"]]
    df = add_fk_scores(df)
    df = add_lda_topics(df, 20, 20)
    X = df[["Length of Title", "Length of Abstract", "Age of Paper", "No. Authors",
            "Abstract FK grade", "Title FK grade", "Topic 1", "Topic 2", "Topic 3",
            "Topic 4", "Topic 5", "Topic 6", "Topic 7", "Topic 8", "Topic 9", "Topic 10", "Topic 11",
            "Topic 12", "Topic 13", "Topic 14", "Topic 15", "Topic 16", "Topic 17", "Topic 18",
            "Topic 19", "Topic 20", "Publisher_Scores"]]
    print(X)
    print(s.predict(X))