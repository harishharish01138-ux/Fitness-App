import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.mixture import GaussianMixture

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Data', 'fitness_dataset_100000.csv'
)

MUSCLE_GROUPS = ['Chest', 'Back', 'Legs', 'Shoulders', 'Arms', 'Core']


def _load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


class WorkoutRecommendation:
    """
    Uses LightGBM to predict a suitable rep count for a user (based on
    age/gender/weight/goal + the exercise's muscle group) and a Gaussian
    Mixture Model to cluster exercises by (Sets, Reps) intensity so the
    plan pulls from the intensity band that best matches the prediction.
    """

    def __init__(self):
        self.df = _load_data()
        self._train()

    def _train(self):
        df = self.df.copy()
        df['Gender_enc'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
        df['Goal_enc'] = df['Fitness Goal'].map({'Bulking': 1, 'Cutting': 0}).fillna(0)
        df['Muscle_enc'] = df['Target Muscle'].astype('category').cat.codes
        self.df = df

        X = df[['Age', 'Gender_enc', 'Weight(kg)', 'Goal_enc', 'Muscle_enc']]
        y = df['Reps']

        self.model = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1
        )
        self.model.fit(X, y)

        self.gmm = GaussianMixture(n_components=3, random_state=42)
        self.gmm.fit(df[['Sets', 'Reps']])
        df['Cluster'] = self.gmm.predict(df[['Sets', 'Reps']])

    def recommend_workout(self, user_input, days=30):
        gender_enc = 1 if str(user_input.get('gender', 'Male')).lower().startswith('m') else 0
        goal_enc = 1 if str(user_input.get('goal', 'Bulking')).lower() == 'bulking' else 0
        age = user_input.get('age', 25)
        weight = user_input.get('weight', 70.0)

        muscle_codes = dict(zip(
            self.df['Target Muscle'],
            self.df['Target Muscle'].astype('category').cat.codes
        ))

        workout_plan = {}
        for day in range(1, days + 1):
            if day % 7 == 0:
                workout_plan[f'Day {day}'] = {'Rest': 'Today is a rest day!'}
                continue

            target_muscle = MUSCLE_GROUPS[(day - 1) % len(MUSCLE_GROUPS)]
            exercises = self.df[self.df['Target Muscle'] == target_muscle]

            if exercises.empty:
                workout_plan[f'Day {day}'] = {'Rest': f"No exercises available for {target_muscle}. Rest today!"}
                continue

            muscle_code = muscle_codes.get(target_muscle, 0)
            X_pred = pd.DataFrame([[age, gender_enc, weight, goal_enc, muscle_code]],
                                   columns=['Age', 'Gender_enc', 'Weight(kg)', 'Goal_enc', 'Muscle_enc'])
            predicted_reps = float(self.model.predict(X_pred)[0])

            # Pick the exercise cluster whose mean reps is closest to the prediction
            cluster_means = exercises.groupby('Cluster')['Reps'].mean()
            best_cluster = (cluster_means - predicted_reps).abs().idxmin()
            candidates = exercises[exercises['Cluster'] == best_cluster]
            if candidates.empty:
                candidates = exercises

            n_pick = min(4, len(candidates))
            picked = candidates.sample(n_pick)

            workout_plan[f'Day {day}'] = {
                'Target Muscle': target_muscle,
                'Predicted Reps': round(predicted_reps),
                'Exercises': [
                    {'Exercise Name': row['Exercise'], 'Sets': int(row['Sets']), 'Reps': int(row['Reps'])}
                    for _, row in picked.iterrows()
                ]
            }

        return workout_plan
