import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.mixture import GaussianMixture

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Data', 'fitness_dataset_100000.csv'
)

REQUIRED_COLUMNS = [
    'Age', 'Gender', 'Weight(kg)', 'Fitness Goal', 'Calories',
    'Carbs(g)', 'Protein(g)', 'Fats(g)', 'Meal Type', 'Food Item', 'Allergies'
]


def _load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the dataset: {missing}")
    return df


class FoodRecommendation:
    """
    Predicts a user's daily calorie target with XGBoost, then uses a
    Gaussian Mixture Model to cluster foods by macro profile (Calories,
    Carbs, Protein, Fats) so recommendations come from a nutritionally
    coherent cluster instead of a pure random pick.
    """

    def __init__(self):
        self.raw_df = _load_data()
        self.df = self.raw_df.copy()
        self.df['Gender_enc'] = self.df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
        self.df['Goal_enc'] = self.df['Fitness Goal'].map({'Bulking': 1, 'Cutting': 0}).fillna(0)

        self.model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )
        self.gmm = GaussianMixture(n_components=5, random_state=42)
        self._train()

    def _train(self):
        X = self.df[['Age', 'Gender_enc', 'Weight(kg)', 'Goal_enc']]
        y = self.df['Calories']
        self.model.fit(X, y)
        self.gmm.fit(self.df[['Calories', 'Carbs(g)', 'Protein(g)', 'Fats(g)']])
        self.df['Cluster'] = self.gmm.predict(self.df[['Calories', 'Carbs(g)', 'Protein(g)', 'Fats(g)']])

    def _filter_allergies(self, food_items, allergies):
        if not allergies:
            return food_items
        allergy_terms = [a.strip().lower() for a in allergies if a and a.strip()]
        if not allergy_terms:
            return food_items

        def is_safe(allergy_field):
            field = str(allergy_field).lower()
            return not any(term in field for term in allergy_terms)

        safe_items = food_items[food_items['Allergies'].apply(is_safe)]
        # Fall back to the unfiltered set if filtering removes everything,
        # rather than returning an empty meal.
        return safe_items if not safe_items.empty else food_items

    def recommend_food(self, user_input, days=30):
        gender_enc = 1 if str(user_input.get('gender', 'Male')).lower().startswith('m') else 0
        goal_enc = 1 if str(user_input.get('goal', 'Bulking')).lower() == 'bulking' else 0

        X_pred = pd.DataFrame(
            [[user_input.get('age', 25), gender_enc, user_input.get('weight', 70.0), goal_enc]],
            columns=['Age', 'Gender_enc', 'Weight(kg)', 'Goal_enc']
        )
        daily_calories = float(self.model.predict(X_pred)[0])

        diet_type = user_input.get('diet_type', None)
        allergies = user_input.get('allergies', [])

        pool = self.df
        if diet_type:
            pool = pool[pool['Diet Type'].str.lower() == str(diet_type).lower()]
            if pool.empty:
                pool = self.df

        meal_plan = {}
        meal_types = ['Breakfast', 'Lunch', 'Snacks', 'Dinner']

        for day in range(days):
            day_plan = {}
            for meal in meal_types:
                food_items = pool[pool['Meal Type'] == meal]
                food_items = self._filter_allergies(food_items, allergies)

                if food_items.empty:
                    day_plan[meal] = "No suitable food available"
                    continue

                target = daily_calories / len(meal_types)
                cluster_means = food_items.groupby('Cluster')['Calories'].mean()
                best_cluster = (cluster_means - target).abs().idxmin()
                candidates = food_items[food_items['Cluster'] == best_cluster]
                if candidates.empty:
                    candidates = food_items

                pick = candidates.sample(1).iloc[0]
                day_plan[meal] = pick['Food Item']

            meal_plan[f'Day {day + 1}'] = day_plan

        return {'daily_calorie_target': round(daily_calories), 'plan': meal_plan}
