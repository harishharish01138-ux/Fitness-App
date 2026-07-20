import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.food_recommendation import FoodRecommendation
from models.workout_recommendation import WorkoutRecommendation

_food_recommender = None
_workout_recommender = None


def get_food_recommender():
    global _food_recommender
    if _food_recommender is None:
        _food_recommender = FoodRecommendation()
    return _food_recommender


def get_workout_recommender():
    global _workout_recommender
    if _workout_recommender is None:
        _workout_recommender = WorkoutRecommendation()
    return _workout_recommender


def generate_plan(user_input, days=30):
    """Runs both models and returns a combined 30-day plan."""
    food_result = get_food_recommender().recommend_food(user_input, days=days)
    workout_plan = get_workout_recommender().recommend_workout(user_input, days=days)

    return {
        'daily_calorie_target': food_result['daily_calorie_target'],
        'food_plan': food_result['plan'],
        'workout_plan': workout_plan,
    }
