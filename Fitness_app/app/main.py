import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation import generate_plan
from user_input import get_user_input


def main():
    user_input = get_user_input()
    plan = generate_plan(user_input, days=30)

    print(f"\nEstimated daily calorie target: {plan['daily_calorie_target']} kcal\n")

    print("30-Day Food Plan:")
    for day, meals in plan['food_plan'].items():
        print(f"{day}: Breakfast: {meals['Breakfast']}, Lunch: {meals['Lunch']}, "
              f"Snacks: {meals['Snacks']}, Dinner: {meals['Dinner']}")

    print("\n30-Day Workout Plan:")
    for day, workout in plan['workout_plan'].items():
        if 'Rest' in workout:
            print(f"{day}: {workout['Rest']}")
        else:
            print(f"{day}: Target Muscle: {workout['Target Muscle']}")
            for exercise in workout['Exercises']:
                print(f" - Exercise: {exercise['Exercise Name']}, Sets: {exercise['Sets']}, Reps: {exercise['Reps']}")
    print()


if __name__ == "__main__":
    main()
