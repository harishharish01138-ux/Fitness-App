def get_default_input():
    """Fallback/demo values."""
    return {
        'age': 25,
        'gender': 'Male',
        'weight': 70.0,
        'goal': 'Bulking',
        'diet_type': 'Non-Vegetarian',
        'allergies': ['Nuts'],
    }


def get_user_input():
    """Interactive CLI prompt (used by main.py)."""
    try:
        return {
            'age': int(input("Enter your age: ")),
            'gender': input("Enter your gender (Male/Female): ").strip(),
            'weight': float(input("Enter your weight in kg: ")),
            'goal': input("Enter your fitness goal (Bulking/Cutting): ").strip(),
            'diet_type': input("Enter your diet preference (Vegetarian/Non-Vegetarian): ").strip(),
            'allergies': [a.strip() for a in input("Enter any allergies (comma separated, or leave blank): ").split(',') if a.strip()],
        }
    except (ValueError, EOFError):
        print("Invalid or no input received, using default demo values instead.")
        return get_default_input()


def parse_form_input(form):
    """Parses a Flask request.form (web form) into the user_input dict."""
    allergies_raw = form.get('allergies', '')
    return {
        'age': int(form.get('age', 25)),
        'gender': form.get('gender', 'Male'),
        'weight': float(form.get('weight', 70.0)),
        'goal': form.get('goal', 'Bulking'),
        'diet_type': form.get('diet_type', 'Non-Vegetarian'),
        'allergies': [a.strip() for a in allergies_raw.split(',') if a.strip()],
    }
