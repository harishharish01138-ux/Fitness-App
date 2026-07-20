# FitSmart — Smart Fitness & Diet Recommendation System

A Flask web app combining:
- **ML-based recommendations**: XGBoost predicts calorie/rep targets, LightGBM predicts workout intensity,
  and Gaussian Mixture Models cluster foods/exercises so picks come from a coherent nutritional/intensity band.
- **Real-time pose tracking**: MediaPipe + OpenCV count reps live from your webcam (squats, push-ups,
  pull-ups, bicep curls, crunches), streamed into the browser as MJPEG.

## Project layout
```
app.py                     # Flask entry point (routes for plan + live tracker)
Fitness_app/
  Data/fitness_dataset_100000.csv
  app/
    main.py                # CLI entry point (python -m Fitness_app.app.main)
    user_input.py           # CLI prompt / default / web-form parsing
    recommendation.py       # Orchestrates food + workout models
  models/
    food_recommendation.py  # XGBoost + GMM meal planner
    workout_recommendation.py # LightGBM + GMM workout planner
pose/
  posemodule.py             # MediaPipe pose wrapper (landmark detection)
  state.py                  # Thread-safe live rep/calorie counters
  squats.py, push_up.py, pull_up.py, weight_lifting.py, crunches.py
                             # Each exposes generate_frames() -> MJPEG stream generator
templates/                  # Jinja2 HTML (home, plan results, live tracker)
static/css/style.css
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run the web app
```bash
python app.py
```
Then open http://127.0.0.1:5000

- `/` — enter age, gender, weight, goal, diet type, allergies
- `/plan` — view your generated 30-day food + workout plan
- `/exercises` — pick an exercise to track live via webcam
- `/exercise/<name>` — live MJPEG feed with on-screen rep/calorie counter

## Run the recommender from the command line
```bash
cd Fitness_app/app
python main.py
```

## Notes on this build
- The dataset drives both models; `Data/fitness_dataset_100000.csv` must stay in place.
- Live pose tracking needs a webcam and a display-capable OpenCV build (`opencv-python`,
  not `opencv-python-headless`), since MJPEG frames are encoded server-side and streamed to the browser.
- Original reference-video comparison logic (present in the source `pull_up.py`/`push_up.py`) was
  simplified to live-webcam-only counting, since no reference video assets were provided.
