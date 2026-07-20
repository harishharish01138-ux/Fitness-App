import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fitness_app'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fitness_app', 'app'))

from flask import Flask, render_template, request, Response, jsonify

from recommendation import generate_plan
from user_input import parse_form_input, get_default_input

from pose import state as pose_state
from pose.squats import generate_frames as squats_frames
from pose.push_up import generate_frames as pushup_frames
from pose.pull_up import generate_frames as pullup_frames
from pose.weight_lifting import generate_frames as biceps_frames
from pose.crunches import generate_frames as crunches_frames

app = Flask(__name__)
app.config['DEBUG'] = True

EXERCISE_STREAMS = {
    'squats': squats_frames,
    'pushup': pushup_frames,
    'pullup': pullup_frames,
    'biceps': biceps_frames,
    'crunches': crunches_frames,
}
EXERCISE_LABELS = {
    'squats': 'Squats',
    'pushup': 'Push-ups',
    'pullup': 'Pull-ups',
    'biceps': 'Bicep Curls',
    'crunches': 'Crunches',
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/plan', methods=['GET', 'POST'])
def plan():
    if request.method == 'POST':
        user_input = parse_form_input(request.form)
    else:
        user_input = get_default_input()

    result = generate_plan(user_input, days=30)
    return render_template(
        'plan.html',
        user_input=user_input,
        daily_calories=result['daily_calorie_target'],
        food_plan=result['food_plan'],
        workout_plan=result['workout_plan'],
    )


@app.route('/exercises')
def exercises():
    return render_template('exercises.html', exercises=EXERCISE_LABELS)


@app.route('/exercise/<name>')
def exercise_page(name):
    if name not in EXERCISE_STREAMS:
        return "Unknown exercise", 404
    return render_template('exercise.html', name=name, label=EXERCISE_LABELS[name])


@app.route('/video_feed/<name>')
def video_feed(name):
    if name not in EXERCISE_STREAMS:
        return "Unknown exercise", 404
    target = request.args.get('target', type=int)
    return Response(
        EXERCISE_STREAMS[name](target_reps=target),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/count/<name>')
def count(name):
    return jsonify(pose_state.snapshot(name))


if __name__ == '__main__':
    app.run()
