import threading

_lock = threading.Lock()

counters = {
    'squats': {'count': 0, 'calories': 0.0, 'active': False},
    'pushup': {'count': 0, 'calories': 0.0, 'active': False},
    'pullup': {'count': 0, 'calories': 0.0, 'active': False},
    'biceps': {'count': 0, 'calories': 0.0, 'active': False},
    'crunches': {'count': 0, 'calories': 0.0, 'active': False},
}


def reset(exercise):
    with _lock:
        counters[exercise] = {'count': 0, 'calories': 0.0, 'active': True}


def update(exercise, count, calories):
    with _lock:
        counters[exercise]['count'] = count
        counters[exercise]['calories'] = round(calories, 2)


def stop(exercise):
    with _lock:
        counters[exercise]['active'] = False


def snapshot(exercise):
    with _lock:
        return dict(counters.get(exercise, {'count': 0, 'calories': 0.0, 'active': False}))
