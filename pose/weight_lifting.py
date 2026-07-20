import cv2
from . import posemodule as pm
from . import state

EXERCISE = 'biceps'
CALORIES_PER_REP = 0.4


def generate_frames(target_reps=None):
    """
    Yields MJPEG-encoded frames with live bicep-curl counting overlaid.
    Counting logic: tracks the vertical gap between wrist (id 17) and
    elbow (id 13).
    """
    state.reset(EXERCISE)
    detector = pm.poseDetector()
    cap = cv2.VideoCapture(0)

    count = 0
    calories = 0.0
    flag = 0

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            img = detector.findPose(img)
            lmlist = detector.getPosition(img, draw=False)

            if len(lmlist) != 0:
                cv2.circle(img, (lmlist[17][1], lmlist[17][2]), 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (lmlist[13][1], lmlist[13][2]), 15, (0, 255, 0), cv2.FILLED)
                y1 = lmlist[17][2]
                y2 = lmlist[13][2]
                length = y2 - y1

                if length >= 0 and flag == 0:
                    flag = 1
                elif length < 0 and flag == 1:
                    flag = 0
                    count += 1
                    calories = CALORIES_PER_REP * count
                    state.update(EXERCISE, count, calories)

            cv2.putText(img, f"Bicep curls: {count}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 100, 255), 3)
            cv2.putText(img, f"Calories: {calories:.1f}", (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 100, 255), 3)

            ok, buffer = cv2.imencode('.jpg', img)
            if not ok:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if target_reps and count >= target_reps:
                break
    finally:
        cap.release()
        state.stop(EXERCISE)
