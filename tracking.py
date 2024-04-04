from ultralytics import YOLO
import cv2

# model = YOLO('/home/insomnius/.pyenv/runs/detect/train9/weights/best.pt')
model = YOLO('/home/insomnius/.pyenv/runs/detect/train11/weights/best.pt')

# video_path = './2863232-uhd_3840_2160_30fps.mp4'
video_path = './Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4'
# video_path = '8MP 4K Dahua CCTV System Sample Video - Night Time.mp4'

results = model.track(video_path, persist=True, stream=True, conf=0.25)

max_track_id = 0

cap = cv2.VideoCapture(video_path)
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 25, (int(cap.get(3)),int(cap.get(4))))

for result in results:
    summary = result.summary()
    for s in summary:
      if 'track_id' in s and 'name' in s and s['track_id'] > max_track_id and s['name'] == 'person':
        max_track_id = s['track_id']
    tracked_frame = result.plot()
    output.write(tracked_frame)
    cv2.imshow('frame', tracked_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
print("Tracking video complete...")
print(f"There are {max_track_id} peoples in video")