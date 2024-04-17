from ultralytics import YOLO
from urllib.parse import urlparse
import fiftyone as fo
import fiftyone.zoo as foz
import matplotlib.pyplot as plt
import matplotlib.patches as patches

datasets = foz.load_zoo_dataset('coco-2017', splits=('train', 'validation', 'test'), classes=['person'], progress=True, max_samples=10000)

for sample in datasets:
  if sample.ground_truth == None:
    continue

  detections = [detection for detection in sample.ground_truth.detections if detection.label == "person"]
  sample.ground_truth.detections = detections
  sample.save()

# Export the splits
for split in ['train', 'validation', 'test']:
    split_view = datasets.match_tags(split)
    split_view.export(
        export_dir='./yolov5-coco-datasets',
        dataset_type=fo.types.YOLOv5Dataset,
        label_field='ground_truth',
        split=split,
        classes=['person'],
    )

models = ['yolov8n', 'yolov8s', 'yolov8m']

for model_name in models:
  print("==========")
  print(f"Model: {model_name}")

  print("Model training...")
  model = YOLO(f"{model_name}.pt")
  train_result = model.train(data='./yolov5-coco-datasets/dataset.yaml', epochs=50, imgsz=640, device=0, batch=8, plots=True, seed=18, project=f"./training/{model_name}")
  print("Train result: ", train_result)

  print("Model validations...")
  metrics = model.val(save_json=True)
  print("Metrics: ", metrics)
  print("Mean average precisions: ", metrics.box.maps)
  print("Testing predictions...")

  datatest = ['https://cdn.antaranews.com/cache/1200x800/2023/10/13/Pengendara-Sepeda-Motor-Trotoar-060323-aaa-5.jpg', 'https://img.harianjogja.com/posts/2022/11/14/1117643/jalur-pedestrian-malioboro.jpg', 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/i.KTm08H6tuM/v1/1200x810.jpg', 'https://static.promediateknologi.id/crop/0x0:0x0/0x0/webp/photo/radarjogja/2023/01/web-JOG-Pedestrian-Harus-Sesuai-Fungsinya-FAT-010122.jpg']
  predictions = model.predict(source=datatest)

  for k, p in enumerate(predictions):
    url = datatest[k]
    parsed_url = urlparse(url=url)
    file_name = parsed_url.path.split('/')[-1]
    p.save(f"./training/{model_name}/predicted/{file_name}")
