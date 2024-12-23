import numpy as np
import supervision as sv
from roboflow import Roboflow
from tqdm import tqdm

SOURCE_VIDEO_PATH = "D:\\Projects\\Internship\\video\\video.mp4"
TARGET_VIDEO_PATH = "D:\\Projects\\Internship\\video\\video_out.mp4"

# use https://roboflow.github.io/polygonzone/ to get the points for your line
LINE_START = sv.Point(0, 300)
LINE_END = sv.Point(800, 300)

rf = Roboflow(api_key="g9uehZxbo5yMJeasZnBm")
project = rf.workspace().project("production-line-package-tracking-wi71d")
model = project.version(6).model

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25, 
    lost_track_buffer=30, 
    minimum_matching_threshold=0.8, 
    frame_rate=video_info.fps)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)

# create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    # with tempfile.NamedTemporaryFile(suff?ix=".jpg") as temp:

    results = model.predict(frame).json()

    detections = sv.Detections.from_inference(results)

    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame=box_annotator.annotate(
        scene=annotated_frame,
        detections=detections)

    # update line counter
    line_zone.trigger(detections)
    # return frame with box and line annotated result
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# process the whole video
total_frames = video_info.total_frames

with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        for index, frame in enumerate(generator):
            processed_frame = callback(frame, index)
            pbar.update(1)
            sink.write_frame(frame=processed_frame)