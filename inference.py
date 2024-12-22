from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="g9uehZxbo5yMJeasZnBm"
)

PATH = "D:\\Projects\\Internship\\video\\Задание.mp4"

for file_path, image, prediction in CLIENT.infer_on_stream(PATH, model_id="production-line-package-tracking-wi71d/6"):
    print(prediction)
    pass