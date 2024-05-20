# Ontariotech Robomaster2024 v2

The second attempt on running YOLO on the jetson nano

## How does this work?
This works by using the onnx runtime and converting our pytorch model into an onnx file, which should be able to run on the Jetson nano

## Why V2
This is a completely different repository from the original attempt, because we were initially trying to use ROS and ultralytics, but due to the jetson nano being obsoleted, it only supports python 3.6, however Ultralytics needs python 3.7 and higher, making it incompatible, especially since the docker containers available would also only give us python without a working torch version.

ROS was also removed due to the overhead required to run it, as well as the complexity of it, of which there is not enough time to the competition to solve this nicely, as well as the realsense node was far too slow for our needs. a simple ROS node might be added for debugging off of the jetson.





###### Sources
- https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/
