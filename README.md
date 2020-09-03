# weapons_yolo3_detectron2_hybrid

A project I worked on optimizing the computational requirements for gun detection in videos by combing the speed of YOLO3 with the accuracy of Masked-RCNN (detectron2). 
It worked, it had better accuracy than YOLO-tiny by itself and was far faster than using detectron2. As YOLO only iterates over and image once, it was used as a filter 
(with lowered detection threshold) after which a frame with a suspected gun would be passed on to detectron2. This would allow for live monitoring and drastically reduced the 
processing time required.

I need to clean up the folder and upload examples of the hybrid approach vs. yolo-tiny only vs. detectron2 only.
