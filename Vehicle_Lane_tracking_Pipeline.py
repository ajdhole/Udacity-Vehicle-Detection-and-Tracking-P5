import Vehicle_Detection_Pipeline
import Lane_Finding
from moviepy.editor import VideoFileClip


#  Following method combines the 2 pipelines of Advanced lane detection and Vehicle tracking.
def combinedProjectPipelines(img):
    p5img = Vehicle_Detection_Pipeline.process_vehicle(img)
    return Lane_Finding.process_image(p5img)

#  Following section creates the video
def createOutVideo():
    white_output = 'project_result_1.mp4'
    #clip1 = VideoFileClip("project_video.mp4")
    clip1 = VideoFileClip("project_video.mp4").subclip(21,35)

    white_clip = clip1.fl_image(combinedProjectPipelines)
    white_clip.write_videofile(white_output, audio=False)
    return white_clip

# Create Video
white_clip = createOutVideo()
