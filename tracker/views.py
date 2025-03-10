from django.http import JsonResponse, FileResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .models import Video
import cv2
import mediapipe as mp
import math
import os
import subprocess
from django.conf import settings

# MediaPipe and OpenCV initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_theta = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_theta)
    return math.degrees(angle)

def get_person_position(landmarks):
    if landmarks:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_angle = calculate_angle([left_shoulder.x, left_shoulder.y], 
                                     [left_elbow.x, left_elbow.y], 
                                     [left_wrist.x, left_wrist.y])
        
        right_angle = calculate_angle([right_shoulder.x, right_shoulder.y], 
                                      [right_elbow.x, right_elbow.y], 
                                      [right_wrist.x, right_wrist.y])

        angle_threshold = 160
        if left_angle < angle_threshold and right_angle < angle_threshold:
            return "down"
        else:
            return "up"
    return None


class PushUpCountView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'video' not in request.FILES:
            return JsonResponse({"error": "No video file found in request."}, status=400)
        
        video_file = request.FILES['video']

        # Create a new Video object to store the uploaded video
        video_record = Video.objects.create(uploaded_video=video_file)

        # Get the path where the video is stored
        uploaded_video_path = video_record.uploaded_video.path

        # Create temporary output path for OpenCV processing
        temp_video_path = os.path.join(os.path.dirname(uploaded_video_path), "temp_" + os.path.basename(uploaded_video_path))
        
        # Create the final output path for the browser-compatible video
        processed_video_path = os.path.join(os.path.dirname(uploaded_video_path), "processed_" + os.path.basename(uploaded_video_path))
        
        # Make sure the processed video has .mp4 extension
        if not processed_video_path.lower().endswith('.mp4'):
            processed_video_path = os.path.splitext(processed_video_path)[0] + '.mp4'

        # Open the input video file
        cap = cv2.VideoCapture(uploaded_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary format
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare to write the temporary output video
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize push-up count and position tracking
        pushup_count = 0
        in_down_position = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                position = get_person_position(results.pose_landmarks.landmark)

                if position == "down" and not in_down_position:
                    in_down_position = True
                elif position == "up" and in_down_position:
                    pushup_count += 1
                    in_down_position = False

                # Draw the pose landmarks on the frame (optional)
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )

            # Draw the push-up count on the frame
            font_scale = 3
            thickness = 4
            color = (0, 255, 0)  # Green color
            
            # Calculate the text size
            text = f"Push-ups: {pushup_count}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Calculate the position so that the text stays within the frame
            text_width, text_height = text_size
            x_pos = 10
            y_pos = 50
            
            # Ensure that the text doesn't go out of bounds on the right side of the frame
            if x_pos + text_width > frame.shape[1]:
                x_pos = frame.shape[1] - text_width - 10  # Leave some padding on the right
            
            # Draw the text on the frame
            cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Write the frame with the text to the output video
            out.write(frame)

        cap.release()
        out.release()

        # Convert the video to a browser-compatible format using FFmpeg
        try:
            # Ensure the final file has .mp4 extension
            if not processed_video_path.endswith('.mp4'):
                processed_video_path = os.path.splitext(processed_video_path)[0] + '.mp4'
                
            # Use FFmpeg to convert the temporary video to a web-compatible format
            subprocess.run([
                'ffmpeg',
                '-i', temp_video_path,          # Input video (the temporary file)
                '-c:v', 'libx264',              # H.264 video codec
                '-crf', '23',                   # Constant Rate Factor (quality setting)
                '-preset', 'medium',            # Encoding speed/compression tradeoff
                '-movflags', '+faststart',      # Optimize for web playback
                '-pix_fmt', 'yuv420p',          # Pixel format for maximum compatibility
                '-profile:v', 'baseline',       # H.264 profile for better compatibility
                '-level', '3.0',                # H.264 level for better compatibility
                '-c:a', 'aac',                  # AAC audio codec
                '-b:a', '128k',                 # Audio bitrate
                '-strict', 'experimental',
                processed_video_path            # Output file
            ], check=True)
            
            # Clean up the temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
        except subprocess.CalledProcessError as e:
            return JsonResponse({"error": f"Failed to convert video: {str(e)}"}, status=500)
        except Exception as e:
            return JsonResponse({"error": f"Error processing video: {str(e)}"}, status=500)

        # Update the Video model with the processed video
        relative_path = os.path.join("processed_videos", os.path.basename(processed_video_path))
        video_record.processed_video.name = relative_path
        video_record.pushup_count = pushup_count
        video_record.save()

        # Return JSON response with the URL to the processed video
        return JsonResponse({
            "message": "Video processed successfully.",
            "pushup_count": pushup_count,
            "processed_video_url": video_record.get_processed_video_url()
        })
    
    