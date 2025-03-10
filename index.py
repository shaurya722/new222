import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import argparse,csv,os

class ExerciseTracker:
    def __init__(self, model_path='/home/shaurya/Desktop/new111111/exercise_tracker/yolo11n-pose.pt'):  
        self.model = YOLO(model_path)
        self.people_data = {}
        self.last_positions = {}
        self.frames_since_seen = {}
        self.id_counter = 0
        self.max_distance_threshold = 100
        self.max_frames_missing = 30
        self.frame_log = []
        self.start_time = time.time()


    def get_person_center(self, keypoints):
        """Calculate the center point of a person based on their keypoints"""
        valid_points = []
        # Use shoulder and hip points for center calculation
        for point_idx in [5, 6, 11, 12]:  # right/left shoulder, right/left hip
            if not (keypoints[point_idx][0].item() == 0 and keypoints[point_idx][1].item() == 0):
                valid_points.append((keypoints[point_idx][0].item(), keypoints[point_idx][1].item()))
        
        if not valid_points:
            return None
            
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return (center_x, center_y)

    def assign_id(self, center, keypoints):
        """Assign an ID to a person based on their position"""
        min_distance = float('inf')
        closest_id = None
        
        # Check distance to all known positions
        for person_id, last_pos in self.last_positions.items():
            if self.frames_since_seen[person_id] < self.max_frames_missing:
                distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                if distance < min_distance and distance < self.max_distance_threshold:
                    min_distance = distance
                    closest_id = person_id
        
        # If no close match found, assign new ID
        if closest_id is None:
            closest_id = self.id_counter
            self.id_counter += 1
            self.people_data[closest_id] = {
                'reps': 0,
                'direction': 0,
                'position': 'up'
            }
        
        # Update position and frames counter
        self.last_positions[closest_id] = center
        self.frames_since_seen[closest_id] = 0
        
        return closest_id

    def log_frame_data(self, person_id, frame_number, right_shoulder_angle,left_shoulder_angle,right_elbow_angle,left_elbow_angle):
        current_time = time.time()
        time_elapsed = round(current_time - self.start_time, 2)
        self.frame_log.append([person_id, frame_number, time_elapsed, right_shoulder_angle,left_shoulder_angle,right_elbow_angle,left_elbow_angle])

    def save_csv(self, filename="pushups_data.csv"):
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["person_id", "frame_number", "time_in_seconds", "right_shoulder_angle","left_shoulder_angle","right_elbow_angle","left_elbow_angle"])
            writer.writerows(self.frame_log)
        print(f"CSV saved to: {file_path}")

    def process_frame(self, frame,frame_number):
        """Process a single frame and return the annotated frame"""
        results = self.model(frame, verbose=False)
        
        if len(results) == 0:
            return frame, "No person detected"
        
        people = results[0].keypoints.data
        if len(people) == 0:
            return frame, "No keypoints detected"
        
        # Increment frames_since_seen for all known IDs
        for person_id in self.frames_since_seen:
            self.frames_since_seen[person_id] += 1
        
        try:
            for keypoints in people:
                center = self.get_person_center(keypoints)
                if center is None:
                    continue
                
                # Get or assign ID for this person
                person_id = self.assign_id(center, keypoints)
                
                # Extract keypoints
                right_shoulder = (int(keypoints[5][0]), int(keypoints[5][1]))
                right_elbow = (int(keypoints[7][0]), int(keypoints[7][1]))
                right_wrist = (int(keypoints[9][0]), int(keypoints[9][1]))
                left_shoulder = (int(keypoints[6][0]), int(keypoints[6][1]))
                left_elbow = (int(keypoints[8][0]), int(keypoints[8][1]))
                left_wrist = (int(keypoints[10][0]), int(keypoints[10][1]))
                right_hip = (int(keypoints[11][0]), int(keypoints[11][1]))
                left_hip = (int(keypoints[12][0]), int(keypoints[12][1]))
                
                # Calculate angles
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_elbow)
                left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_elbow)
                right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_wrist)
                left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_wrist)

                # Rep counting logic using person_id
                if right_elbow_angle and left_elbow_angle:
                    person_data = self.people_data[person_id]
                    
                    # # Going down condition
                    if (right_elbow_angle < 140 or left_elbow_angle < 140):
                        if person_data['position'] == 'up':
                            person_data['position'] = 'down'
                            break
                    
                    # Going up condition
                    elif (right_elbow_angle > 160 or left_elbow_angle > 160):
                        if person_data['position'] == 'down':
                            person_data['reps'] += 1.0
                            person_data['position'] = 'up'
                            break

                self.log_frame_data(person_id, frame_number ,right_shoulder_angle,left_shoulder_angle,right_elbow_angle,left_elbow_angle)


                # if right_elbow_angle > 145 and person_data['position'] == 'up':
                #         # Count a rep when hands go from above to below (down position)
                #             person_data['position'] = 'down'
                #             print(f"Rep {person_data['reps']} counted (hands down transition)")
                #             break

                #     # Transition from "down" (hands below) to "up" (hands above)
                # elif right_elbow_angle < 145 and person_data['position'] == 'down'  :
                #             person_data['position'] = 'up'
                #             person_data['reps'] += 1
                #             break

                # Display ID and angles
                cv2.putText(frame, f'ID: {person_id}', (right_shoulder[0] - 30, right_shoulder[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display rep count
                cv2.putText(frame, f'Reps: {self.people_data[person_id]["reps"]}',
                           (right_shoulder[0] - 30, right_shoulder[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Display angles and draw visualization lines
                # [Rest of your visualization code remains the same]
                
                # Show angles
                if right_elbow_angle is not None:
                    cv2.putText(frame, f'{int(right_elbow_angle)}', right_elbow, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_elbow_angle is not None:
                    cv2.putText(frame, f'{int(left_elbow_angle)}', left_elbow, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if right_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(right_shoulder_angle)}', right_shoulder, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(left_shoulder_angle)}', left_shoulder, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if right_hip_angle is not None:
                    cv2.putText(frame, f'{int(right_hip_angle)}', right_hip, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_hip_angle is not None:
                    cv2.putText(frame, f'{int(left_hip_angle)}', left_hip, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw keypoints and connections
                cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
                cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)
                cv2.circle(frame, right_hip, 5, (0, 255, 0), -1)
                # cv2.line(frame, right_shoulder, right_hip, (0, 255, 0), 2)

                cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, left_shoulder, left_elbow, (255, 0, 0), 2)
                cv2.line(frame, left_elbow, left_wrist, (255, 0, 0), 2)
                cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)
                # cv2.line(frame, left_shoulder, left_hip, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing keypoints: {e}")
            
        # Clean up old IDs
        self._cleanup_old_ids()
            
        return frame, None

    def _cleanup_old_ids(self):
        """Remove IDs that haven't been seen for too long"""
        ids_to_remove = []
        for person_id in self.frames_since_seen:
            if self.frames_since_seen[person_id] > self.max_frames_missing:
                ids_to_remove.append(person_id)
        
        for person_id in ids_to_remove:
            self.frames_since_seen.pop(person_id)
            self.last_positions.pop(person_id)
            # Keep the people_data in case they come back

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        if None in (point1, point2, point3):
            return None
            
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
def main(video_path):
    # Create an instance of the ExerciseTracker
    tracker = ExerciseTracker()

    # Open the video file or capture device
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_number = 0

    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = 1200
    frame_height = 1000
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.join(os.getcwd(), "pushup_multiplePerson_output_video1.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        # h,w,k = frame.shape
        
        if not ret:
            print("End of video or error reading frame.")
            break

            return
        frame_number += 1

        
        # Resize frame for better performance (optional)
        frame = cv2.resize(frame, (frame_width, frame_height))
        # frame = cv2.resize(frame, (1080, 720))


        # Process the frame through the tracker
        processed_frame, message = tracker.process_frame(frame,frame_number)
        frame_number += 1

        # Display any messages
        if message:
            cv2.putText(processed_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the processed frame      
        cv2.imshow('Exercise Tracker', processed_frame)
        out.write(processed_frame)


        # Press 'q' to exit the video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close display window
    # tracker.save_csv()
    cap.release()
    out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Exercise Tracker")
    # parser.add_argument('--video', type=str, required=True, help="/home/kranti/Documents/yolov7-object-tracking-main/twoPersonPushpups.mp4")
    # args = parser.parse_args()

    # Run the main function with the provided video path
    main("/home/shaurya/Desktop/new111111/exercise_tracker/media/media/multiplePersonPushups2_bVlmyiQ.mp4")