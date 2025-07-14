import cv2
import mediapipe as mp
import numpy as np
import pygame


class FaceCamera:

    FACE_DETECTOR = None
    CAP = None

    @staticmethod
    def init(num_faces):
        # Initialize MediaPipe Face Mesh
        FaceCamera.FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=num_faces,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        # Open webcam
        FaceCamera.CAP = cv2.VideoCapture(0)

    @staticmethod
    def get_frame(draw):
        if not FaceCamera.CAP.isOpened():
            return None

        success, frame = FaceCamera.CAP.read()
        if not success:
            FaceCamera.CAP.release()
            cv2.destroyAllWindows()
            return None, None

        # Process frame
        h, w, _ = frame.shape
        results = FaceCamera.FACE_DETECTOR.process(frame)

        if not results.multi_face_landmarks:
            return None, None

        face_infos = []
        for face_landmarks in results.multi_face_landmarks:
            # Get the pixel coordinates of each face landmark
            xs, ys = [], []
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                xs.append(x)
                ys.append(y)

            # Use this to get the height and the center of the face
            face_height = max(ys) - min(ys)
            face_radius = int(face_height * 0.75)
            center_x = int(sum(xs) / len(xs))
            center_y = int(sum(ys) / len(ys))

            # Draw a circle around the face
            if draw:
                cv2.circle(frame, (center_x, center_y), face_radius, (0, 0, 255), 10)

            # Save this face
            face_infos.append((center_x, center_y, face_radius))

        return frame, face_infos

    @staticmethod
    def get_surfaces():
        face_surfaces = []

        # Get the frame from the camera and make it RGB
        frame, face_infos = FaceCamera.get_frame(False)
        if frame is None:
            return face_surfaces

        for i, face_info in enumerate(face_infos):
            # Crop the frame to just the face part we want for this face
            h, w, _ = frame.shape
            cx, cy, r = face_info
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            frame_cropped = frame[y1:y2, x1:x2]

            # Convert the frame to the right color space (RGB)
            frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)

            # Convert from h,w,d to w,h,d
            frame_array = np.transpose(frame_rgb, (1, 0, 2))

            # Convert to a pygame surface
            surface = pygame.surfarray.make_surface(frame_array).convert_alpha()

            # Mask the surface to only the circle around the face
            mask = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(mask, (255, 255, 255, 255), (r, r), r)
            surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            face_surfaces.append(surface)

        return face_surfaces


if __name__ == "__main__":
    FaceCamera.init(num_faces=1)
    while True:
        frame, frame_infos = FaceCamera.get_frame(True)
        if frame is not None:
            cv2.imshow("Face Camera", frame)

        # Break loop if 'q' is pressed or window is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
