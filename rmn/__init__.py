import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from models import densenet121, resmasking_dropout1
# from .version import __version__


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


local_checkpoint_path = "trained_net"

local_prototxt_path = "deploy.prototxt.txt"

local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"


for local_path in [
    ( local_checkpoint_path),
    (local_prototxt_path),
    (local_ssd_checkpoint_path),
]:
    if not os.path.exists(local_path):
        print(f"{local_path} does not exists!")


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


def ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def get_ssd_face_detector():
    ssd_face_detector = cv2.dnn.readNetFromCaffe(
        prototxt=local_prototxt_path,
        caffeModel=local_ssd_checkpoint_path,
    )
    return ssd_face_detector


transform = transforms.Compose(
    transforms=[transforms.ToPILImage(), transforms.ToTensor()]
)

FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

is_cuda = torch.cuda.is_available()

# load configs and set random seed
package_root_dir = os.path.dirname(__file__)
config_path = os.path.join(package_root_dir, "configs/fer2013_config.json")
with open(config_path) as ref:
    configs = json.load(ref)

image_size = (configs["image_size"], configs["image_size"])


def get_emo_model():
    emo_model = resmasking_dropout1(in_channels=3, num_classes=7)
    if is_cuda:
        emo_model.cuda(0)
    state = torch.load(local_checkpoint_path, map_location="cpu")
    emo_model.load_state_dict(state["net"])
    emo_model.eval()
    return emo_model


def convert_to_square(xmin, ymin, xmax, ymax):
    # convert to square location
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    square_length = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
    square_length *= 1.1

    xmin = int(center_x - square_length)
    ymin = int(center_y - square_length)
    xmax = int(center_x + square_length)
    ymax = int(center_y + square_length)
    return xmin, ymin, xmax, ymax

def face_tilt_calc (face_cascade,eye_cascade,ret,frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    x, y, w, h = 0, 0, 0, 0
    for (x, y, w, h) in faces:
        continue
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.circle(frame, (x + int(w * 0.5), y +int(h * 0.5)), 4, (0, 255, 0), -1)
    eyes = eye_cascade.detectMultiScale(gray[y:(y + h), x:(x + w)], 1.1, 4)
    index = 0
    eye_1 = [None, None, None, None]
    eye_2 = [None, None, None, None]
    for (ex, ey, ew, eh) in eyes:
        if index == 0:
            eye_1 = [ex, ey, ew, eh]
        elif index == 1:
            eye_2 = [ex, ey, ew, eh]
        # cv2.rectangle(frame[y:(y + h), x:(x + w)], (ex, ey),(ex + ew, ey + eh), (0, 0, 255), 2)
        index = index + 1
    if (eye_1[0] is not None) and (eye_2[0] is not None):
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
          int(left_eye[1] + (left_eye[3] / 2)))
         
        right_eye_center = (
            int(right_eye[0] + (right_eye[2] / 2)),
          int(right_eye[1] + (right_eye[3] / 2)))
         
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
 
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y / delta_x) 
         
        # Converting radians to degrees
        angle = (angle * 180) / np.pi 
        # print(angle)
    else:
        print('no_face')

    return angle


class RMN:
    def __init__(self, face_detector=True):
        if face_detector is True:
            self.face_detector = get_ssd_face_detector()
        self.emo_model = get_emo_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    @torch.no_grad()
    def detect_emotion_for_single_face_image(self, face_image):
        """
        Params:
        -----------
        face_image : np.ndarray
            a cropped face image

        Return:
        -----------
        emo_label : str
            dominant emotion label

        emo_proba : float 
            dominant emotion proba

        proba_list : list
            all emotion label and their proba
        """
        assert isinstance(face_image, np.ndarray)
        face_image = ensure_color(face_image)
        face_image = cv2.resize(face_image, image_size)
        
        face_image = transform(face_image)
        if is_cuda:
            face_image = face_image.cuda(0)

        face_image = torch.unsqueeze(face_image, dim=0)

        output = torch.squeeze(self.emo_model(face_image), 0)
        proba = torch.softmax(output, 0)
    
        # get dominant emotion
        emo_proba, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        emo_proba = emo_proba.item()
        emo_label = FER_2013_EMO_DICT[emo_idx]
    
        # get proba for each emotion
        proba = proba.tolist()
        proba_list = []
        for emo_idx, emo_name in FER_2013_EMO_DICT.items():
            proba_list.append({emo_name: proba[emo_idx]})

        return emo_label, emo_proba, proba_list
    
    @torch.no_grad()
    def video_demo(self):
        vid = cv2.VideoCapture(0)
        
        while True:
            ret, frame = vid.read()
            if frame is None or ret is not True:
                continue

            try:
                frame = np.fliplr(frame).astype(np.uint8)
                angle = face_tilt_calc(self.face_cascade, self.eye_cascade, ret, frame)
                print(angle)

                # frame_1 = frame
                # (h, w) = frame_1.shape[:2]
                # center = (w / 2, h / 2)
                # rot_angle = -angle
                # scale = 1
                # M = cv2.getRotationMatrix2D(center, rot_angle, scale)
                # frame_1 = cv2.warpAffine(frame, M, (w, h))
                # num_gen = np.random.randint(1000000000)

                # cv2.imwrite(f'delete_pics\{num_gen}.jpg',frame_1)

                
                # (h, w) = frame_1.shape[:2]
                # center = (w / 2, h / 2)
                # M1 = cv2.getRotationMatrix2D(center, rot_angle, scale)
                # frame_1 = cv2.warpAffine(frame, M1, (w, h))

                # cv2.imwrite(f'delete\{num_gen}.jpg',frame_1)

                results = self.detect_emotion_for_single_frame(frame, angle)
                frame = self.draw(frame, results)
                if angle > 10:
                    cv2.putText(frame, 'RIGHT TILT :' + str(int(angle))+' degrees',(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2, cv2.LINE_4)
                elif angle < -10:
                    cv2.putText(frame, 'LEFT TILT :' + str(int(angle))+' degrees',(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2, cv2.LINE_4)
                else:
                    cv2.putText(frame, 'STRAIGHT :', (20, 30),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2, cv2.LINE_4)


                # cv2.rectangle(frame, (1, 1), (220, 25), (223, 128, 255), cv2.FILLED)
                # cv2.putText(frame, f"press q to exit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.imshow("disp", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            except Exception as err:
                print(err)
                continue

        cv2.destroyAllWindows()
    
    @staticmethod
    def draw(frame, results):
        """
        Params:
        ---------
        frame : np.ndarray

        results : list of dict.keys('xmin', 'xmax', 'ymin', 'ymax', 'emo_label', 'emo_proba')

        Returns:
        ---------
        frame : np.ndarray
        """
        for r in results:
            xmin = r["xmin"]
            xmax = r["xmax"]
            ymin = r["ymin"]
            ymax = r["ymax"]
            emo_label = r["emo_label"]
            emo_proba = r["emo_proba"]

            label_size, base_line = cv2.getTextSize(
                f"{emo_label}: 000", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            # draw face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (179, 255, 179), 2)

            cv2.rectangle(
                frame,
                (xmax, ymin + 1 - label_size[1]),
                (xmax + label_size[0], ymin + 1 + base_line),
                (223, 128, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                f"{emo_label} {int(emo_proba * 100)}",
                (xmax, ymin + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

        return frame
   
    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            False, 
            False
        )
        self.face_detector.setInput(blob)
        faces = self.face_detector.forward()

        face_results = []
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence < 0.5:
                continue
            xmin, ymin, xmax, ymax = (faces[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            xmin, ymin, xmax, ymax = convert_to_square(xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
        
            face_results.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })

        return face_results

    @torch.no_grad()
    def detect_emotion_for_single_frame(self, frame, angle):
        gray = ensure_gray(frame)

        results = []
        face_results = self.detect_faces(frame)
        print(f"num faces: {len(face_results)}")
            
        for face in face_results:
            xmin = face["xmin"]
            ymin = face["ymin"]
            xmax = face["xmax"]
            ymax = face["ymax"]
            
            face_image = gray[ymin:ymax, xmin:xmax]

            (h, w) = face_image.shape[:2]
            center = (w / 2, h / 2)
            angle = angle
            scale = 1

            M = cv2.getRotationMatrix2D(center, angle, scale)
            face_image = cv2.warpAffine(face_image, M, (w, h))

            num_gen = np.random.randint(1000000000)
            cv2.imwrite(f'face_frames_0_rot\{num_gen}.jpg',face_image)

            if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                continue
            emo_label, emo_proba, proba_list = self.detect_emotion_for_single_face_image(face_image)
            
            results.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "emo_label": emo_label,
                "emo_proba": emo_proba,
                "proba_list": proba_list
            })
        return results


