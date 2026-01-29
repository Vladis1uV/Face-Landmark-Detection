import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET


class FaceLandmarksDataset(Dataset):

    def __init__(self, root_dir="", transform=None):
        self.root_dir = Path(root_dir) / 'ibug_300W_large_face_landmark_dataset'
        xml_file = self.root_dir / 'labels_ibug_300W_train.xml'
   
        tree = ET.parse(xml_file)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        
        for file_node in root[2]:
            img_path = self.root_dir / file_node.attrib['file']
            self.image_filenames.append(img_path)

            self.crops.append(file_node[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(file_node[0][num].attrib['x'])
                y_coordinate = int(file_node[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks, dtype=np.float32)

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_path = self.image_filenames[index]

        image = cv2.imread(str(img_path), 0)
        landmarks = self.landmarks[index]
        
        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks