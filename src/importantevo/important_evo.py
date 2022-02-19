import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time


import torch
from torchvision import datasets, transforms

class ImportantEvo:
    """Find important evolution."""

    """
    Constructor
    """
    def __init__(self, args, data_path, model):
        self.args = args
        self.data_path = data_path
        self.model = model

        self.label_to_synset = {}


    """
    A wrapper function called by main.py
    """
    def find_important_evolution(self):
        self.init_setting()
        # self.test_input()


    """
    Initial setting
    """
    def init_setting(self):
        self.get_synset_info()

        data_transform = transforms.Compose([
            transforms.Resize((self.model.input_size, self.model.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ])

        self.training_dataset = datasets.ImageFolder(
            self.data_path.get_path('train_data'), data_transform
        )

        self.class_training_dataset = []
        self.gen_training_dataset_of_class()

        self.data_loader = torch.utils.data.DataLoader(
            self.class_training_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )


    def get_synset_info(self):
        df = pd.read_csv(self.args.data_label_path, sep='\t')
        for synset, label in zip(df['synset'], df['tfrecord_label']):
            self.label_to_synset[int(label) - 1] = synset


    def gen_training_dataset_of_class(self):
        total = len(self.training_dataset)
        unit = int(total / 1000)
        start = max(0, unit * (self.args.label - 1))
        end = min(total, unit * (self.args.label + 1))

        with tqdm(total=(end - start)) as pbar:
            for i in range(start, end):
                img, label = self.training_dataset[i]
                if label == self.args.label:
                    self.class_training_dataset.append([img, label])
                pbar.update(1)


    """
    Get gradient for each layer
    """
    def get_gradients(self):
        for imgs, labels in self.training_data_loader:

            # Send input images and their labels to GPU
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # Forward
            # TODO: Check this
            outputs = self.model(imgs).logits
            loss = outputs[self.args.label]

            # Backward
            loss.backward()

            # Get gradient for each layer
            for layer in self.model:
                print(layer)
                print(layer.grad)
    
    
    """
    Utils
    """
    def test_input(self):
        for batch_idx, (imgs, labels) in enumerate(self.data_loader):
            for idx in range(5):
                img = imgs[idx] * 255
                img = np.einsum('kij->ijk', img)
                file_path = f'img-{idx}.jpg'
                cv2.imwrite(
                    file_path, 
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )

