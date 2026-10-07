from collections import deque
import os
import threading
from urllib.request import urlretrieve

import cv2

from utils.common import get_filename


class ImageLoader:
    def __init__(self, init_data=[], num_workers=4):
        self.queue = deque(init_data)
        self.num_workers = num_workers
        self.lock = threading.Lock()
        self.out_queue = []
        self.name_queue = []

    def set_queue(self, data):
        self.queue = deque(data)

    def get_file_names(self):
        return self.name_queue

    def run(self):
        self.out_queue = []
        self.name_queue = []
        threads = []
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self.thread_main)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        return self.out_queue

    def thread_main(self):
        while True:
            self.lock.acquire()
            if not self.queue:
                self.lock.release()
                break
            image_path = self.queue.popleft()
            self.lock.release()

            image = self.read_image(image_path)
            self.lock.acquire()
            self.out_queue.append(image)
            self.name_queue.append(image_path)
            self.lock.release()

    def read_image(self, path):
        if not os.path.exists(path):
            id = get_filename(path)
            url = f"https://images.ygoprodeck.com/images/cards_small/{id}.jpg"
            print(f"FileNotFound: {path}")
            print(f"Try download a image from {url}")
            try:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                urlretrieve(url, path)
            except Exception as e:
                raise e
        return cv2.imread(path)[:, :, ::-1].copy()