import queue
import threading


class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return True, self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        try:
            self.cap.close()
        except Exception as e:
            print(e)
        try:
            self.cap.release()
        except Exception as e:
            print(e)
