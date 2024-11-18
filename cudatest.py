from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"