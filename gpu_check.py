import cv2

print("cv2.__version__ =", cv2.__version__)
try:
    cnt = cv2.cuda.getCudaEnabledDeviceCount()
except Exception as e:
    cnt = f"error: {e}"
print("CUDA device count =", cnt)

print("\n=== CUDA build flag in OpenCV build info ===")
try:
    print("CUDA in build info? ->", "CUDA: YES" in cv2.getBuildInformation())
except Exception as e:
    print("getBuildInformation error:", e)
