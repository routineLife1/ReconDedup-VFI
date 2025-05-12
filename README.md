# 📖ReconDedup-VFI

## 🔧Installation
```bash
git clone https://github.com/routineLife1/ReconDedup-VFI.git
cd ReconDedup-VFI
pip3 install -r requirements.txt
```
download weights from [Google Drive](https://drive.google.com/file/d/1_PjZCKso1Gpiw4V68wL1CL5kNIAmsqog/view?usp=sharing) and unzip it, put them to ./weights/


The cupy package is included in the requirements, but its installation is optional. It is used to accelerate computation. If you encounter difficulties while installing this package, you can skip it.


## ⚡Usage 
- run the follwing command to finish interpolation
  ```bash
  python infer.py -i [VIDEO] -o [VIDEO_OUTPUT] -t [TIMES] -m [MODEL_TYPE] -mrl [MAX_RECON_LEN] -dr [DUP_RES] -s -st 12 -scale [SCALE]
  # or use the following command to export video at any frame rate
  python infer.py -i [VIDEO] -o [VIDEO_OUTPUT] -fps [OUTPUT_FPS] -mrl [MAX_RECON_LEN] -dr [DUP_RES] -m [MODEL_TYPE] -s -st 12 -scale [SCALE]
  ```
  
 **example(smooth a 23.976fps video with on three/two and interpolate it to 60fps):**

  ```bash
  python infer.py -i E:/MyVideo/01.mkv -o E:/MyVideo/out.mkv -fps 60 -dr 0.95 -mrl 4 -m gmfss -s -st 0.3 -scale 1.0
  ```

**Full Usage**
```bash
Usage: python infer.py -i in_video -o out_video [options]...
       
  -h                   show this help
  -i input             input video path (absolute path of output video)
  -o output            output video path (absolute path of output video)
  -fps dst_fps         target frame rate (default=60)
  -s enable_scdet      enable scene change detection (default Enable)
  -st scdet_threshold  ssim scene detection threshold (default=0.3)
  -hw hwaccel          enable hardware acceleration encode (default Enable) (require nvidia graph card)
  -s scale             flow scale factor (default=1.0), generally use 1.0 with 1080P and 0.5 with 4K resolution
  -m model_type        model type (default=gmfss)
  -mrl max_recon_len   max recon length (default=4)
  -dr dup_res          ssim deduplication threshold (default=0.95)
```

- input accept absolute video file path. Example: E:/input.mp4
- output accept absolute video file path. Example: E:/output.mp4
- dst_fps = target interpolated video frame rate. Example: 60
- enable_scdet = enable scene change detection.
- scdet_threshold = scene change detection threshold. The larger the value, the more sensitive the detection.
- hwaccel = enable hardware acceleration during encoding output video.
- scale = flow scale factor. Decrease this value to reduce the computational difficulty of the model at higher resolutions. Generally, use 1.0 for 1080P and 0.5 for 4K resolution.
- model_type = model type. Currently, gmfss, rife and gimm is supported.
- max_recon_len = max recon length.
- dup_res = ssim deduplication threshold. The larger the value, the more sensitive the deduplication.
