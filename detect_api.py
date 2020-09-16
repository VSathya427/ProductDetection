import argparse
from werkzeug.utils import secure_filename

import torch.backends.cudnn as cudnn

from flask import Flask, request, render_template ,jsonify
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import sys
import json
import glob
import shutil
from IPython.display import Image, display


app = Flask(__name__)
UPLOAD_FOLDER = r'C:\Users\HP\Desktop\Model_Deployment\ProductDetection\static\uploads'
UPLOAD_FOLDER2 = r'C:\Users\HP\Desktop\Model_Deployment\ProductDetection\inference\images'
#DETECTION_FOLDER = '/home/shubham-sakha/project/Repo/custom_object/FLask/static/detections'
if os.path.exists(UPLOAD_FOLDER):
     shutil.rmtree(UPLOAD_FOLDER)  # delete output folder
os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

#weights = 'yolov5s.pt' if len(sys.argv) == 1 else sys.argv[1]
device_number = '' if len(sys.argv) <=2  else sys.argv[2]
device = torch_utils.select_device(device_number)

#model = attempt_load(weights, map_location=device)  # load FP32 model

ff=""


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():

    if request.method == 'POST':
      f = request.files.get('file')
      # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      filepath2 = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
      print(filepath)
      f.save(filepath)
      ff=filepath
      #get_image(filepath, filename)    
      
    shutil.copy(filepath,r'C:\Users\HP\Desktop\Model_Deployment\ProductDetection\inference\images')  
    json_file='api.json'
    save_img=False
    with open(json_file) as f:
          form_data = json.load(f)
    print(form_data)
    #save_img=False
    #form_data = request.json
    #print(form_data)

    weights = form_data['weights']
    source = form_data['source']
    out = form_data['output']
    imgsz = form_data['imgsz']
    conf_thres = form_data['conf_thres']
    iou_thres = form_data['iou_thres']
    view_img = form_data['view_img']
    save_txt = form_data['save_txt']
    classes = form_data['classes']
    agnostic_nms = form_data['agnostic_nms']
    augment = form_data['augment']
    update = form_data['update']


    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    # Initialize
    # device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                text_file = open("sample.txt", "w+")
                n = text_file.write(s)
                text_file.close()
                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    #return 'DONE'
    #return render_template("index.html", image_loc="static/output/")
    with open('sample.txt', 'r') as f:
      remove(filepath2)
      return render_template("uploaded.html", display_detection = filename, fname = filename,text=f.read())#,remove=remove(filepath))
    #for imageName in glob.glob(r'C:\Users\HP\Desktop\Model_Deployment\Newfolder\yolov5\inference\output/*.jpg'): #assuming JPG
     #  display(Image(filename=imageName))
    #print("\n")
  

def remove(filepath):
         os.remove(filepath)
        
@app.route('/camera', methods=['GET', 'POST'])
def camera():
    # save_img=False
    #form_data = request.json
    #print(form_data)
    json_file='api2.json'
    save_img=False
    with open(json_file) as f:
          form_data = json.load(f)
    print(form_data)

    weights = form_data['weights']
    source = form_data['source']
    out = form_data['output']
    imgsz = form_data['imgsz']
    conf_thres = form_data['conf_thres']
    iou_thres = form_data['iou_thres']
    view_img = form_data['view_img']
    save_txt = form_data['save_txt']
    classes = form_data['classes']
    agnostic_nms = form_data['agnostic_nms']
    augment = form_data['augment']
    update = form_data['update']


    webcam = source == "0" or source.startswith('rtsp') or source.startswith('http')

    # Initialize
    # device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            
            try:  
             if view_img:
                 cv2.imshow(p, im0)
                 if cv2.waitKey(1) == ord('q'):  # q to quit
                     raise StopIteration
            except StopIteration: 
                 cv2.destroyAllWindows()
                 return render_template('index.html') 
                 break
          

                 

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    #return 'DONE'
    #return render_template("index2.html")

if __name__ == '__main__':
    #app.run()
    app.run( port=4000,debug="true")