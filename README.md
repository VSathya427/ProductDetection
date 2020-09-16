# ProductDetection
## Using the API for Inference

To use your model as a API run the **detect_api.py** script as shown below:

```
python detect_api.py 
Example: python detect_api.py
```

If you have a single GPU, the script will use it by default.
In case you have multiple GPUs you can pass the GPU device number(s) in order to use multiple GPUs or a specific one as shown below:

```
python detect_api.py  <GPU-device-number(s)>
Example: python detect_api.py 1 # to use the GPU with device number '1'
Example: python detect_api.py  0, 1, 2 # to use GPUs with dsimply change theevice values numbers '0', '1' and '2'
```
