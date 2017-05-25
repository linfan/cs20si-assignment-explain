For detailed instruction, you should read the assignment handout on the course website:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf

The model have been update to fit for Python 3 and TensorFlow 1.1

## Usage

**Step 1**: Put your content and style images in `content` and `style` folder.

**Step 2**: Change `STYLE` and `CONTENT` variable in `style_transfer.py` to the style and content image name. E.g.
```
STYLE = 'guernica'
CONTENT = 'deadpool'
```

**Step 3**: Run the model !!!
```
python3 style_transfer.py
```

You will get the merged images in `outputs` folder.