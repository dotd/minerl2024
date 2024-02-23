# minerl2024
https://minerl.readthedocs.io/en/latest/


## obs 
"obs" is a dict. The fields are:
- "pov" is numpy.ndarray with size 360, 640, 3

## Actions
https://minerl.readthedocs.io/en/latest/environments/index.html#action-space

```
OrderedDict([('ESC', array(1)),
             ('attack', array(1)),  # 1
             ('back', array(0)),    # S
             ('camera', array([-136.68874 ,   55.120068], dtype=float32)),
             ('drop', array(1)),    # Q
             ('forward', array(0)),     # W
             ('hotbar.1', array(1)),
             ('hotbar.2', array(1)),
             ('hotbar.3', array(0)),
             ('hotbar.4', array(0)),
             ('hotbar.5', array(0)),
             ('hotbar.6', array(0)),
             ('hotbar.7', array(1)),
             ('hotbar.8', array(0)),
             ('hotbar.9', array(0)),
             ('inventory', array(0)),   # E
             ('jump', array(0)),    # SPACE
             ('left', array(1)),    # A
             ('pickItem', array(1)),    # 2
             ('right', array(0)),   # D
             ('sneak', array(1)),   # SHIFT
             ('sprint', array(0)),
             ('swapHands', array(1)),
             ('use', array(1))])    # 3

```

## Some installation problems
To install on ubuntu I needed to install:
```
venv/bin/pip install git+https://github.com/minerllabs/minerl
```


