**Installation issues and how to fix them**

1.) ValueError: Failed to initialize Pyglet window with an OpenGL >= 3+ context. If you're logged in via SSH, ensure that you're running your script with vglrun (i.e. VirtualGL). The internal error message was "Cannot connect to "None""
solved:
```
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
```

Find here: https://github.com/muelea/shapy/issues/33

2.) error:pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
solved:
```
pip install pyrender==0.1.43
```

Find here: https://github.com/muelea/shapy/issues/33
