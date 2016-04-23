Machine Learning tutorials
==========================

Scripts from following
[Machine Learning Recipes with Josh Gordon](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal).

Setup
-----

Installed NumPy, SciPy (used the system packages).
``` bash
sudo apt-get install python3-numpy python3-scipy
```

Created a virtualenv with virtualenvwrapper
``` bash
mkproject --system-site-packages -p python3 ${vitualenv_name}
```

Installed scikit-learn using pip
``` bash
pip install -U scikit-learn
```

For video_2 used ``pydotplus`` as ``pydot`` doesn't work on python3
``` bash
pip install pydotplus
```

External Docs
-------------
* [virtualenvwrapper docs](https://virtualenvwrapper.readthedocs.org/en/latest/)
* [Installing Scikit-learn](http://scikit-learn.org/stable/install.html)
