# RodDynamics
Implementations for the rod dynamics simulations and some documentation on how to use and how it is developed.

To read about the development of the theory and the implementation look in the `Tutorial` directory. The simulations are split into directories to organize the order things are developed in and tested. Since this also documents the thought process behind the developments things get updated and replaced with each iteration in the tutorials. The recommended to use version is in the `python` folder for the python code.

The `convert.py` script converts all the markdown files to html and pdf files and keeps the latex-like formatting for equations. This should be ran already, but just run `python convert.py` in the base directory and the files will be converted using [pandoc](https://pandoc.org/).
