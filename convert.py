"""
Converts all the markdown files to html using pandoc
"""

from subprocess import call
from os import walk, path

def convert(name):
    text = "pandoc " + name + ".md -t html --standalone --mathjax -o " + name + ".html"
    call(text,shell=True)

if __name__ == "__main__":
    for root, _, files in walk("."):
        for file in files:
            if file.endswith(".md"):
                f = path.join(root,file)
                name, _ = path.splitext(f)
                convert(name)