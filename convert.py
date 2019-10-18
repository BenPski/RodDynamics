"""
Converts all the markdown files to html using pandoc
"""

from subprocess import call
from os import walk, path

def convert(name):
    # html pages
    if not path.isfile(name+".html") or path.getmtime(name+".md") > path.getmtime(name+".html"):
        text = "pandoc " + name + ".md -t html --standalone --mathjax -o " + name + ".html"
        call(text,shell=True)

    # pdf pages
    if not path.isfile(name+".pdf") or path.getmtime(name+".md") > path.getmtime(name+".pdf"):
        dir, _ = path.split(name)
        text = "pandoc " + name + ".md -t latex --standalone --resource-path=" + dir + " -o " + name + ".pdf"
        call(text,shell=True)

if __name__ == "__main__":
    for root, _, files in walk("Tutorial"):
        for file in files:
            if file.endswith(".md"):
                f = path.join(root,file)
                name, _ = path.splitext(f)
                convert(name)