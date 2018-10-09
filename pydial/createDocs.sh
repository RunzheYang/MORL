#!/usr/bin/sh

DIRECTORY=documentation

# in case target directory for documentation already exists, delete it
if [ -d "$DIRECTORY" ]; then
	rm -r $DIRECTORY
fi

#create target directory
mkdir $DIRECTORY

# create sphinx-based documentation
make html

# copy sphinx-based htmls to target directory
cp -r _build/html/* $DIRECTORY

# copy cued logo
cp Docs/*.png $DIRECTORY/_static/

# create tutorials directory in target dir
#mkdir $DIRECTORY/Docs/tutorials

# write out tutorials as html to target directory
jupyter-nbconvert --to html --output-dir=$DIRECTORY/Docs/tutorials/ Tutorials/*.ipynb

sed -i '' -e 's/In&nbsp;\[&nbsp;\]://' $DIRECTORY/Docs/tutorials/*.html

# copy additional files of tutorials to target dir
cp Tutorials/* $DIRECTORY/Docs/tutorials/
rm -f $DIRECTORY/tutorials/*.ipynb

echo ""
echo "Documentation created. To view, open ${DIRECTORY}/Docs/index.html"
echo ""
